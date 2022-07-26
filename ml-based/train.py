import pickle

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from pytorch_lightning import Callback, LightningModule, Trainer
from sklearn.model_selection import train_test_split
import mlflow
import numpy as np

from common import CHARS, PHONEMES, CHAR_PAD, PHONEME_BOS, PHONEME_EOS, PHONEME_PAD
from model import Seq2Seq

def make_optimizer(params, name, **kwargs):
    return torch.optim.__dict__[name](params, **kwargs)

def log_nested_params(params: DictConfig, namespace=""):
    for k, v in params.items():
        name = namespace + "." + k
        if isinstance(v, DictConfig):
            log_nested_params(v, name)
        else:
            mlflow.log_param(name, v)

class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, words, training=True):
        self.words = words
        self.training = training
    def __len__(self):
        return len(self.words)
    def __getitem__(self, idx):
        source, target, _accent = self.words[idx]
        source = torch.LongTensor(source)
        if self.training:
            target = torch.LongTensor(target)
            target_in = torch.cat([torch.LongTensor([PHONEME_BOS]), target])
            target_out = torch.cat([target, torch.LongTensor([PHONEME_EOS])])
            return source, target_in, target_out
        else:
            return source

    def collate_fn_training(self, data):
        source, target_in, target_out = list(zip(*data))
        return (
            pad_sequence(source, batch_first=False, padding_value=CHAR_PAD),
            torch.LongTensor([len(x) for x in source]),
            pad_sequence(target_in, batch_first=False, padding_value=PHONEME_PAD),
            torch.LongTensor([len(x) for x in target_in]),
            pad_sequence(target_out, batch_first=False, padding_value=PHONEME_PAD),
            torch.LongTensor([len(x) for x in target_out]),
        )

class NLPModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def prepare_data(self) -> None:
        val_pct: float = self.cfg.data.val_pct
        random_state: int = self.cfg.data.random_state

        with open(self.cfg.preprocess.pickle_filename, "rb") as f:
            data = pickle.load(f)

        words = data["data"]
        if self.cfg.data.reverse_source:
            words_flip = []
            for x, *others in words:
                words_flip.append((x[::-1], *others))
            words = words_flip
        print("loaded {} words.".format(len(words)))
        train_words, val_words, _, _ = train_test_split(
            words,
            range(len(words)),
            test_size=val_pct,
            random_state=random_state)
        print("TRAIN: {} words".format(len(train_words)))
        print("VAL:   {} words".format(len(val_words)))
        self.ds_train = NLPDataset(train_words)
        self.ds_val = NLPDataset(val_words)
        
    def setup(self, stage=None):
        self.model = Seq2Seq(self.cfg)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PHONEME_PAD)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=False,
            collate_fn=self.ds_train.collate_fn_training,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.cfg.val.batch_size,
            num_workers=self.cfg.val.num_workers,
            pin_memory=False,
            collate_fn=self.ds_val.collate_fn_training,
        )

    def configure_optimizers(self):
        return make_optimizer(self.parameters(), **self.cfg.optim)

    def training_step(self, batch, batch_idx):
        xs, xls, ys_in, yls, ys_out, _ = batch
        pred, _ = pad_packed_sequence(self.model.forward(xs, xls, ys_in, yls), batch_first=False)
        loss = self.loss_fn(torch.flatten(pred, end_dim=1), torch.flatten(ys_out, end_dim=1))
        mlflow.log_metric("loss", loss.cpu().detach().item(), self.global_step)
        return loss
    
    def on_validation_start(self):
        self.vals = []
        self.accs = 0
        self.counts = 0
        self.preds = []
    def validation_step(self, batch, batch_idx):
        xs, xls, ys_in, yls, ys_out, _ = batch
        pred, _ = pad_packed_sequence(self.model.forward(xs, xls, ys_in, yls), batch_first=False)
        loss = self.loss_fn(torch.flatten(pred, end_dim=1), torch.flatten(ys_out, end_dim=1))
        self.vals.append(loss.cpu().detach().item())
        pred_an = pred.argmax(-1)
        correct = (pred_an == ys_out).cpu().detach().numpy() # (sequence, batch)
        weights = (ys_out != PHONEME_PAD).cpu().detach().numpy()
        self.accs += np.average(correct, axis=0, weights=weights).sum()
        self.counts += correct.shape[1]

        pred_eval = self.model.forward(xs, xls)
        pred_tokens, token_lengths = self.model.analyze_pred(pred_eval)

        for x, x_len, y_out, y_out_len, y_pred, y_pred_len in zip(xs.t(), xls, ys_out.t(), yls, pred_tokens.t(), token_lengths):
            x = x[:x_len].cpu().detach().numpy()
            y_out = y_out[:y_out_len-1].cpu().detach().numpy()    # last token is EOS
            y_pred = y_pred[:y_pred_len].cpu().detach().numpy()

            if self.cfg.data.reverse_source:
                x = x[::-1]
            xstr = "".join([CHARS[i] for i in x])
            y_out = " ".join([PHONEMES[i] for i in y_out])
            y_pred = " ".join([PHONEMES[i] for i in y_pred if i < len(PHONEMES)])

            self.preds.append(xstr + " / " + y_out + " / " + y_pred)
        return loss
    def on_validation_end(self):
        mlflow.log_metric("val_loss", np.mean(self.vals), self.global_step)
        mlflow.log_metric("val_acc", self.accs / self.counts, self.global_step)
        mlflow.log_text("\n".join(self.preds), f"prediction-{self.global_step:05d}.txt")

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    log_nested_params(cfg)

    model = NLPModule(cfg)
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=cfg.train.epoch,
        logger=False,
    )
    trainer.fit(model)
    mlflow.pytorch.log_model(model.model, f"model_final")

if __name__ == "__main__":
    main()