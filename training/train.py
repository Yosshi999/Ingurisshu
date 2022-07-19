import pickle

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
from pytorch_lightning import Callback, LightningModule, Trainer
from sklearn.model_selection import train_test_split
import mlflow
import numpy as np

model_rnns = {
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
}

def make_optimizer(params, name, **kwargs):
    return torch.optim.__dict__[name](params, **kwargs)

def squash_packed(x, fn):
    return torch.nn.utils.rnn.PackedSequence(
        fn(x.data), x.batch_sizes, 
        x.sorted_indices, x.unsorted_indices)

class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, words, CHARS, PHONEMES, training=True):
        self.words = words
        self.CHAR_PAD = len(CHARS)
        self.PHONEME_BOS = len(PHONEMES)
        self.PHONEME_EOS = len(PHONEMES)
        self.PHONEME_PAD = len(PHONEMES) + 1
        self.training = training
    def __len__(self):
        return len(self.words)
    def __getitem__(self, idx):
        source, target, _accent = self.words[idx]
        source = torch.LongTensor(source)
        if self.training:
            target = torch.LongTensor(target)
            target_in = torch.cat([torch.LongTensor([self.PHONEME_BOS]), target])
            target_out = torch.cat([target, torch.LongTensor([self.PHONEME_EOS])])
            return source, target_in, target_out
        else:
            return source

    def collate_fn_training(self, data):
        source, target_in, target_out = list(zip(*data))
        return (
            pad_sequence(source, batch_first=False, padding_value=self.CHAR_PAD),
            torch.LongTensor([len(x) for x in source]),
            pad_sequence(target_in, batch_first=False, padding_value=self.PHONEME_PAD),
            torch.LongTensor([len(x) for x in target_in]),
            pad_sequence(target_out, batch_first=False, padding_value=self.PHONEME_PAD),
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
        self.CHARS = data["chars"]
        self.PHONEMES = data["phonemes"]

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
        self.ds_train = NLPDataset(train_words, self.CHARS, self.PHONEMES)
        self.ds_val = NLPDataset(val_words, self.CHARS, self.PHONEMES)
        
    def setup(self, stage=None):
        self.PHONEME_BOS = len(self.PHONEMES)
        self.PHONEME_EOS = len(self.PHONEMES)
        self.PHONEME_PAD = len(self.PHONEMES) + 1
        self.emb_x = nn.Embedding(len(self.CHARS) + 1, self.cfg.model.source_embedding_dim, padding_idx=-1)
        self.emb_y = nn.Embedding(len(self.PHONEMES) + 2, self.cfg.model.target_embedding_dim, padding_idx=-1)
        self.encoder = model_rnns[self.cfg.model.name](
            input_size=self.cfg.model.source_embedding_dim,
            hidden_size=self.cfg.model.hidden_dim,
            num_layers=self.cfg.model.num_layers,
            batch_first=False,
        )
        self.decoder = model_rnns[self.cfg.model.name](
            input_size=self.cfg.model.target_embedding_dim,
            hidden_size=self.cfg.model.hidden_dim,
            num_layers=self.cfg.model.num_layers,
            batch_first=False,
        )
        self.out = nn.Linear(self.cfg.model.hidden_dim, len(self.PHONEMES) + 1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.PHONEME_PAD)

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
        pred, _ = pad_packed_sequence(self.forward(xs, xls, ys_in, yls), batch_first=False)
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
        pred, _ = pad_packed_sequence(self.forward(xs, xls, ys_in, yls), batch_first=False)
        loss = self.loss_fn(torch.flatten(pred, end_dim=1), torch.flatten(ys_out, end_dim=1))
        self.vals.append(loss.cpu().detach().item())
        pred_an = pred.argmax(-1)
        correct = (pred_an == ys_out).cpu().detach().numpy() # (sequence, batch)
        weights = (ys_out != self.PHONEME_PAD).cpu().detach().numpy()
        self.accs += np.average(correct, axis=0, weights=weights).sum()
        self.counts += correct.shape[1]

        pred_eval = self.forward(xs, xls)
        pred_tokens = squash_packed(pred_eval, lambda x: x.argmax(1))
        pred_tokens, token_lengths = pad_packed_sequence(pred_tokens, batch_first=False, padding_value=self.PHONEME_PAD)

        for x, x_len, y_out, y_out_len, y_pred, y_pred_len in zip(xs.t(), xls, ys_out.t(), yls, pred_tokens.t(), token_lengths):
            x = x[:x_len].cpu().detach().numpy()
            y_out = y_out[:y_out_len-1].cpu().detach().numpy()    # last token is EOS
            y_pred = y_pred[:y_pred_len-1].cpu().detach().numpy() # last token is EOS

            if self.cfg.data.reverse_source:
                x = x[::-1]
            xstr = "".join([self.CHARS[i] for i in x])
            y_out = " ".join([self.PHONEMES[i] for i in y_out])
            y_pred = " ".join([self.PHONEMES[i] for i in y_pred])

            self.preds.append(xstr + " / " + y_out + " / " + y_pred)
        return loss
    def on_validation_end(self):
        mlflow.log_metric("val_loss", np.mean(self.vals), self.global_step)
        mlflow.log_metric("val_acc", self.accs / self.counts, self.global_step)
        mlflow.log_text("\n".join(self.preds), f"prediction-{self.global_step:05d}.txt")

    def forward(self, xs, xls, ys_in=None, yls=None):
        """
        Parameters
        ====
        xs: padded LongTensor (sequence_x, batch)
        xls: lengths of xs before padding (batch,)
        ys_in: padded LongTensor with BOS (sequence_y, batch)
        yls: lengths of ys_in before padding (batch,)

        Returns
        ====
        preds: PackedSequence (sequence_pred, batch, len(PHONEMES)+1)
            Logits of phoneme|EOS.
        """
        exs = self.emb_x(xs)
        pexs = pack_padded_sequence(exs, xls.cpu(), batch_first=False, enforce_sorted=False)
        _, hidden_state = self.encoder(pexs)

        if ys_in is None:
            sequences = []
            for batch_index in range(xs.shape[1]):
                if self.cfg.model.name == "LSTM":
                    hidden = hidden_state[0][:, batch_index, :].contiguous(), hidden_state[1][:, batch_index, :].contiguous()
                else:
                    hidden = hidden_state[:, batch_index, :].contiguous()

                ey = self.emb_y(torch.LongTensor([self.PHONEME_BOS]).to(xs.device))
                maximum_length = xls[batch_index] * 4
                seq = []
                for i in range(maximum_length):
                    ey, hidden = self.decoder(ey, hidden)
                    pred = self.out(ey)
                    seq.append(pred)
                    if pred[0].argmax() == self.PHONEME_EOS:
                        break
                    ey = self.emb_y(pred.argmax(1))
                sequences.append(torch.cat(seq))
            return pack_sequence(sequences, enforce_sorted=False)
        else:
            eys = self.emb_y(ys_in)
            peys = pack_padded_sequence(eys, yls.cpu(), batch_first=False, enforce_sorted=False)
            feats, _ = self.decoder(peys, hidden_state)
            preds = squash_packed(feats, self.out)
            return preds

@hydra.main(config_path="conf", config_name="seq2seq", version_base=None)
def main(cfg: DictConfig) -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    model = NLPModule(cfg)
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=cfg.train.epoch,
        logger=False,
    )
    trainer.fit(model)
    mlflow.pytorch.log_model(model, f"model_final")

if __name__ == "__main__":
    main()