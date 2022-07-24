import torch
from omegaconf import DictConfig, OmegaConf

from common import *
from model import Seq2Seq

def main(cfg: DictConfig, model_path: str):
    #model = Seq2Seq(cfg)
    #print(list(torch.load(model_path).keys()))
    #model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.eval()

    while True:
        text = input("> ").upper()
        x = [char2id[c] for c in text if c in CHARS]
        if cfg.data.reverse_source:
            x = x[::-1]

        with torch.no_grad():
            xs, xls = torch.LongTensor([x]).t(), torch.LongTensor([len(x)])
            pred_eval = model.forward(xs, xls)
            pred_tokens, token_lengths = model.analyze_pred(pred_eval)

            y = pred_tokens[:token_lengths[0], 0].cpu().detach().numpy()
            y = [i for i in y if i < len(PHONEMES)]

            # to katakana (arpabet)
            print(to_kana(y), "(%s)" % decode_phonemes(y))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str)
    parser.add_argument("model", type=str)
    args = parser.parse_args()

    main(OmegaConf.load(args.cfg), args.model)