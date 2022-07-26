from typing import List
from unicodedata import bidirectional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence

from common import CHARS, PHONEMES, CHAR_PAD, PHONEME_BOS, PHONEME_EOS, PHONEME_PAD

model_rnns = {
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
}

def squash_packed(x, fn):
    return torch.nn.utils.rnn.PackedSequence(
        fn(x.data), x.batch_sizes, 
        x.sorted_indices, x.unsorted_indices)


class Seq2Seq(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb_x = nn.Embedding(len(CHARS) + 1, self.cfg.model.source_embedding_dim, padding_idx=-1)
        self.emb_y = nn.Embedding(len(PHONEMES) + 2, self.cfg.model.target_embedding_dim, padding_idx=-1)
        self.encoder = model_rnns[self.cfg.model.name](
            input_size=self.cfg.model.source_embedding_dim,
            hidden_size=self.cfg.model.hidden_dim,
            num_layers=self.cfg.model.num_layers,
            batch_first=False,
            bidirectional=self.cfg.model.bidirectional,
        )

        D = 2 if self.cfg.model.bidirectional else 1
        self.decoder = model_rnns[self.cfg.model.name](
            input_size=self.cfg.model.target_embedding_dim,
            hidden_size=self.cfg.model.hidden_dim * D,
            num_layers=self.cfg.model.num_layers,
            batch_first=False,
        )
        self.out = nn.Linear(self.cfg.model.hidden_dim * D, len(PHONEMES) + 1)

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
        if self.cfg.model.name == "LSTM":
            if self.cfg.model.bidirectional:
                hidden_state = torch.cat([hidden_state[0][0:1], hidden_state[0][1:2]], dim=-1), torch.cat([hidden_state[1][0:1], hidden_state[1][1:2]], dim=-1)
        elif self.cfg.model.name == "GRU":
            if self.cfg.model.bidirectional:
                hidden_state = torch.cat([hidden_state[0:1], hidden_state[1:2]], dim=-1)

        if ys_in is None:
            sequences = []
            for batch_index in range(xs.shape[1]):
                if self.cfg.model.name == "LSTM":
                    hidden = hidden_state[0][:, batch_index, :].contiguous(), hidden_state[1][:, batch_index, :].contiguous()
                else:
                    hidden = hidden_state[:, batch_index, :].contiguous()

                ey = self.emb_y(torch.LongTensor([PHONEME_BOS]).to(xs.device))
                maximum_length = xls[batch_index] * 4
                seq = []
                for i in range(maximum_length):
                    ey, hidden = self.decoder(ey, hidden)
                    pred = self.out(ey)
                    seq.append(pred)
                    if pred[0].argmax() == PHONEME_EOS:
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
        
    def analyze_pred(self, preds):
        pred_tokens = squash_packed(preds, lambda x: x.argmax(1))
        pred_tokens, token_lengths = pad_packed_sequence(pred_tokens, batch_first=False, padding_value=PHONEME_PAD)
        return pred_tokens, token_lengths
