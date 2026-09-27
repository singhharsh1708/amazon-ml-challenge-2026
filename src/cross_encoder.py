"""Cross-encoder model and scoring helpers.

A cross-encoder reads the query text and the source 1 text together ("name | address" on each
side) with a pretrained multilingual transformer (default intfloat/multilingual-e5-small),
mean-pools the last hidden layer and maps it to one match logit with a linear head. Library
module used by train_cross_encoder.py, score_cross_encoder.py and score_rescue.py.
"""

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_LEN = 128
SCORE_BATCH = 256


class CrossEncoder(nn.Module):
    """Transformer encoder with mean pooling and a linear head that outputs one match logit per pair."""
    def __init__(self, name=MODEL_NAME):
        """Load the pretrained encoder and create the linear scoring head."""
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Return one logit per input pair from token ids and the attention mask."""
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        return self.head(pooled).squeeze(-1)


def tokenizer(name=MODEL_NAME):
    """Load the tokenizer for a model name."""
    return AutoTokenizer.from_pretrained(name)


def device(kind="auto"):
    """Return the requested torch device, or the best available one (cuda, mps, cpu) for auto."""
    if kind != "auto":
        return torch.device(kind)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def half_precision(dev):
    """Return True when float16 autocast should be used on the device."""
    return dev.type in ("cuda", "mps")


def encode(tok, q, s, max_len=MAX_LEN):
    """Tokenise query and source 1 texts as sentence pairs; returns unpadded token id lists."""
    return tok(list(q), list(s), truncation=True, max_length=max_len, padding=False)["input_ids"]


def collate(ids_list, pad_id, dev):
    """Pad a list of token id lists into id and attention-mask tensors on the device."""
    n = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), n), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    return ids.to(dev), att.to(dev)


def load(weights, dev, name=MODEL_NAME):
    """Build a CrossEncoder, load saved weights and return it in eval mode on the device."""
    model = CrossEncoder(name)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(dev).eval()


def score(model, tok, ids, dev, batch_size=SCORE_BATCH, log_every=0):
    """Score token id lists in length-sorted batches and return float32 logits in input order."""
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.zeros(len(ids), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            idx = order[i:i + batch_size]
            x, a = collate([ids[j] for j in idx], tok.pad_token_id, dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=half_precision(dev)):
                out[idx] = model(x, a).float().cpu().numpy()
            if log_every and (i // batch_size) % log_every == 0:
                print(f"scored {i + len(idx):,}/{len(ids):,} pairs", flush=True)
    return out
