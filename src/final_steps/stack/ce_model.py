"""Cross-encoder model definition: a transformer encoder with a linear head over the pooled output, plus tokenization and batching helpers.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.
"""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

NAME = 'intfloat/multilingual-e5-small'


class CrossEncoder(nn.Module):
    """Cross-encoder: transformer encoder with a linear scoring head."""
    def __init__(self):
        """Build the encoder and the scoring head."""
        super().__init__()
        self.enc = AutoModel.from_pretrained(NAME)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Return one logit per pair."""
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1)
        return self.head(pooled).squeeze(-1)


def tokenizer():
    """Load the tokenizer of the base model."""
    return AutoTokenizer.from_pretrained(NAME)


def encode(tok, q, s, max_len):
    """Tokenize a query/candidate text pair to at most max_len tokens."""
    return tok(list(q), list(s), truncation=True, max_length=max_len, padding=False)['input_ids']


def collate(ids_list, pad_id, device):
    """Pad a list of token id lists into input and attention tensors."""
    n = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), n), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    return ids.to(device), att.to(device)
