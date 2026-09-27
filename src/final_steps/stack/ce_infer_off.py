"""Variant of ce_infer.py that scores a slice of a pairs file given an offset.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Command-line arguments used: argv[1], argv[2], argv[3], argv[4].
"""
import sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from ce_model import CrossEncoder, tokenizer, encode, collate, NAME
from torch import nn
from transformers import AutoConfig, AutoModel


class OfflineCrossEncoder(CrossEncoder):
    """Cross-encoder: transformer encoder with a linear scoring head."""
    def __init__(self):
        """Build the encoder and the scoring head."""
        nn.Module.__init__(self)
        self.enc = AutoModel.from_config(AutoConfig.from_pretrained(NAME))
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

IN, WEIGHTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BS = int(sys.argv[4]) if len(sys.argv) > 4 else 256
dev = torch.device('mps')
t0 = time.time()
tab = pq.read_table(IN, columns=['rid', 's1', 'q_text', 's_text'])
tok = tokenizer()
ids = encode(tok, tab.column('q_text').to_pylist(), tab.column('s_text').to_pylist(), 128)
order = np.argsort([len(x) for x in ids], kind='stable')
model = OfflineCrossEncoder()
model.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
model.to(dev).eval()
out = np.zeros(len(ids), dtype=np.float32)
print(f'tokenized {len(ids):,} ({time.time()-t0:.0f}s)', flush=True)
with torch.no_grad():
    for i in range(0, len(ids), BS):
        idx = order[i:i + BS]
        x, a = collate([ids[j] for j in idx], tok.pad_token_id, dev)
        with torch.autocast('mps', dtype=torch.float16):
            out[idx] = model(x, a).float().cpu().numpy()
        if (i // BS) % 400 == 0:
            done = i + len(idx)
            print(f'{done:,}/{len(ids):,} {done/(time.time()-t0):.0f} pairs/s', flush=True)
pq.write_table(pa.table({'rid': tab.column('rid'), 's1': tab.column('s1'), 'ce': out}), OUT)
print(f'done {time.time()-t0:.0f}s', flush=True)
