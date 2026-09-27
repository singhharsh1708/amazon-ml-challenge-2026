"""Step 5 (France push). Experiment/validation script cpu_score.py for this step.

Step context: Step 5 (France push). A French-adapted cross-encoder trained on synthetic French name/address perturbations, used to propose France removals.

Command-line arguments used: argv[1], argv[2], argv[3].
"""
import sys
sys.dont_write_bytecode = True
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from ce_model import CrossEncoder, tokenizer, encode, collate
IN, WEIGHTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(3)
tab = pq.read_table(IN, columns=['rid', 's1', 'q_text', 's_text'])
tok = tokenizer()
ids = encode(tok, tab.column('q_text').to_pylist(), tab.column('s_text').to_pylist(), 128)
model = CrossEncoder(); model.load_state_dict(torch.load(WEIGHTS, map_location='cpu')); model.eval()
out = np.zeros(len(ids), np.float32)
order = np.argsort([len(x) for x in ids], kind='stable')
with torch.no_grad():
    for i in range(0, len(ids), 64):
        idx = order[i:i + 64]
        x, a = collate([ids[j] for j in idx], tok.pad_token_id, 'cpu')
        out[idx] = model(x, a).numpy()
pq.write_table(pa.table({'rid': tab.column('rid'), 's1': tab.column('s1'), 'ce': out}), OUT)
print('done', len(ids))
