"""Trains a cross-encoder (ce1/ce2, e5-small) on the training pairs on the local machine.

Step context: Step 2 (cross-encoder stacking). Cross-encoders re-read the uncertain band of stage-2 pairs as text; a LightGBM stacker combines logit(p2), the cross-encoder logits, logit(p1), address_missing, source and rank.

Command-line arguments used: argv[1], argv[2], argv[3], argv[4], argv[5], argv[6].

Inputs (paths relative to the WORK_DIR/REPO_DIR layout):
    $WORK_DIR/ce
    {CE}/train_pairs.parquet

Outputs:
    {CE}/{TAG}.pt
"""
import os
REPO_DIR = os.path.abspath(os.environ.get("REPO_DIR", "."))
WORK_DIR = os.path.abspath(os.environ.get("WORK_DIR", "work"))
import sys, time, math
import numpy as np, pyarrow.parquet as pq, torch
from ce_model import CrossEncoder, tokenizer, encode, collate
CE = f'{WORK_DIR}/ce'
N, BS, LR, MAXLEN, TAG = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
BENCH = len(sys.argv) > 6 and sys.argv[6] == 'bench'
torch.manual_seed(0)
dev = torch.device('mps')
t0 = time.time()
tab = pq.read_table(f'{CE}/train_pairs.parquet', columns=['q_text', 's_text', 'label']).slice(0, N)
tok = tokenizer()
ids = encode(tok, tab.column('q_text').to_pylist(), tab.column('s_text').to_pylist(), MAXLEN)
y = torch.tensor(tab.column('label').to_numpy(), dtype=torch.float32)
print(f'tokenized {len(ids):,} pairs, mean len {np.mean([len(x) for x in ids]):.1f} ({time.time()-t0:.0f}s)', flush=True)
model = CrossEncoder().to(dev)
opt = torch.optim.AdamW([{'params': model.enc.parameters(), 'lr': LR}, {'params': model.head.parameters(), 'lr': LR * 20}], weight_decay=0.01)
steps = math.ceil(len(ids) / BS); warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
lossf = torch.nn.BCEWithLogitsLoss()
model.train(); t1 = time.time(); run = 0.0
for step in range(steps):
    sl = slice(step * BS, (step + 1) * BS)
    x, a = collate(ids[sl], tok.pad_token_id, dev)
    with torch.autocast('mps', dtype=torch.bfloat16):
        logit = model(x, a)
    loss = lossf(logit.float(), y[sl].to(dev))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
    if step % (20 if BENCH else 200) == 0 or step == steps - 1:
        rate = (step + 1) * BS / (time.time() - t1)
        print(f'step {step}/{steps} loss {run:.4f} {rate:.0f} pairs/s eta {(steps-step-1)*BS/rate/60:.1f} min', flush=True)
    if BENCH and step == 60:
        break
    if not BENCH and step and step % 3000 == 0:
        torch.save(model.state_dict(), f'{CE}/{TAG}.pt')
if not BENCH:
    torch.save(model.state_dict(), f'{CE}/{TAG}.pt')
print(f'done {time.time()-t0:.0f}s', flush=True)
