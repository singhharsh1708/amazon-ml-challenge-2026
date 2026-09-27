import argparse
import glob
import math
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)


def collate(ids_list, pad_id, device):
    n = max(len(x) for x in ids_list)
    ids = torch.full((len(ids_list), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), n), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    return ids.to(device, non_blocking=True), att.to(device, non_blocking=True)


def encode(tok, table, max_len):
    return tok(table.column("q_text").to_pylist(), table.column("s_text").to_pylist(),
               truncation=True, max_length=max_len, padding=False)["input_ids"]


def train(args, tok, dev):
    table = pq.read_table(args.train, columns=["q_text", "s_text", "label"]).slice(args.offset, args.pairs)
    ids = encode(tok, table, args.max_len)
    y = torch.tensor(table.column("label").to_numpy(), dtype=torch.float32)
    print(f"training pairs {len(ids):,}", flush=True)
    model = CrossEncoder(args.model)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model.to(dev)
    if args.freeze_embeddings:
        model.enc.embeddings.word_embeddings.weight.requires_grad_(False)
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": model.head.parameters(), "lr": args.lr * 20}],
                            weight_decay=0.01)
    steps_per_epoch = math.ceil(len(ids) / args.batch)
    steps = steps_per_epoch * args.epochs
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    model.train()
    t0, step, run = time.time(), 0, 0.0
    for epoch in range(args.epochs):
        order = rng.permutation(len(ids))
        for b in range(steps_per_epoch):
            idx = order[b * args.batch:(b + 1) * args.batch]
            x, a = collate([ids[i] for i in idx], tok.pad_token_id, dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(x, a)
            loss = lossf(logit.float(), y[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            step += 1
            if step % 500 == 0 or step == steps:
                rate = step * args.batch / (time.time() - t0)
                print(f"epoch {epoch} step {step}/{steps} loss {run:.4f} {rate:.0f} pairs/s eta {(steps - step) * args.batch / rate / 60:.1f} min", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, f"{args.tag}.pt"))
    return model


@torch.no_grad()
def score(model, tok, dev, path, out_path, max_len, batch):
    table = pq.read_table(path, columns=["rid", "s1", "q_text", "s_text"])
    ids = encode(tok, table, max_len)
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.zeros(len(ids), dtype=np.float32)
    model.eval()
    t0 = time.time()
    for i in range(0, len(ids), batch):
        idx = order[i:i + batch]
        x, a = collate([ids[j] for j in idx], tok.pad_token_id, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out[idx] = model(x, a).float().cpu().numpy()
    pq.write_table(pa.table({"rid": table.column("rid"), "s1": table.column("s1"), "ce": out}), out_path)
    print(f"scored {len(ids):,} pairs from {os.path.basename(path)} in {time.time() - t0:.0f}s -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--tag", default="gpu_ce")
    ap.add_argument("--pairs", type=int, default=1_392_272)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--freeze-embeddings", action="store_true")
    ap.add_argument("--init", default="")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    args.train = os.path.join(args.data, "train_pairs.parquet")
    if args.score_only:
        model = CrossEncoder(args.model)
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        model.to(dev)
    else:
        model = train(args, tok, dev)
    for path in sorted(glob.glob(os.path.join(args.data, "*_band.parquet")) + glob.glob(os.path.join(args.data, "*_rescue.parquet"))):
        name = os.path.basename(path).replace(".parquet", "")
        score(model, tok, dev, path, os.path.join(args.out, f"{name}_{args.tag}.parquet"), args.max_len, 512)


if __name__ == "__main__":
    main()
