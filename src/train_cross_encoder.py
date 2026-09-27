"""Fine-tune a cross-encoder on labelled training pairs.

Trains CrossEncoder with binary cross-entropy, AdamW (higher learning rate on the head), linear
warm-up and decay, gradient clipping and optional fp16 on CUDA. Word embeddings are frozen by
default. Checkpoints are written during training so a run can be resumed with --init.

Input: data/cross_encoder/train_pairs.parquet (or --pairs).
Output: models/cross_encoder/{tag}.pt and {tag}.json.
Run from src/: python train_cross_encoder.py [tag] [--model NAME] [--rows N] [--offset N]
  [--epochs N] [--batch N] [--lr X] [--init TAG_OR_PATH] [--device auto|cuda|mps|cpu]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from config import DATA_DIR, MODEL_DIR
from cross_encoder import MAX_LEN, MODEL_NAME, CrossEncoder, collate, device, encode, tokenizer

PAIRS_FILE = DATA_DIR / "cross_encoder" / "train_pairs.parquet"
WEIGHTS_DIR = MODEL_DIR / "cross_encoder"

ROWS = 250_000
BATCH = 64
LR = 3e-5
HEAD_LR_MULT = 20
WEIGHT_DECAY = 0.01
WARMUP_DIV = 20
CLIP_NORM = 1.0
SEED = 0
LOG_EVERY = 200
CHECKPOINT_EVERY = 3000


def weights_file(tag, weights_dir=WEIGHTS_DIR):
    """Return the path of the final weights for a tag."""
    return Path(weights_dir) / f"{tag}.pt"


def checkpoint_file(tag, weights_dir=WEIGHTS_DIR):
    """Return the path of the in-progress checkpoint for a tag."""
    return Path(weights_dir) / f"{tag}.ckpt.pt"


def meta_file(tag, weights_dir=WEIGHTS_DIR):
    """Return the path of the JSON metadata for a tag."""
    return Path(weights_dir) / f"{tag}.json"


def init_file(init, weights_dir=WEIGHTS_DIR):
    """Resolve --init to a weights path: an existing file or .pt path, otherwise a tag."""
    path = Path(init)
    return path if path.suffix == ".pt" or path.exists() else weights_file(init, weights_dir)


def batches(n, batch, epochs, shuffle, seed=SEED):
    """Yield (epoch, index batch) pairs, shuffled per epoch when requested."""
    rng = np.random.default_rng(seed)
    steps_per_epoch = math.ceil(n / batch)
    for epoch in range(epochs):
        order = rng.permutation(n) if shuffle else np.arange(n)
        for b in range(steps_per_epoch):
            yield epoch, order[b * batch:(b + 1) * batch]


def train(args):
    """Tokenise the pairs, fine-tune the model and save weights and metadata."""
    start = time.time()
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_file = weights_file(args.tag, weights_dir)
    ckpt_file = checkpoint_file(args.tag, weights_dir)
    torch.manual_seed(SEED)
    dev = device(args.device)
    fp16 = dev.type == "cuda"
    tab = pq.read_table(args.pairs, columns=["q_text", "s_text", "label"]).slice(args.offset, args.rows)
    tok = tokenizer(args.model)
    ids = encode(tok, tab.column("q_text").to_pylist(), tab.column("s_text").to_pylist(), args.max_len)
    y = torch.tensor(tab.column("label").to_numpy(), dtype=torch.float32)
    print(f"tokenized {len(ids):,} pairs from row {args.offset:,}, mean length {np.mean([len(x) for x in ids]):.1f} "
          f"({time.time() - start:.0f}s); model {args.model} on {dev.type}{' fp16' if fp16 else ''}", flush=True)
    model = CrossEncoder(args.model)
    if args.init:
        model.load_state_dict(torch.load(init_file(args.init, weights_dir), map_location="cpu"))
    model = model.to(dev)
    if args.freeze_embeddings:
        model.enc.embeddings.word_embeddings.weight.requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr * HEAD_LR_MULT},
    ], weight_decay=WEIGHT_DECAY)
    steps = math.ceil(len(ids) / args.batch) * args.epochs
    warm = max(1, steps // WARMUP_DIV)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t1 = time.time()
    run = 0.0
    last_epoch = 0
    for step, (epoch, idx) in enumerate(batches(len(ids), args.batch, args.epochs, args.shuffle)):
        if epoch != last_epoch:
            torch.save(model.state_dict(), ckpt_file)
            last_epoch = epoch
        x, a = collate([ids[i] for i in idx], tok.pad_token_id, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            logit = model(x, a)
        loss = lossf(logit.float(), y[idx].to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % LOG_EVERY == 0 or step == steps - 1:
            rate = (step + 1) * args.batch / (time.time() - t1)
            print(f"epoch {epoch} step {step}/{steps} loss {run:.4f} {rate:.0f} pairs/s "
                  f"eta {(steps - step - 1) * args.batch / rate / 60:.1f} min", flush=True)
        if step and step % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(), ckpt_file)
    torch.save(model.state_dict(), out_file)
    ckpt_file.unlink(missing_ok=True)
    meta_file(args.tag, weights_dir).write_text(json.dumps({
        "tag": args.tag, "model": args.model, "max_len": args.max_len, "rows": len(ids), "offset": args.offset,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "init": args.init or None,
        "freeze_embeddings": args.freeze_embeddings, "shuffle": args.shuffle, "device": dev.type, "fp16": fp16,
    }, indent=2))
    print(f"saved {out_file} in {time.time() - start:.0f}s")


def parse_args(argv=None):
    """Parse the training command line."""
    ap = argparse.ArgumentParser(description="Train a cross-encoder on labelled record pairs")
    ap.add_argument("tag", nargs="?", default="ce1")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--init", default="")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--pairs", default=str(PAIRS_FILE))
    ap.add_argument("--weights-dir", default=str(WEIGHTS_DIR))
    return ap.parse_args(argv)


def main():
    """Command line entry point."""
    train(parse_args())


if __name__ == "__main__":
    main()
