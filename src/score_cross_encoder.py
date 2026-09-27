"""Score candidate band files with a trained cross-encoder.

Input: models/cross_encoder/{tag}.pt (and {tag}.json for the base model name and max length),
data/cross_encoder/{band}_band.parquet.
Output: data/cross_encoder/{band}_{tag}.parquet with columns rid, s1, ce (logit).
Run from src/: python score_cross_encoder.py [tag] [valid] [test] [--weights PATH] [--device ...]
"""

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR, MODEL_DIR
from cross_encoder import MAX_LEN, MODEL_NAME, SCORE_BATCH, device, encode, load, score, tokenizer

CE_DIR = DATA_DIR / "cross_encoder"
WEIGHTS_DIR = MODEL_DIR / "cross_encoder"
LOG_EVERY = 400


def model_info(tag, weights_dir=WEIGHTS_DIR, model=None, max_len=None):
    """Return (base model name, max length) for a tag from its metadata, with overrides."""
    meta = Path(weights_dir) / f"{tag}.json"
    info = json.loads(meta.read_text()) if meta.exists() else {}
    return model or info.get("model", MODEL_NAME), max_len or info.get("max_len", MAX_LEN)


def score_band(model, tok, dev, band_file, out_file, log_every=LOG_EVERY, max_len=MAX_LEN, batch_size=SCORE_BATCH):
    """Score every pair of a band parquet and write rid, s1 and the logit to parquet."""
    tab = pq.read_table(band_file, columns=["rid", "s1", "q_text", "s_text"])
    ids = encode(tok, tab.column("q_text").to_pylist(), tab.column("s_text").to_pylist(), max_len)
    ce = score(model, tok, ids, dev, batch_size=batch_size, log_every=log_every)
    pq.write_table(pa.table({"rid": tab.column("rid"), "s1": tab.column("s1"), "ce": ce}), out_file)
    return ce


def main():
    """Command line entry: load a cross-encoder and score the requested bands."""
    ap = argparse.ArgumentParser(description="Score candidate bands with a trained cross-encoder")
    ap.add_argument("tag", nargs="?", default="ce1")
    ap.add_argument("bands", nargs="*", default=["valid", "test"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--batch", type=int, default=SCORE_BATCH)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--weights-dir", default=str(WEIGHTS_DIR))
    ap.add_argument("--band-dir", default=str(CE_DIR))
    ap.add_argument("--out-dir", default=str(CE_DIR))
    args = ap.parse_args()
    name, max_len = model_info(args.tag, args.weights_dir, args.model, args.max_len)
    dev = device(args.device)
    tok = tokenizer(name)
    model = load(Path(args.weights) if args.weights else Path(args.weights_dir) / f"{args.tag}.pt", dev, name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for band in args.bands:
        start = time.time()
        out_file = out_dir / f"{band}_{args.tag}.parquet"
        ce = score_band(model, tok, dev, Path(args.band_dir) / f"{band}_band.parquet", out_file,
                        max_len=max_len, batch_size=args.batch)
        print(f"wrote {out_file} ({len(ce):,} pairs, {name}, mean logit {ce.mean():.3f}) in {time.time() - start:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
