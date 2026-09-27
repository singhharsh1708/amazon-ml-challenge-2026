"""Score rescue candidate files with a trained cross-encoder.

Input: models/cross_encoder/{tag}.pt, data/rescue/{valid,test}_rescue.parquet.
Output: data/rescue/{valid,test}_rescue_{tag}.parquet.
Run from src/: python score_rescue.py [tag] [train] [test]
"""

import sys
import time

from cross_encoder import device, load, tokenizer
from rescue_candidates import OUT_FILES
from score_cross_encoder import WEIGHTS_DIR, model_info, score_band
from train_rescue import ce_files


def main():
    """Load the cross-encoder for a tag and score the rescue files of the requested splits."""
    tag = sys.argv[1] if len(sys.argv) > 1 else "ce1"
    splits = sys.argv[2:] or ["train", "test"]
    name, max_len = model_info(tag)
    dev = device()
    tok = tokenizer(name)
    model = load(WEIGHTS_DIR / f"{tag}.pt", dev, name)
    for split in splits:
        start = time.time()
        out_file = ce_files(split, [tag])[0]
        ce = score_band(model, tok, dev, OUT_FILES[split], out_file, max_len=max_len)
        print(f"wrote {out_file} ({len(ce):,} pairs, mean logit {ce.mean():.3f}) in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
