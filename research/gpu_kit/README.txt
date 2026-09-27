Cross-encoder training package (Inno8, private: do not share the data files)

Files
  gpu_ce.py            training + scoring script
  train_pairs.parquet  1,392,272 labelled record pairs from the train split
  valid_band.parquet   82,133 validation pairs to score
  test_band.parquet    902,470 test pairs to score

Run (Kaggle / Colab / any CUDA machine, GPU with 16 GB or more)
  pip install -q transformers pyarrow
  python gpu_ce.py --data . --out out --model intfloat/multilingual-e5-base --epochs 1 --batch 128

Faster fallback if the GPU is small or time is short
  python gpu_ce.py --data . --out out --model intfloat/multilingual-e5-small --epochs 2 --batch 256 --tag small2

Send back the whole out/ folder (the two *_band_*.parquet score files are what matter, about 15 MB).
Rough time: H100 about 15 min, A100 about 25 min, T4 or P100 about 70 min for e5-base.
