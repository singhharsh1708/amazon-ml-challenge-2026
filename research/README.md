# Research tools

Scripts used for the experiments behind the final submission. The production pipeline lives in `src/`; these are the tools for trying new ideas quickly. None of them contain data: the dataset, `data/`, `models/` and `output/` stay out of git.

## gpu_kit: train a cross-encoder on any CUDA GPU

`gpu_ce.py` is standalone (needs only `torch`, `transformers`, `pyarrow`). It trains a multilingual-e5 cross-encoder on labelled record pairs and scores the validation and test bands.

Inputs, built on a machine with the full pipeline (`python src/build_ce_pairs.py train valid test`), copied next to the script:

- `train_pairs.parquet`: `q_text`, `s_text`, `label`
- `valid_band.parquet`, `test_band.parquet`: `rid`, `s1`, `q_text`, `s_text`

Run:

```
python gpu_ce.py --data . --out out --model intfloat/multilingual-e5-base --epochs 1 --batch 128
```

Outputs `out/valid_band_<tag>.parquet` and `out/test_band_<tag>.parquet` (`rid`, `s1`, `ce`). Put them in `data/cross_encoder/` and add the tag to `src/train_stacker.py`. For `intfloat/multilingual-e5-large` on a 16 GB GPU use `--freeze-embeddings --batch 32 --lr 2e-5`, otherwise it runs out of memory.

Results so far (AUC on the uncertain validation band, stage-2 model alone 0.971):

| Model | Trained on | AUC alone | Stacked |
|---|---|---|---|
| e5-small, 250k pairs (Mac) | ce1 | 0.927 | 0.978 |
| e5-small, +500k pairs (Mac) | ce2 | 0.946 | 0.980 (with ce1) |
| e5-base, 1.39M pairs (Kaggle T4) | kbase | 0.959 | 0.983 (with ce1, ce2) |

## kaggle: run the GPU kit on a free Kaggle notebook

1. Upload the kit folder (script plus the three parquet files) as a private Kaggle dataset.
2. `run.py` is the notebook script: it finds `gpu_ce.py` under `/kaggle/input`, trains, scores and deletes the weights so only the score files remain.
3. Push it with the Kaggle CLI (`kaggle kernels push --accelerator NvidiaTeslaT4`) using a `kernel-metadata.json` that lists the dataset, `enable_gpu: true` and `enable_internet: true`. GPU and internet need a phone-verified Kaggle account.

## france_rerun: re-run France with a different normalization

`s1_norm.py` to `s6_compare.py` re-run only the French test records through normalization, blocking, features, both stage-2 models, the saved stacker and the decision rules, then compare the result with a previous submission. Paths come from environment variables read in `frc.py`:

- `REPO_DIR` (default `.`): this repository
- `WORK_DIR` (default `work`): folder with saved predictions and cross-encoder scores
- `FR_RERUN_DIR` (default `work/fr_rerun`): output folder for the re-run

Run with the old normalization first; it must reproduce the submitted French scores exactly before any change is trusted. The filter that turned the re-run into the final French rows is `python src/france_rules.py partial` (see `submission/README.md`).

## tools

- `apply_changes.py`: applies removal and addition sets (parquet `rid`, `s1`) to a submission pair, keeping one S1 per record and adding new pairs to the candidate file.
- `fr_swap_sets.py`: builds the removal and addition sets that replace a submission's French rows with a new French assignment.

## Ideas not finished

- Cross-encoder rescoring of French rescue candidates (rescue is US/India only today).
- A larger cross-encoder (e5-large) in the stack; the first run needed frozen embeddings to fit a T4.
- Address-less records whose name is shared by many companies stay the largest unsolved loss (about 0.005 of local F0.5); no signal separates the branches of a chain.
