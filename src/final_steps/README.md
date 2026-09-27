# Final steps: from stage-2 scores to the submitted file

The main pipeline in `src/` (normalize, blocking, features, stage 1, odd features, stage 2 v10 model, France guard v6 model) produces stage-2 scores for all 19.2M test candidate pairs. The scripts in this folder are the steps that ran after it to produce the final `matching_results.tsv`, together with the validation and gate scripts that justified each step.

## Environment

Every script reads its locations from environment variables instead of machine paths:

| Variable | Meaning | Default |
|---|---|---|
| `REPO_DIR` | repository root (holds `src/`, `data/`, `models/`, `output/`, `student_resource/`) | `.` |
| `WORK_DIR` | experiment workspace holding the intermediate parquet files and checkpoints | `work` |
| `PYTHON` | interpreter used by the shell scripts | `python` |
| `FR_RERUN_DIR` | France re-run workspace (only `france_rerun/`) | `work/fr_rerun` |

Relative values are resolved against the directory you launch from, so run everything from the repository root, for example:

```bash
export REPO_DIR=$PWD WORK_DIR=$PWD/work PYTHON=$PWD/.venv/bin/python
```

`WORK_DIR` uses this layout (the scripts read and write these subfolders):

```
work/
  ce/            cross-encoder bands, checkpoints (ce1.pt, ce2.pt), CE scores, stacker outputs
  v11/           test prediction parquets (pred_v22.parquet, pred_v22h.parquet)
  wf/hunt-odd-one-out-model/   held-out split tables (truth.parquet, valid_s1.parquet, ...)
  rescue/        rescue candidates, scores and test_rescue_adds.parquet
  france_audit/judge/          France audit sets (judge_keep_removals, judge_keep_adds, judge_n2_adds)
  fr_rerun/      France re-run data, work and out/france_assign_partial.parquet
  push/frce/, push/frrescue/, push/gate/       France push sets
  top50/errors/, top50/reverse/, top50/gate/   name-key and reverse rescue sets
  fill/frfill/, fill/gate/                     France empty-S1 fill sets
```

Python modules that the scripts import from each other are found through `sys.path` entries that point into this folder (`lib/`, `france_rerun/src/`, `france_rerun/norm/`, `france_push/frce/`, `rescue_extra/errors/`) or into `src/`.

## Order and commands

### 1. Main pipeline

Run the `src/` pipeline as described in the top-level README. It writes the stage-2 predictions under `data/features/` and the word lists `data/translit_map.json` (`src/learn_translit.py`), `data/decoy_words.json` (`src/learn_decoy_words.py`) and `data/replacement_words.json` (`src/decoy_families.py`).

### 2. Cross-encoders and stacking (`stack/`, `kaggle/`)

```bash
$PYTHON src/final_steps/stack/build_bands.py          # uncertain band, valid + test text pairs -> work/ce/
$PYTHON src/final_steps/stack/build_train.py          # training pairs for the cross-encoders
$PYTHON src/final_steps/stack/ce_train.py ...          # ce1, ce2 (multilingual e5-small, local)
$PYTHON src/final_steps/stack/ce_infer.py <pairs.parquet> <ckpt.pt> <out.parquet> 256
```

`kbase` (e5-base), `klarge` (e5-large with frozen embeddings, 900k pairs) and `kbge` (bge-reranker-v2-m3) were trained on Kaggle GPUs with `kaggle/gpu_ce.py`, launched by `kaggle/run.py` (set `--model`, `--pairs`, `--tag` in `run.py`). Their score files are copied to `work/ce/` as `valid_<tag>.parquet` and `test_<tag>.parquet`.

`stack/stack_eval.py` and `stack/stack_eval2.py` are the held-out validation of the stacker. `stack/stack_test2.py` fits the stacker on the validation band and scores the test band; `stack/mkpred2.py` writes the full prediction parquet with the France least rule. Both are called by `assemble/build_v22.sh`.

### 3. High-confidence recheck (`recheck/`)

`recheck/high_eval.py` is the validation; `recheck/high_apply.py <pred_in> <pred_out>` applies it (called by `build_v22.sh`).

### 4. Decision stage and France assembly (`assemble/`, `france_rerun/`)

Inputs that must exist before the chain runs:

- France audit sets: `assemble/france_audit/j01..j32` (shared helpers `q.py`, `jcls.py`, `jrun.py`; `j32_final.py` writes the kept sets).
- Rescue additions: `assemble/rescue/make_rescue.py train|valid|test`, score with `stack/after_rescue.sh`, validate with `rescue_eval*.py`, then `rescue_apply.py` writes `work/rescue/test_rescue_adds.parquet`.
- France re-run with the fixed normalization: `france_rerun/s1_norm.py`, `s2_block.py`, `s3_score.py`, `s4_stack.py`, `s5_decide.py` in that order (`s6_compare.py` reports the change). Writes `work/fr_rerun/out/france_assign_partial.parquet`. `france_rerun/src/` holds the five `src/` modules that differ for this run and `france_rerun/norm/norm_v2.py` the fixed address normalization. `france_rerun/ofeats.py` is the odd-one-out feature code the v10 stage 2 model was trained with (used by `s3_score.py`; it reads `data/decoy_words.json` and `data/v10/l2_words.json`).

`assemble/build_from_pred.sh <pred.parquet> <tag>` runs `predict_submission.write_outputs(0.85, pred, ("sibling", "additions", "expected"))`, applies the audit sets and rescue additions with `apply_changes.py`, and swaps in the France re-run rows with `fr_swap_sets.py`.

### 5. France push (`france_push/`)

- `frce/`: `p1_prep.py`, `a1_words.py` to `a5_adds.py`, `b1_build.py` (synthetic French pairs), `ce_train_fr.py` (fine-tune from ce1), `run_gpu.sh` / `run_gpu2.sh` (train and score), `e_lab.py`, `e_dis.py`, `e_final.py` (evaluation), `f_write.py` (removal set).
- `frrescue/`: `s1_pool.py`, `s2_filter.py`, `s3_cls.py`, `s4_look.py` (France rescue additions).
- `gate/`: `g1_base.py` to `g5_a.py` checks, `g6_final.py` writes `work/push/gate/removals_trim.parquet` and `additions_trim.parquet`.

### 6. Name-key rescue, hc and reverse additions (`rescue_extra/`)

- `errors/`: `build_db.py`, `feat.py`, `nm1b_lib.py` helpers; `gen_*.py` build candidates per family (nm1, nm1b, nm1c, nm1d, hc, hca, adr); `eval_*.py` validate each family on the held-out split; `combo*.py` compare combinations; `apply_test.py` writes the test additions.
- `reverse/`: `k20a.py`, `k20feat.py`, `k20rf.py`, `k20b.py` train and validate the reverse search; `k20test.py`, `k20testrf.py` score test; `k20apply.py` writes `work/top50/extras_adds.parquet`.
- `gate/`: `build_db.py`, `save15.py`, `rescue15.py`, `combo*.py`, `combined.py`, `k20b.py`, `check_test.py` validate the combined sets and write `work/top50/gate/nm1d_adds.parquet`.

### 7. France empty-S1 fill (`france_fill/`)

`frfill/b1.py` to `b6.py` build the pool; `gate/g1.py` to `g9.py` apply the strict filters and write `work/fill/gate/additions_strict.parquet` (24 pairs).

### 8. Final chain

```bash
bash src/final_steps/assemble/build_v22.sh
```

It stacks the cross-encoder scores, writes predictions, applies the recheck, runs `build_from_pred.sh`, applies the France push, nm1d, reverse extra and France fill sets, and validates the result with `student_resource/utils/validate_submission.py`. Output: `output/v22/matching_results_v22.tsv` and `output/v22/candidate_pairs_v22.tsv`. Removing `valid_kbge.parquet` and `test_kbge.parquet` from `V` and `T` in the script rebuilds v21 (four cross-encoders) exactly.

### Packaging

```bash
cd src && $PYTHON package_submission.py <team> <matching.tsv> <candidates.tsv>
```
