# Business Entity Resolution: reproduction

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data. It uses only the provided training and test files: no external lookups, APIs, geocoding or internet data. The only outside artefacts are the pretrained weights of `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base` and `intfloat/multilingual-e5-large` (MIT licence) and, only if v22 is the final file, `BAAI/bge-reranker-v2-m3` (Apache 2.0). The cross-encoder steps download them from the public model hub and fine-tune them on the training pairs. The methodology is in `Documentation_template.md`.

The final file is v21 (`matching_results_v21.tsv`, `candidate_pairs_v21.tsv`): steps 1 to 12 below produce the scores, rule sets and French rows it is built from, and step 13 (`src/final_steps/`) builds it.

## Environment

- Python 3.12, 16 GB of RAM and at least 25 GB of free disk. Every DuckDB step caps its own memory (1.2 to 4 GB) and spill and deletes its temp files when it finishes or fails.
- An Apple Silicon GPU (MPS) or a CUDA GPU for the e5-small cross-encoders (the scripts fall back to CPU, which is much slower).
- **A CUDA GPU for the e5-base and e5-large cross-encoders (`kbase`, `klarge`).** We used free Kaggle notebooks with a Tesla T4 (14.56 GiB usable), which fits e5-base at batch 128 in fp16 and e5-large at batch 32 with frozen word embeddings. The optional fifth cross-encoder (`kbge`, v22 only) was trained on a free Kaggle P100.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a GPU notebook that already has PyTorch, `pip install transformers pyarrow` is enough for the cross-encoder steps.

## Data

Point the code at the challenge `student_resource` folder, the one containing `dataset/train`, `dataset/test` and `utils/validate_submission.py`:

```bash
export AMAZON_ML_DATA_ROOT=/path/to/student_resource
```

All intermediate files go to `data/`, models to `models/`, outputs to `output/`. The validation split is `hash(source 1 id) % 20 = 0` with DuckDB's hash, so keep DuckDB at the pinned 1.5.5.

## Run order

Run from this folder, in this order. Each step reads the files the previous steps wrote.

### 1. Normalization and blocking

```bash
python src/learn_translit.py
python src/build_normalized.py
python src/block_candidates.py train
python src/block_candidates.py test
python src/evaluate_blocking.py
```

`learn_translit.py` learns the transliterated-word map (`data/translit_map.json`) from the training pairs. `build_normalized.py` writes `data/norm/`. It now includes the French address fixes (step 12); v15 was built from `data/norm/` written before them, so running it replaces the French rows that v15 used. US and Indian rows are unchanged (identical for all 10,007,688 US and Indian test records), and the training files have no French records. Blocking writes `data/candidates/{train,test}.parquet` (up to 10 candidates per Source 2/3 record); `evaluate_blocking.py` prints recall against the training ground truth and is optional.

### 2. Pair features and stage 1

```bash
python src/build_features.py train
python src/build_features.py test
python src/train_matcher.py
python src/predict_submission.py stage1
```

`train_matcher.py` trains the stage 1 LightGBM (`models/matcher.txt`) and writes validation predictions. `predict_submission.py stage1` only writes the stage 1 test predictions (`data/features/test_predictions.parquet`) that the odd-one-out features need; it writes no output files.

### 3. Learned word lists and the France guard

```bash
python src/learn_decoy_words.py
python src/decoy_families.py
python src/train_stage2.py
mkdir -p models/v6
cp models/stage2.txt models/stage2.json models/v6/
```

The two word-list scripts write `data/decoy_words.json` and `data/replacement_words.json`. Running `train_stage2.py` now, before the odd-one-out features exist, gives the 27-feature stage 2 model that serves as the France guard (`models/v6/stage2.txt`). Our guard was trained with the duplicate-copy augmentation switched on in `train_stage2.py` (`COPY_TENTHS` above 0; see Section 6 of the methodology). The shipped default is 0, so a rebuilt guard can move French results slightly.

### 4. Odd-one-out features and stage 2

```bash
python src/build_odd_features.py train
python src/build_odd_features.py test
python src/train_stage2.py
```

Writes `data/features/{valid,test}_odd.parquet`, the final 55-feature stage 2 model (`models/stage2.txt`) and its out-of-fold validation scores (`data/features/valid_stage2_oof.parquet`).

### 5. Cross-encoder pair tables

```bash
python src/build_ce_pairs.py train valid test
```

Writes `data/cross_encoder/train_pairs.parquet` (1,392,272 labelled training pairs), `valid_band.parquet` (82,133 pairs) and `test_band.parquet` (902,470 pairs). The `test` part also scores stage 2 and the France guard on test (`data/features/test_predictions_stage2.parquet`).

### 6. Cross-encoder training

```bash
python src/train_cross_encoder.py ce1 --rows 250000 --lr 3e-5
python src/train_cross_encoder.py ce2 --offset 250000 --rows 500000 --lr 2e-5 --init ce1
```

ce1 and ce2 are `intfloat/multilingual-e5-small` with frozen word embeddings, batch 64, max length 128, one pass over their rows. On an Apple M2 they take about 67 and 122 minutes.

**kbase needs a CUDA GPU:**

```bash
python src/train_cross_encoder.py kbase --model intfloat/multilingual-e5-base --rows 1392272 \
    --batch 128 --lr 3e-5 --no-freeze-embeddings --shuffle --device cuda
```

This is `intfloat/multilingual-e5-base`, all weights trainable, fp16, one epoch over all 1,392,272 pairs (10,878 steps, about 245 pairs/s on a T4). To run it on a hosted notebook, copy `src/`, `data/cross_encoder/train_pairs.parquet` and the two band files there. For the submitted file, kbase was trained on Kaggle with a standalone copy of this training loop (same model, data, pair order and hyperparameters). That run did not fix the PyTorch seed and GPU training is not bit-for-bit deterministic, so rebuilt scores differ slightly.

Weights go to `models/cross_encoder/<tag>.pt` with the settings in `<tag>.json`.

### 7. Cross-encoder scoring

```bash
python src/score_cross_encoder.py ce1
python src/score_cross_encoder.py ce2
python src/score_cross_encoder.py kbase --device cuda
```

Each writes `data/cross_encoder/valid_<tag>.parquet` and `test_<tag>.parquet`. The base model is read from `<tag>.json`. On an M2 an e5-small model scores the test band in about 21 minutes; e5-base on a T4 takes about 8 minutes.

**klarge (and kbge for v22)** were trained and scored on Kaggle with the standalone script `gpu_ce.py` (`src/final_steps/kaggle/gpu_ce.py`, with its Kaggle runner `src/final_steps/kaggle/run.py`). It reads `data/cross_encoder/train_pairs.parquet` and the two band files and writes validation and test band scores in the same `rid`, `s1`, `ce` format. klarge is `intfloat/multilingual-e5-large` with frozen word embeddings, batch 32 and 900,000 training pairs; kbge is `BAAI/bge-reranker-v2-m3`. Their score files go next to the others as `valid_klarge.parquet`, `test_klarge.parquet` (and `valid_kbge.parquet`, `test_kbge.parquet`).

### 8. Stacker

```bash
python src/train_stacker.py ce1 ce2 kbase
```

Trains the LightGBM stacker on the validation band (prints per-model and stacked out-of-fold AUC), saves `models/stacker.txt` and writes `data/cross_encoder/test_stacked.parquet`. `predict_submission.py` picks the stacked scores up automatically. This step reproduces the v15 three-model stack; the final file uses the four-model stack with klarge, built in step 13.

### 9. First assignment (without rescue)

```bash
python src/predict_submission.py 0.85 sibling,additions,expected,france
```

This writes an assignment without the rescue pass. The rescue step uses it to find the test records that are still unassigned.

### 10. Rescue candidates, scores and model

```bash
python src/rescue_candidates.py train
python src/rescue_candidates.py test output/matching_results.tsv
python src/score_rescue.py ce1 train test
python src/score_rescue.py ce2 train test
python src/train_rescue.py ce1,ce2
```

`rescue_candidates.py` searches six extra key families for records blocking missed and writes `data/rescue/{valid,test}_rescue.parquet`. The `train` part needs two validation files in `data/rescue/`: `valid_scores.parquet` (`rid`, `s1`, `p2`: the validation-slice scores after stacking) and `valid_assigned.parquet` (`rid`, `s1`: the validation assignment after the veto, expected-F0.5 selection and sibling rule), which define the records that are still unassigned on validation. No script in `src/` writes these two files yet; for the submitted file they were exported from our validation evaluation of the stacked system. `score_rescue.py` scores the rescue pairs with the two e5-small cross-encoders. `train_rescue.py` trains the acceptance model (`models/rescue.txt`, threshold 0.7, US and India only) and prints the validation gain for each threshold.

### 11. Final files

```bash
python src/predict_submission.py 0.85
```

Threshold 0.85 with the default rules (`sibling,additions,expected,france,rescue`): best candidate per record, decoy-word veto, expected-F0.5 selection per Source 1 entity for US and India, the 0.85 threshold and stage 2 guard for France, the house-number sibling rule, French exact-address additions, the France audit rules and the rescue additions. It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs `utils/validate_submission.py` on them (exit code 0 means PASS).

For reference, v15 has 1,732,544 Source 1 rows, 100,309 of them empty, 5,817,948 matched records and 19,246,911 candidate pairs. v17 (step 12) has 1,732,544 rows, 100,036 of them empty, 5,829,018 matched records and 19,250,779 candidate pairs, and passes the validator.

### Differences from v15

Given the exact v15 scores (stacked scores and guard scores) and the saved rescue candidates and cross-encoder scores, this step writes 5,817,267 matches instead of 5,817,948 and passes the validator. 4,393 pairs in 4,332 records differ:

- 4,049 French records: the v15 file applied France rule lists computed on the v11 inputs, while `france_rules.py` recomputes them from the current scores (16,711 removals and 23,399 additions on the v15 scores).
- 281 US and Indian records: the v15 rescue additions came from an earlier fit of the acceptance model (on the v11 validation assignment) than the saved model used in this comparison.
- 2 Indian records whose two best candidates have exactly the same score: the code now picks the smaller Source 1 id, and expected-F0.5 selection then drops both records.

In a full rebuild the rescue candidates are generated against the step 9 file; for v15 they were generated against the v11 matching file.

The stage 2 model used for v15 (55 features, same features and LightGBM parameters as step 4) was trained by an earlier version of `train_stage2.py`; steps 3 and 4 have not been re-run end to end with the final code.

v17 has the same US and Indian rows as v15. Its French rows come from step 12, not from this step. The final file v21 is built from these pieces by step 13.

### 12. French address fix and the French rows (v17)

v17 is v15 with only the French Source 1 rows replaced. They come from a France-only re-run of the pipeline on the fixed address normalization, with the v15 models, followed by a filter against v15:

1. `normalize.py` applies four fixes to French addresses (`normalize_address_france`): it removes region and department names and the `N°` sign, splits number suffixes (`19BIS` to `19 bis`, `5B` to `5 bis`), and maps French street-type abbreviations to one form.
2. Re-run scripts normalize the French records into their own folder, block them, score them with stage 1, stage 2 and the France guard, re-stack the band with the v15 stacker, and apply the France decision and rules. They import the pipeline modules (`block_candidates`, `build_features`, `train_stage2`, `address_rules`, `france_rules`, `decoy_veto`) from a copy of `src/` taken before the changes described in this step, whose `config.py` points at the re-run folder. The re-run scripts themselves are working scripts kept outside `src/` and are not in this folder.
3. `france_rules.py partial` drops the new French pairs whose house number or street differs from the Source 1 address and puts back v15 pairs that the re-run lost through its score, guard or blocking when name and address agree.
4. A merge script replaces the French rows of the v15 matching file with the filtered pairs.

With `F` the re-run folder (the scripts take their own paths from `frc.py` and the `src/config.py` copy in that folder):

```bash
cd $F
python s1_norm.py
python s2_block.py new
python s3_score.py new $F/work/new/cand.parquet feat,s1,s2
python s4_stack.py new
python s5_decide.py new $F/work/new/cand.parquet
python s6_compare.py new
cd /path/to/this/folder
python src/france_rules.py partial --new $F/out/france_assign.parquet \
    --previous output/v15/matching_results_v15.tsv \
    --scores $F/work/new/scores.parquet --base $F/work/new/base_assign.parquet \
    --norm-dir $F/data/norm --out $F/out/france_assign_partial.parquet
cd $F/chk && python c8.py
```

On the saved re-run files the `partial` step prints:

```
partial: new 856,026, previous 845,271, new_only 15,812, previous_only 5,057, dropped 1,112, restore_candidates 1,449, restored 1,427, final 856,341, duplicate_rids 0
partial: previous-only pairs by reason: france_changes 1,645, guard 973, not_candidate 313, other 68, score 2,058
```

Its 856,341 pairs are exactly the French pairs of v17 (0 differences either way). `c8.py` writes the merged matching file, which is byte-identical to `output/v17/matching_results_v17.tsv`. The v17 candidate file is the v15 candidate file with the 3,868 v17 French matches it did not contain added to their Source 1 rows; the commands above do not write it.

`--norm-dir` must hold test files normalized with the French fix: the re-run's `$F/data/norm`, or `data/norm/` after `python src/build_normalized.py test` with the current code (both give the same 856,341 pairs). The default `data/norm/` as v15 left it still has the old French addresses; with it the step drops 1,069 new pairs instead of 1,112 and writes 856,384 pairs.

Three limits of this step:

- **Not bit-for-bit repeatable.** `s2_block.py` carries its own copy of the blocking query with the old floating-point score sum (see Determinism), so a fresh run will not reproduce the saved French candidates exactly. The saved intermediate files are the reference for v17.
- **Not all from raw data.** `s4_stack.py` reads the v15 cross-encoder band scores and the v15 stacker, which this sequence does not rebuild. 147,634 of the 205,609 French band pairs have v15 cross-encoder scores and are re-stacked; the others keep their stage 2 score.
- **`build_normalized.py` overwrites `data/norm/`.** `python src/build_normalized.py test` now gives the same French rows as `s1_norm.py` (0 differences over 1,694,445 French records), but it replaces the files v15 was built from. The re-run writes its normalized files to its own folder instead.

### 13. Final steps: the v21 file (`src/final_steps/`)

The final file is built by one chain in `src/final_steps/`. The exact commands, inputs and outputs are in `src/final_steps/README.md`; this is the order and what each step does:

1. **Four-model stack.** A LightGBM stacker over logit(stage 2), the ce1, ce2, kbase and klarge logits, logit(stage 1), address_missing, source and rank, trained on the validation band and applied to the 902,470 test band pairs (+0.00013 over the three-model stack on validation). For v22, kbge is a fifth input.
2. **Prediction table with the France rule.** Stacked scores replace stage 2 scores in the band; for French pairs the score is the lower of stage 2 and the stacked score, and the France guard is kept.
3. **High-confidence recheck.** The 429,125 US and Indian top-1 pairs with a stage 2 score in (0.99, 0.999] are scored by ce1 and ce2 and a small stacker; each keeps the lower of its old and new score (+0.0001 on validation).
4. **Decision.** `predict_submission.write_outputs(0.85, pred, ("sibling", "additions", "expected"))`: best candidate per record, decoy-word veto, expected-F0.5 selection for US and India, 0.85 for France, sibling rule, French exact-address additions.
5. **France audit sets and rescue.** The France removals and additions and the US/India rescue additions are applied, and then the French rows are replaced by the v17 French pairs (step 12).
6. **France push.** 922 French type-word-swap decoys removed (French-adapted cross-encoder, trimmed by hand-read and fingerprint gates) and 2,248 French rescue additions.
7. **Name-key, hc and reverse rescue.** 3,169 nm1d additions (+0.00017 held-out) and 2,491 hc and reverse additions (+0.000075 held-out), US and India only, each to a record that is still unassigned.
8. **France empty-entity fill.** 24 strict French pairs for Source 1 entities that would otherwise stay empty.
9. **Validator.** `utils/validate_submission.py` on the two files.

The scripts take their locations from two environment variables: `REPO_DIR` (this folder, default `.`) and `WORK_DIR` (intermediate files, default `work`). The chain rebuilds v21 exactly from the saved scores and change sets. `src/final_steps/README.md` also covers the scripts that produced the change sets (the French-adapted cross-encoder, the French rescue pool, the nm1d, hc and reverse searches, the fill pool and their gates); the methodology (Section 4.9) gives the numbers each gate produced.

Reference counts for v21: 1,732,544 Source 1 rows, 99,991 of them empty, 5,833,349 matched records and 19,258,685 candidate pairs (11.12 per Source 1 on average, median 9, 45 rows with no candidate). Every match is in the candidate file, and the validator passes.

### 14. Package

```bash
python src/package_submission.py Inno8 output/matching_results.tsv output/candidate_pairs.tsv
```

Writes `output/Inno8_submission.zip` with both output files, `src/`, this README, `requirements.txt` and the methodology document. For the final file:

```bash
python src/package_submission.py Inno8 output/v21/matching_results_v21.tsv output/v21/candidate_pairs_v21.tsv
```

(or the `v22` files if v22 is the last upload).

## What each script does

| Script | Purpose |
| --- | --- |
| `config.py` | Paths; `AMAZON_ML_DATA_ROOT` points at the challenge data |
| `normalize.py` | Name and address normalization shared by every step, with four extra fixes for French addresses |
| `learn_translit.py` | Transliterated word to English word map from matched training pairs |
| `build_normalized.py` | Normalizes all six source files to parquet |
| `block_candidates.py` | Up to 10 Source 1 candidates per Source 2/3 record from IDF-weighted rare keys |
| `evaluate_blocking.py` | Blocking recall against the training ground truth |
| `build_features.py` | Pair features for train, validation and test |
| `train_matcher.py` | Stage 1 LightGBM and its validation predictions |
| `learn_decoy_words.py` | Words that decoys add to real business names |
| `decoy_families.py` | Replacement words and decoy-family flags |
| `decoy_veto.py` | Decoy-word veto used at assignment time |
| `odd_features.py`, `build_odd_features.py` | Odd-one-out features against the other records of the same Source 1 entity |
| `train_stage2.py` | Stage 2 LightGBM with record, entity and odd-one-out context, out-of-fold on the validation slice |
| `cross_encoder.py` | Cross-encoder model, tokenization and batched scoring |
| `build_ce_pairs.py` | Cross-encoder training pairs and the validation and test bands |
| `train_cross_encoder.py` | Fine-tunes a cross-encoder (MPS, CUDA or CPU) |
| `score_cross_encoder.py` | Scores the bands with a trained cross-encoder |
| `train_stacker.py` | LightGBM stacker over stage 1, stage 2 and cross-encoder scores |
| `expected_f.py` | Expected-F0.5 selection per Source 1 entity |
| `address_rules.py` | French exact-address additions |
| `france_rules.py` | France audit rules (removals and additions); also runs on its own and writes `data/france_changes/`; `france_rules.py partial` is the v17 filter of a French re-run against an earlier submission |
| `rescue_candidates.py`, `score_rescue.py`, `train_rescue.py` | Rescue pass for records blocking missed |
| `predict_submission.py` | Test scoring, stacking, assignment rules, both output files and the official validator |
| `final_steps/` | The chain that builds the final file: four-model stack, high-confidence recheck, decision, France audit sets and rescue, v17 French rows, France push, name-key/hc/reverse rescue, France fill (commands in `final_steps/README.md`) |
| `package_submission.py` | Builds the submission zip |

## Cached predictions

`predict_submission.py` and `build_ce_pairs.py` reuse `data/features/test_predictions.parquet` and `data/features/test_predictions_stage2.parquet` when they exist. After retraining stage 1, stage 2 or the guard, delete both files so they are recomputed.

## Determinism

The validation split (DuckDB `hash`), features and all LightGBM models (fixed seeds) are deterministic for the pinned DuckDB and LightGBM versions. Blocking now is too. It used to sum the key IDF weights as doubles, and DuckDB adds them in a different order on each parallel run, so candidates that tie at a record's cut changed from run to run (38,703 French records got a different candidate set when the French blocking was re-run on unchanged data). `block_candidates.py` now rounds each IDF times 10^9 to an integer and sums integers. On a sample of 71,532 French records blocked three times, the new query gave identical results, where the old one differed in 3,140 and 1,983 pairs between runs. The v15 candidate files were made with the old query, so a rebuild can differ from them in tie pairs. Cross-encoder training is seeded, but GPU kernels are not bit-for-bit deterministic, so cross-encoder scores, and therefore a small number of borderline assignments, can differ between runs and machines.
