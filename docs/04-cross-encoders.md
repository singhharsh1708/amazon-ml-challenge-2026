# 4. Cross-encoders on the uncertain band

Part 4 of the technical deep dive. Previous: [3. The two LightGBM stages](03-lightgbm-stages.md). Next: [5. Decision rules](05-decision-rules.md).

By the final day, LightGBM on hand-built features had stopped improving on one kind of pair: names that differ in ways a similarity score cannot express, such as heavily transliterated Indian names ("al tek teknaljis piraivet limitet" for "Al Tech Technologies") or suffix variants of a real name ("summit foundation cii" next to "summit foundation ii").

A cross-encoder reads both records at once, as text, and decides whether they are the same business. We fine-tuned four multilingual E5 models, ran them only on the 902,470 test pairs where stage 2 was unsure, and combined them with the LightGBM scores in a small stacker. It was the largest late gain. The upload that introduced the first stack (v11, together with the odd-one-out stage 2 and expected-F0.5 selection) took the public leaderboard from 0.974 to 0.982, and every later model added a little more.

This page assumes the two LightGBM stages described in [03-lightgbm-stages.md](03-lightgbm-stages.md). Sections 4.3, 4.4 and 4.9 of [methodology.md](methodology.md) cover the same models in the official write-up.

## At a glance

| Tag | Base model | Trained on | Where | Band AUC alone | Stacked band AUC after adding it | Local F0.5 gain |
| --- | --- | --- | --- | --- | --- | --- |
| (stage 2 only) | LightGBM | | Apple M2 CPU | 0.9714 | 0.9714 | |
| ce1 | multilingual-e5-small | 250,000 pairs | Apple M2 GPU (MPS) | 0.9269 | 0.9777 | +0.00246 (with expected-F0.5 selection) |
| ce2 | ce1, continued | next 500,000 pairs | Apple M2 GPU (MPS) | 0.9457 | 0.9807 | +0.00061 |
| kbase | multilingual-e5-base | all 1,392,272 pairs | Kaggle T4 | 0.9591 | 0.9828 | +0.00054 |
| klarge | multilingual-e5-large | 900,000 pairs | Kaggle T4 | 0.9600 | 0.9834 | +0.00013 |

AUC is measured on the 82,133 validation band pairs (41.3% true). The stacked AUCs are 2-fold out of fold. The F0.5 gains are the test-weighted macro F0.5 on the 110,565 held-out US and Indian entities.

![Band AUC of each model alone and of the growing stack](../assets/cross_encoder_auc.png)

## 1. Architecture

Each side of a pair becomes one string, `name_full | address`, built from the same normalized name and address that the LightGBM features use. The two strings are encoded as a sentence pair and truncated to 128 tokens. The encoder's last hidden layer is mean-pooled over the real tokens, and one linear unit turns the pooled vector into a match logit. Training uses binary cross-entropy on that logit.

```python
h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
m = attention_mask.unsqueeze(-1).to(h.dtype)
pooled = (h * m).sum(1) / m.sum(1)
return self.head(pooled).squeeze(-1)
```

That is the whole model (`src/cross_encoder.py`). A model input looks like this (shortened):

```
record:    toledo womens health | 1788 hamilton st
Source 1:  toledo womens health | 1781 hamilton st
```

A French decoy of the legal-form kind, shown as raw text (the model reads its normalized form):

```
record:    Soutien Amicale SNC, 9 Rue Hans Christian Andersen
Source 1:  Soutien Amicale SAS, 2 Rue Hans Christian Andersen
```

| Base model | Parameters | Licence |
| --- | --- | --- |
| intfloat/multilingual-e5-small | 117,653,760 + 385 (head) | MIT |
| intfloat/multilingual-e5-base | 278,043,648 + 769 (head) | MIT |
| intfloat/multilingual-e5-large | about 560M | MIT |

Why these checkpoints: they are multilingual (28% of Indian business names in Source 2 and 19% in Source 3 are written in Devanagari, Bengali, Tamil, Telugu or Gujarati, and France adds French), MIT licensed, and the small one trains on a 16 GB laptop. The challenge allows MIT or Apache 2.0 models up to 8 billion parameters, so every model here is far inside the limit.

## 2. Training data

The training table is built once by `build_ce_pairs.py train` and shared by every model:

| Step | Rows |
| --- | --- |
| Train-split pairs with a stage 1 score in [0.003, 0.997] | 1,304,531 |
| Plus a 1.5% random sample of all other train-split pairs | 87,741 |
| **Total** | **1,392,272 (45.5% true)** |

- **Never from the validation slice.** The pairs come from the stage 1 train part, whose records never touch a held-out entity, so every validation number for the cross-encoders and the stacker is out of sample.
- **Mostly hard pairs.** Pairs that stage 1 already scores near 0 or 1 teach little. The band keeps the pairs that are actually contested; the 1.5% sample keeps some easy pairs in the mix, since the models also score pairs outside the training band.
- **Ordered once.** The table is sorted by a hash of the pair ids and stored in that order, so every model can take a contiguous slice and two models trained on different slices see disjoint pairs.

| Model | Rows of the table |
| --- | --- |
| ce1 | 0 to 250,000 |
| ce2 | 250,000 to 750,000 (continues from ce1) |
| ce3 (dropped, see section 7) | 750,000 to 1,392,272 (642,272 rows) |
| kbase | all 1,392,272, reshuffled |
| klarge | the first 900,000, reshuffled |

## 3. Where they are applied: the band

Cross-encoders are slow compared with LightGBM, so they only read the pairs where they can change a decision.

| Split | Rule | Pairs |
| --- | --- | --- |
| Test | a record's first- or second-ranked candidate by stage 2, with a stage 2 score in [0.01, 0.99] | 902,470 (France 223,482, India 383,737, US 295,251) |
| Validation | slice pairs with an out-of-fold stage 2 score in [0.002, 0.998] | 82,133 (41.3% true) |

The test band is 4.7% of the 19,225,118 blocking candidates. On the M2 one e5-small model scores it in 1,268 to 1,274 seconds, about 21 minutes; scoring every candidate would cost about 21 times as much, per model, for pairs whose decision is already settled.

## 4. The four models

All four share one training loop (`src/train_cross_encoder.py` on the Mac, the standalone `src/final_steps/kaggle/gpu_ce.py` on Kaggle):

| Shared setting | Value |
| --- | --- |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate of the linear head | 20 times the encoder learning rate |
| Schedule | linear warm-up over the first 5% of steps, then linear decay to 0 |
| Gradient clipping | norm 1.0 |
| Loss | binary cross-entropy with logits |
| Epochs | 1 |
| Max length | 128 tokens, padded per batch |

| Tag | Base | Rows | Word embeddings | Precision | Batch | Learning rate | Hardware | Band AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ce1 | e5-small | 250,000 | frozen | fp32 | 64 | 3e-5 | Apple M2, MPS | 0.9269 |
| ce2 | ce1 | 500,000 | frozen | fp32 | 64 | 2e-5 | Apple M2, MPS | 0.9457 |
| kbase | e5-base | 1,392,272 | trainable | fp16 | 128 | 3e-5 | Kaggle T4 | 0.9591 |
| klarge | e5-large | 900,000 | frozen | fp16 | 32 | 2e-5 | Kaggle T4 | 0.9600 |

For reference, stage 1 scores 0.9415 and stage 2 scores 0.9714 on the same 82,133 pairs.

**ce1.** The first model, trained on the Mac in 4,013 seconds (62 pairs/s). On its own it ranks the band worse than stage 1 (0.9269 against 0.9415), and yet stacked with stage 2 it lifted the band AUC from 0.9714 to 0.9777. That result set the direction for the rest of the day.

**ce2.** Instead of training a second model from scratch, ce2 continues from ce1's weights on the next 500,000 rows with a lower learning rate (7,324 seconds, 69 pairs/s). The rows are new, so the model sees 750,000 distinct pairs in total, and the hour already spent on ce1 is not paid again. Alone it rose from 0.9269 to 0.9457; in the stack, from 0.9777 to 0.9807.

**kbase.** e5-base needs a CUDA GPU in our setup, so kbase ran on a free Kaggle T4 (14.56 GiB usable): all weights trainable, fp16, batch 128, one epoch over all 1,392,272 pairs in 10,878 steps at 245 pairs/s. It scored the test band in 468 seconds and the validation band in 40 seconds. Alone it is the strongest of the first three (0.9591), and it lifted the stack to 0.9828 for the v15 upload.

**klarge.** e5-large ran on the same T4 with frozen word embeddings, batch 32, learning rate 2e-5 and the first 900,000 pairs. Alone it barely beats kbase (0.9600 against 0.9591), but it still adds to the stack: out-of-fold AUC 0.9834, and +0.00013 test-weighted F0.5 over the three-model stack.

No cross-encoder beats stage 2 alone. Stage 2 knows things a pair of strings cannot show: how the record's other candidates scored, and whether other records already claim the same entity. The cross-encoders are wrong on different pairs, and that is exactly what the stacker exploits.

## 5. The stacker

A small LightGBM combines everything known about a band pair:

| Input | Why |
| --- | --- |
| logit of the stage 2 score | the context-aware score |
| one logit per cross-encoder | the text readers |
| logit of the stage 1 score | the pair-only score |
| address missing | when there is no address, the name carries all the evidence |
| source (2 or 3) | the two sources carry different noise (for example 28% against 19% of Indian names in native scripts) |
| blocking rank of the candidate (1 to 10) | how the rare-key search ranked it among the record's candidates |

| Setting | Value |
| --- | --- |
| Leaves | 15 |
| Learning rate | 0.05 |
| Rounds | 300 |
| min_data_in_leaf | 100 |
| Seed | 1 |
| Training data | the 82,133 validation band pairs |
| Evaluation | 2-fold cross-validation split by Source 1 entity |

The folds are split by entity so that the stacker is never trained on one record of an entity and tested on another record of the same entity.

### How the stack grew

| Stack | Out-of-fold band AUC | Local test-weighted F0.5 | Public leaderboard |
| --- | --- | --- | --- |
| stage 2 only (v10) | 0.9714 | 0.98470 | not uploaded |
| + ce1, with expected-F0.5 selection (v11) | 0.9777 | 0.98716 | 0.982 |
| + ce2 (v12) | 0.9807 | 0.98777 | not uploaded |
| + kbase (before rescue) | 0.9828 | 0.98831 | |
| v15 (+ France rules and rescue on top) | 0.9828 | 0.98901 | 0.986784 |
| + klarge (v19 to v21) | 0.9834 | +0.00013 over the three-model stack | v20 0.988549, v21 0.988642 |

The leaderboard column includes every other change made in the same upload (see [methodology.md](methodology.md), Section 6). v11 is also where local and leaderboard gains started to agree: +0.0048 locally and +0.008 on the leaderboard over the 0.974 file. Our reading at the time was that the cross-encoders help more on the decoy-heavy test set than the validation slice can show.

### Applying it to test

A stacker fitted on the whole validation band scores the 902,470 test band pairs, and the stacked probability replaces the stage 2 score for those pairs. On test it moved 44,615 pairs above the 0.85 line and 37,592 below it.

France has no labels, so for French pairs the score is the **lower** of the stage 2 score and the stacked score, applied before the France guard. The stacker can therefore only remove French matches, never add them. The same rule runs through every French step of the pipeline ([03-lightgbm-stages.md](03-lightgbm-stages.md#6-the-france-guard)).

## 6. The high-confidence recheck

Stage 2 is very confident on most pairs, but some decoys still reach 0.99 and above, where the band never looks. The recheck takes the 429,125 US and Indian pairs that are a record's best candidate with a stage 2 score in (0.99, 0.999]:

1. ce1 and ce2 score them.
2. A second stacker with 7 inputs (logit of stage 2, the ce1 and ce2 logits, logit of stage 1, address missing, source, blocking rank), trained on the validation band with the same settings, turns that into a probability.
3. Each pair keeps the **lower** of its old score and the stacked score.

The recheck can only lower a score. On the validation slice it adds +0.0001.

It uses ce1 and ce2, the two models that run on the Mac. The Kaggle notebook script deletes the kbase and klarge weights once the bands are scored, so only their band scores exist. The recheck is limited to US and India because France has no labels to validate it on.

## 7. What did not help

| Experiment | Result | Decision |
| --- | --- | --- |
| **ce3**: a third e5-small, trained on the remaining 642,272 rows | Alone it beat ce2 (band AUC 0.9525 against 0.9457). On top of ce1, ce2 and kbase it moved the stacked AUC only from 0.9828 to 0.9830 and gave no F0.5 gain | dropped |
| **kbge**: `BAAI/bge-reranker-v2-m3` (568M, Apache 2.0) fine-tuned on 200,000 pairs on a Kaggle P100 as a fifth cross-encoder | Alone 0.9520. The five-model stack scored 0.9832 AUC against 0.9834 for four, and 0.98841 local F0.5 against 0.98844 | not used; the four-model v21 stayed the final upload |
| **A 95-feature stacker**: the stacker inputs plus every stage 1 pair feature and every odd-one-out feature | +0.00004 once ce2 was in | not worth the complexity; the small stacker stayed |

The pattern is the same in all three: once a few strong scores were in the stack, more inputs trained on the same pairs mostly repeated what it already knew. klarge was the one late addition worth keeping, and even it added only +0.00013.

## 8. Practical notes

### Frozen word embeddings

The multilingual vocabulary makes the word-embedding matrix the largest block of an E5 checkpoint. In e5-small it is 250,037 x 384 = 96,014,208 of the 117,654,145 weights in our cross-encoder, 82%. Freezing it (`--freeze-embeddings`, the default in `train_cross_encoder.py`) leaves about 21.6M trainable weights and removes the matrix's gradients and AdamW state. ce1, ce2 and klarge were trained this way; kbase trained every weight.

### Throughput: M2 against T4

| Job | Hardware | Precision | Batch | Pairs | Time or rate |
| --- | --- | --- | --- | --- | --- |
| Train ce1 (e5-small) | Apple M2, MPS | fp32 | 64 | 250,000 | 4,013 s, 62 pairs/s |
| Train ce2 (e5-small) | Apple M2, MPS | fp32 | 64 | 500,000 | 7,324 s, 69 pairs/s |
| Train the French-adapted cross-encoder (e5-small) | Apple M2, MPS | fp32 | | 100,000 | 1,631 s, 62 pairs/s |
| Train kbase (e5-base) | Kaggle T4 | fp16 | 128 | 1,392,272 | 10,878 steps, 245 pairs/s |
| Score the test band with one e5-small | Apple M2, MPS | fp16 autocast | 256 | 902,470 | 1,268 to 1,274 s |
| Score the test band with kbase | Kaggle T4 | fp16 | 512 | 902,470 | 468 s |
| Score the validation band with kbase | Kaggle T4 | fp16 | 512 | 82,133 | 40 s |

The T4 trained a model 2.4 times larger than e5-small at 3.5 to 4 times the Mac's rate, and scored the band in about a third of the time. On the Mac, training runs in fp32 (the training loop only enables fp16 on CUDA), while scoring uses fp16 autocast on both MPS and CUDA.

Two small things kept the Mac runs usable:

- **Length-sorted scoring.** Pairs are sorted by token length and padded per batch, so a batch of short names is not padded to 128 tokens.
- **Checkpoints.** `train_cross_encoder.py` saves a checkpoint every 3,000 steps and can resume or continue from any weights with `--init`, which is also how ce2 was built on top of ce1.

### The e5-large out-of-memory fix

The first klarge attempt, at batch 64, ran out of memory on the T4. The run that worked froze the word embeddings and halved the batch to 32, with learning rate 2e-5 on the first 900,000 pairs:

```bash
python gpu_ce.py --data . --out out --model intfloat/multilingual-e5-large \
    --epochs 1 --batch 32 --lr 2e-5 --pairs 900000 --freeze-embeddings --tag klarge
```

### Running on free Kaggle GPUs

1. Upload a private Kaggle dataset with `gpu_ce.py` and three parquet files: `train_pairs.parquet` (`q_text`, `s_text`, `label`), `valid_band.parquet` and `test_band.parquet` (`rid`, `s1`, `q_text`, `s_text`). The notebook receives only these, built from the challenge files.
2. `run.py` is the notebook script. It finds `gpu_ce.py` under the input folder, installs `transformers` and `pyarrow`, trains, scores both bands and deletes the `.pt` weights, so only the score files (`rid`, `s1`, `ce`) remain.
3. Push it with the Kaggle CLI on a T4 (`kaggle kernels push`, accelerator `NvidiaTeslaT4`) with GPU and internet enabled; both need a phone-verified Kaggle account. Internet is used only to install the packages and download the public weights.
4. Copy the score files next to the local ones as `valid_<tag>.parquet` and `test_<tag>.parquet` and add the tag to the stacker.

### Determinism

Cross-encoder training is seeded, but GPU kernels are not bit-for-bit deterministic, and the kbase run on Kaggle did not fix the PyTorch seed. A rebuilt model gives slightly different scores, so a small number of borderline assignments can differ between runs and machines. The LightGBM stacker itself is deterministic for fixed inputs.

## 9. Other places the cross-encoders are used

- **Rescue.** Records that blocking missed get a second candidate search; each candidate is scored by ce1 and ce2, and a 13-feature LightGBM acceptance model built on those logits decides. The same holds for the later name-key and hc searches.
- **The French-adapted cross-encoder.** ce1 was trained further on 100,000 French test pairs whose labels come from our own rules (confident pairs as positives, low-scoring pairs and synthetic decoys as negatives), never from ground truth. It flagged French type-word-swap decoys, and 922 of them were removed after hand-read and fingerprint gates. See Section 4.9 of [methodology.md](methodology.md).

## 10. Timeline on the final day (27 Sep, IST)

| Time | Step | Result |
| --- | --- | --- |
| about 09:50 | v11: ce1 stack and expected-F0.5 selection | local 0.98716; leaderboard 0.982 |
| 12:30 | v12: ce1 and ce2 | local 0.98777 |
| 16:20 | v15: kbase from Kaggle in the stack, France rules, rescue | local 0.98901; leaderboard 0.986784 |
| 20:25 | v19: klarge in the stack, high-confidence recheck, France push | +0.00013 and +0.0001 locally |
| 23:10 | kbge checked as a fifth model | no gain; the four-model stack stays |
| 23:36 | v21 scored | leaderboard 0.988642 (final) |

## 11. Reproducing the cross-encoders

From the repository root, after stage 2 ([reproduction.md](reproduction.md), steps 5 to 8):

```bash
python src/build_ce_pairs.py train valid test

python src/train_cross_encoder.py ce1 --rows 250000 --lr 3e-5
python src/train_cross_encoder.py ce2 --offset 250000 --rows 500000 --lr 2e-5 --init ce1
python src/train_cross_encoder.py kbase --model intfloat/multilingual-e5-base --rows 1392272 \
    --batch 128 --lr 3e-5 --no-freeze-embeddings --shuffle --device cuda

python src/score_cross_encoder.py ce1
python src/score_cross_encoder.py ce2
python src/score_cross_encoder.py kbase --device cuda

python src/train_stacker.py ce1 ce2 kbase
```

`train_stacker.py` prints the AUC of every input and the stacked out-of-fold AUC, then writes the stacked test scores. It reproduces the three-model v15 stack; the final four-model stack with klarge and the high-confidence recheck are built by the chain in `src/final_steps/` (`stack/`, `recheck/`, `kaggle/`), whose `README.md` lists the exact commands.

| File | Role |
| --- | --- |
| [`src/cross_encoder.py`](../src/cross_encoder.py) | model, tokenization, length-sorted batched scoring |
| [`src/build_ce_pairs.py`](../src/build_ce_pairs.py) | training table and the validation and test bands |
| [`src/train_cross_encoder.py`](../src/train_cross_encoder.py) | fine-tuning on MPS, CUDA or CPU, with checkpoints |
| [`src/score_cross_encoder.py`](../src/score_cross_encoder.py) | scores a band with a trained model |
| [`src/train_stacker.py`](../src/train_stacker.py) | the LightGBM stacker, out-of-fold AUC, test scores |
| [`src/final_steps/kaggle/gpu_ce.py`](../src/final_steps/kaggle/gpu_ce.py), [`run.py`](../src/final_steps/kaggle/run.py) | standalone GPU trainer and the Kaggle launcher |
| [`src/final_steps/recheck/`](../src/final_steps/recheck/) | the high-confidence recheck |
