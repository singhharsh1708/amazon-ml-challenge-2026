# 3. The two LightGBM stages

Part 3 of the technical deep dive. Previous: [2. Blocking](02-blocking.md). Next: [4. Cross-encoders](04-cross-encoders.md).

After blocking, every Source 2/3 record has at most 10 Source 1 candidates: 19,225,118 pairs on test. Two gradient-boosted models score them.

- **Stage 1** judges each pair on its own, from 60 hand-built features.
- **Stage 2** scores the pair again with information no single pair can see: how the record's other candidates scored, how many other records chose the same Source 1 entity, and whether this record is the odd one out among them.

This page covers both models, the data splits that keep their scores honest, and a leak we introduced into stage 2 and then removed. The cross-encoders that read the uncertain pairs after stage 2 are covered in [04-cross-encoders.md](04-cross-encoders.md). The full write-up is in [methodology.md](methodology.md).

## At a glance

| | Stage 1 | Stage 2 |
| --- | --- | --- |
| Question | Is this record the same business as this candidate? | Given everything else pointing at this entity, is this record still the same business? |
| Inputs | 60 pair features | stage 1 probability, 26 record and entity context features, 28 odd-one-out features (55 in total) |
| Training data | 7,175,385 pairs (48.9% positive) from records that never touch the validation slice | pairs of the 110,565 validation-slice entities, 2-fold out of fold |
| LightGBM | learning rate 0.1, 127 leaves, 1,000 rounds | learning rate 0.05, 63 leaves, 400 rounds |
| Headline result | plain macro F0.5 0.97970 on its own | band AUC 0.9714, against 0.9415 for stage 1 on the same pairs |
| Code | `build_features.py`, `train_matcher.py` | `odd_features.py`, `build_odd_features.py`, `train_stage2.py` |

## 1. Splits that keep every score out of sample

Each model is only as trustworthy as the data it was scored on, so the training data is split before anything is learned.

**The validation slice.** Every Source 1 entity with `hash(id) % 20 = 0` (DuckDB hash) is held out: 110,565 entities.

**Record parts.** `build_features.py` assigns each training record to one part:

| Part | Rule | Used for |
| --- | --- | --- |
| valid | the record has any candidate in the validation slice | stage 2 training (out of fold), the stacker, every validation number |
| train | otherwise, `hash(record id) % 2 = 0` | stage 1 training |
| unused | the other half of the remaining records | nothing |

The valid part is chosen by record, not by entity. A record with a held-out entity among its candidates is scored against *all* of its candidates, so false merges from records that belong to another entity, or to none, are counted when the slice is evaluated.

**Why the order matters.** Stage 1 never sees a pair whose record touches the slice, so its probabilities on the slice are out of sample, just as they are on test. Stage 2 uses those probabilities as features, so it is trained on the slice, where they look like test. Its own scores are then taken out of fold (2 folds, split by record), and those out-of-fold scores feed everything downstream: the cross-encoder band, the stacker, the decision rules and the rescue models.

**The metric.** Validation uses the challenge's macro F0.5 per Source 1 entity with one change: a false positive from a record that has no true entity counts with weight 1.887, because test has 1.887 times as many such records per entity as train. Section 5 of [methodology.md](methodology.md) explains the derivation. This weighting replaced the synthetic copies described in [section 5](#5-the-leak-exact-duplicate-decoy-copies) below.

## 2. Stage 1: the pair classifier

### Features

Sixty features are computed for every record and candidate pair. Country is deliberately not one of them, so French pairs are scored exactly like the US and Indian pairs the model was trained on.

| Family | Count | Features | What it captures |
| --- | --- | --- | --- |
| Blocking | 12 | blocking score, number of shared keys, rank; the record's best and second-best score and candidate count; ratio and gap to the record's best; the record's margin between its first and second candidates; how many records list this entity as a candidate or as their top candidate; ratio to the entity's best blocking score | how strongly the rare-key search already links the pair, and how crowded both sides are |
| Name similarity | 7 | RapidFuzz ratio, token sort, token set, partial ratio and Jaro-Winkler on the core names; ratio and token set on the full normalized names | spelling, word order, dropped words and legal suffixes |
| Address similarity | 4 | RapidFuzz ratio, token sort, token set and partial ratio | reordered, abbreviated or partial addresses |
| Token overlap | 3 | name token Jaccard, first name token equal, address token Jaccard | set-level agreement |
| House numbers | 7 | numbers in common, number count on each side, all record numbers present in the candidate, first numbers equal, absolute difference of the first numbers, record numbers missing from the candidate | the single most common decoy edit |
| Legal form | 4 | clash, same form, number of legal forms on each side | Pvt vs Ltd is noise, SAS vs SNC is a different company |
| Flags and lengths | 8 | source (2 or 3), transliterated name, missing address on each side, name and address lengths on each side | how much text there is to compare |
| Relative | 15 | name plus address token-set sum ("combined"); for seven key similarities, the gap to the best value among the record's candidates and the rank among them | a pair is only as good as its alternatives |

**Legal forms.** Forms from all three countries are mapped to shared codes (Pvt, Private, Ltd and Limited to one code; Inc and Incorporated; Corp and Corporation; LLC, LLP, SARL, SAS, SASU, EURL, SA, SNC, SCI). The clash flag fires when both names carry a legal form and none of them agree. Among hard training pairs it fires on 6% of wrong pairs but only 0.02% (India) and 0.5% (US) of true pairs. It is also how many French decoys differ from the real entity:

```
record:    Chevaux Ecole SASU
Source 1:  Chevaux Ecole E.U.R.L.
```

**Relative features.** A token-set score of 90 means little if another candidate for the same record scores 100. The gap and rank versions let the model judge each pair against the record's own alternatives, which matters because the decision later keeps only one Source 1 entity per record.

### Model

| Setting | Value |
| --- | --- |
| Objective | binary |
| Learning rate | 0.1 |
| Leaves | 127 |
| min_data_in_leaf | 100 |
| Feature fraction, bagging fraction | 0.8, 0.8 (bagging every round) |
| L2 | 1.0 |
| Rounds | up to 1,000, early stopping (patience 50) on a random 5% holdout; all 1,000 were used |
| Seed | 42 |

Training uses the train part only: 7,175,385 pairs, 48.9% of them positive. The positive rate is high because blocking already prunes hard: a record keeps only candidates that score at least half of its best candidate, which leaves 1.73 candidates per training record.

### Result

On its own, with the best candidate per record and a threshold of 0.65, stage 1 reaches a plain macro F0.5 of **0.97970** on the 110,565 validation entities. On the 82,133 validation pairs where stage 2 is least sure (the cross-encoder band), its ranking AUC is 0.9415.

## 3. Stage 2: record, entity and odd-one-out context

### Why a second stage

Many unmatched Source 2/3 records are decoys: copies of a real business with one detail changed. The house number moves by a small offset (1788 instead of 1781), the legal form changes ("Soutien Amicale SNC" next to the real "Soutien Amicale SAS"), or one word is added ("... Enterprises", "... Holdings").

Seen alone, a decoy with a house number off by 7 looks like an ordinary typo copy. The evidence that gives it away is somewhere else: in the other records that point at the same Source 1 entity. If two of them carry 1781 and this one carries 1788, this record is the odd one out. Stage 1 cannot see the other records. Stage 2 can.

### Features (55)

| Group | Count | Features |
| --- | --- | --- |
| Stage 1 score | 1 | stage 1 probability `p` |
| Record context | 5 | gap to the record's best probability, margin over the record's second-best (or gap to its best, for non-best pairs), the second-best probability, number of candidates, whether this pair is the record's best |
| Entity context | 6 | how many *other* records have this entity as their best candidate, how many do so with probability at least 0.5, their summed and maximum probability, total pairs pointing at the entity, the gap in combined name and address similarity to the entity's best pair |
| Stage 1 features reused | 15 | combined, combined gap, name token set, address token set, full-name ratio, first numbers equal, numbers subset, numbers in common, transliterated name, missing address, source, rank, score ratio, legal clash, first-number difference |
| Odd-one-out | 28 | see below |

The first 27 of these (everything except the odd-one-out block) are the complete feature set of the earlier stage 2 model that is still used as the France guard ([section 6](#6-the-france-guard)).

### The odd-one-out features

For each pair, `odd_features.py` looks at the pair's **siblings**: the other records whose best stage 1 candidate is the same Source 1 entity, with stage 1 probability at least 0.5. It then asks whether this record agrees with them or stands out. The features are computed for every pair with a stage 1 probability of at least 0.001; the rest are left missing, which LightGBM handles natively.

| Group | Count | Features | What they ask |
| --- | --- | --- | --- |
| Sibling counts | 3 | siblings at probability 0.5 or more; siblings at 0.9 or more; of those, how many come from the other source | Is this entity already well supported? |
| Text against siblings | 6 | best name and address similarity to any sibling, the same relative to the similarity to the Source 1 entity, number of siblings with an identical name or address | Does this record look more like its siblings or less? |
| House numbers | 11 | difference of the first house numbers (clipped to plus or minus 100) and its class (equal, a decoy offset, +1 or +2, -1 or -2, other); whether any unmatched number pair differs by a decoy offset, by +1 or +2, by -1 or -2; how many siblings share the record's first number and the entity's first number; an "odd number" flag; how many record numbers are missing from the entity, how many of those a sibling carries, and how many stay unexplained | Did this record move the house number the way decoys do? |
| Unit suffixes | 2 | the record adds a number-with-letter token (a unit such as `5b`), and whether a sibling carries it | A new unit is suspicious unless siblings have it too |
| Name words | 4 | real words the record adds; added words no sibling carries; words it removes; removed words that every sibling keeps | Is the extra or missing word unique to this record? |
| Learned word lists | 2 | the record adds a learned decoy word; the record adds a learned replacement word | Does it use the vocabulary decoys use? |

Details that matter:

- **Decoy offsets.** The offsets are the set {3, 4, 5, 7, 9, 11, 13, 21}, the same set used by `decoy_families.py`. Differences of +1, +2, -1 and -2 have their own flags, so the model can treat them differently from the decoy offsets.
- **The odd-number flag** fires when the record's first house number differs from the entity's, at least one sibling shares the entity's number, and no sibling shares the record's.
- **Unexplained numbers.** A record number counts as explained when the entity carries it, a sibling carries it, it is part of a longer entity number, or joined to its neighbour it spells an entity number. Only what is left counts as unexplained.
- **Real words.** An added or removed word must have at least 2 characters, not be a digit, a legal form or a stop word (English and French function words, and honorifics such as `shri`, `smt` and `m/s` fragments), and must not fuzzily match (RapidFuzz ratio 80 or more) or sit inside the other name. This keeps legal-form swaps and spelling noise out of the word features.
- **Word lists.** The decoy words (65 for India, such as `public`, `industries`, `enterprises`; 27 for the US, such as `group`, `holdings`, `southside`) are learned by `learn_decoy_words.py` from training pairs outside the validation slice. The replacement words come from `decoy_families.py`. The same decoy list drives the decoy-word veto in the decision layer.

A short example of the pattern these features target (text as it appears in the methodology's error analysis):

```
Source 1 entity:  toledo womens health, 1781 hamilton st
sibling record:   ... 1781 hamilton st
this record:      toledo womens health, 1788 hamilton st
```

Here the first-number difference is 7, which is a decoy offset; the siblings share the entity's number and none shares the record's, so the odd-number flag is 1. This exact pattern is still listed among the common false positives of the final system ([methodology.md](methodology.md), Section 6): the features help most when siblings exist to compare against.

At test time the siblings come from the test set's own stage 1 scores. No labels are involved; the unlabelled test records are used the same way as the training records.

### Model and training

| Setting | Value |
| --- | --- |
| Learning rate | 0.05 |
| Leaves | 63 |
| min_data_in_leaf | 200 |
| Feature fraction, bagging fraction | 0.9, 0.8 (bagging every round) |
| L2 | 1.0 |
| Rounds | 400, no early stopping |
| Seed | 42 |
| Training rows | pairs whose Source 1 entity is in the validation slice |
| Evaluation | 2-fold cross-validation split by record; out-of-fold scores are kept |
| Test model | refit on the whole slice |

The context features are computed over every pair of the valid part, so a slice pair sees its siblings and competitors even when they point at entities outside the slice. Only slice pairs are used as training rows.

### What each stage adds

| Measurement | Before | After | Notes |
| --- | --- | --- | --- |
| Validation F0.5 when stage 2 context was introduced (v2, older unweighted metric) | 0.9664 | 0.9714 | leaderboard 0.940 (v1) to 0.950 (v2), together with new blocking keys |
| Ranking AUC on the 82,133 validation band pairs | 0.9415 (stage 1) | 0.9714 (stage 2, out of fold) | the pairs where stage 2 is least sure |
| Plain F0.5, 27-feature stage 2 retrained without synthetic rows | stage 1 | +0.0036 | bootstrap interval +0.0033 to +0.0039 |
| Test-weighted F0.5, odd-one-out stage 2 plus house-number sibling rule | 0.98241 (the 0.974 file's configuration) | 0.98470 (v10) | US and India only; the French parts of v10 do not touch this number |

The house-number sibling rule in the decision layer is the hand-written counterpart of the odd-number feature: if another record from the same source is assigned to the same entity and shares the entity's first house number, a record with probability below 0.95 whose first number differs is dropped (4,001 test assignments).

## 4. How stage 2 is scored on test

Stage 1 scores all 19,225,118 test pairs (`predict_submission.py stage1`). `build_odd_features.py test` builds the odd-one-out features from those scores. `predict_submission.py` then computes the record and entity context in DuckDB and scores stage 2 and the France guard together, in 8 slices of Source 1 ids with DuckDB capped at 4 GB. Both scores go to one file, which the cross-encoder band and the decision layer read.

## 5. The leak: exact duplicate decoy copies

### What we did and why

Test is more decoy-heavy than train: 1.887 times as many unmatched Source 2/3 records per Source 1 entity. We wanted stage 2 to learn its context features at test-like density. So `train_stage2.py` appended synthetic unmatched rows: for 90% of the unmatched records in the validation part, an exact copy with a new id, label 0, the same Source 1 candidates, the same stage 1 probabilities and the same features. Each copy sat in the same fold as its original, and the entity context (how many records chose this entity, the best probability among them) was computed over originals and copies together. The same copies also served as extra false positives in the "test-density" score the script printed.

### Why it leaked

Only unmatched records were copied. After augmentation, "another record at my entity has exactly my probability" meant "I have a twin, so I am a decoy". Nothing like that exists in real data, where the same pattern points the other way:

| Pattern: the record is its entity's best match and another best-matching record has exactly the same probability | Rows | Share that are true matches |
| --- | --- | --- |
| Real validation data, no copies | 13,845 | 99.6% |
| With the copies added | 19,232 | 71.7% |

So the copies did two kinds of damage. They taught the model a "twin means decoy" rule that can never fire on test, because test has no synthetic twins. And they diluted a genuinely strong positive signal, so the model learned to trust it less than it deserved. The evaluation had the same flaw: the test-density score counted the copies as false positives, so the model was rewarded for spotting exactly the rows it had learned to spot.

The scale was not small. The out-of-fold file held 5,014,674 rows, of which 1,592,192 were synthetic (266,619 synthetic records). 89.9% of the 296,554 unmatched records had a twin; no matched record did.

### How we found it

Our local scores kept running ahead of the leaderboard: validation sat about 0.014 above the public score. We audited every input to the stage 2 out-of-fold file, counted the synthetic rows, compared the twin pattern with and without copies, and retrained stage 2 without them.

| Measurement | With copies | Without copies |
| --- | --- | --- |
| Gain reported for the augmented model (v6) | +0.003 locally | -0.0017 once re-scored without synthetic rows |
| Plain out-of-fold F0.5 at threshold 0.75, 27-feature stage 2 | 0.98422 | 0.98280 (retrained without copies) |
| The same with the decoy-word veto | 0.98463 | 0.98342 |

The augmented model looked better only because the evaluation contained the rows it had learned to recognise.

### The fix

- **No synthetic rows in training.** The copy share in `train_stage2.py` (`COPY_TENTHS`) defaults to 0. The odd-one-out stage 2 that replaced the leaky model (v10 onward) is trained on real rows only.
- **Density moved into the metric.** Instead of cloning records, validation weights false positives from records with no true entity by 1.887. After the switch, local and leaderboard gains moved together: v11 gained +0.0048 locally and +0.008 on the leaderboard over the 0.974 file.
- **No synthetic rows in evaluation.** Every validation number after the fix is computed on data with no synthetic rows.
- **The leaky model is fenced off.** It survives only as the France guard (next section), where it can remove French matches but never add one.

The model behind our 0.974 upload (v6 plus the decoy-word veto) was the leaky one. The next upload, v11 at 0.982, combined the copy-free odd-one-out stage 2 with the first cross-encoder stack and expected-F0.5 selection.

**Lesson.** Never augment with exact copies of negatives when any feature compares a row with its neighbours, and never let synthetic rows into the evaluation. If test has a different class balance, model that shift in the metric (or in loss weights), not by cloning rows.

## 6. The France guard

France is 15% of test and has no labels, and some stage 2 changes that helped US and India could not be checked there. A French record is therefore kept only when two models agree:

- the final stage 2 model, and
- the earlier stage 2 model without the odd-one-out features (v6, 27 features, trained with the duplicate copies above).

Both must pick the same Source 1 entity, and the record's score becomes the lower of the two. If they disagree, the score is 0. French pairs then face a fixed threshold of 0.85. Because the guard takes a minimum, it can only remove French matches, never add them, so the leak in the older model cannot create false merges. The cross-encoder stacker follows the same rule for France ([04-cross-encoders.md](04-cross-encoders.md)).

## 7. Reproducing the stages

From the repository root, after normalization and blocking ([reproduction.md](reproduction.md), step 1):

```bash
python src/build_features.py train
python src/build_features.py test
python src/train_matcher.py
python src/predict_submission.py stage1

python src/learn_decoy_words.py
python src/decoy_families.py
python src/train_stage2.py            # 27-feature model, kept as the France guard
mkdir -p models/v6
cp models/stage2.txt models/stage2.json models/v6/

python src/build_odd_features.py train
python src/build_odd_features.py test
python src/train_stage2.py            # 55-feature model with the odd-one-out block
```

Our guard was trained with the copies switched on; the shipped default is 0, so a rebuilt guard can move French results slightly. The stage 2 model used for the submission was trained by an earlier version of `train_stage2.py` with the same features and settings. Delete the cached `data/features/test_predictions*.parquet` files after retraining either stage, otherwise the old scores are reused.

## 8. Code map

| File | Role |
| --- | --- |
| [`src/build_features.py`](../src/build_features.py) | the 60 pair features and the train, valid and test parts |
| [`src/train_matcher.py`](../src/train_matcher.py) | stage 1 LightGBM and its validation scores |
| [`src/learn_decoy_words.py`](../src/learn_decoy_words.py), [`src/decoy_families.py`](../src/decoy_families.py) | the learned decoy and replacement words |
| [`src/odd_features.py`](../src/odd_features.py) | the 28 odd-one-out features |
| [`src/build_odd_features.py`](../src/build_odd_features.py) | computes them for the validation slice or for test |
| [`src/train_stage2.py`](../src/train_stage2.py) | stage 2 context features, out-of-fold training and the final model |
| [`src/predict_submission.py`](../src/predict_submission.py) | test scoring of both stages and the France guard |
