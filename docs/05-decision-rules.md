# 5. Decision rules: from scores to a submission

By the time the decision layer runs, every candidate pair has a probability: stage 2 for most pairs, the cross-encoder stack for the uncertain band (see [04-cross-encoders.md](04-cross-encoders.md)). The decision layer turns the 19.2 million scored test pairs into one list of matches per Source 1 entity.

It is a small amount of code (`write_outputs` in [`src/predict_submission.py`](../src/predict_submission.py), plus [`src/expected_f.py`](../src/expected_f.py) and [`src/decoy_veto.py`](../src/decoy_veto.py)), but it decides where the precision and recall trade-off lands under F0.5. The upload that introduced the decoy-word veto gained 0.006 on the public leaderboard.

This page covers the rules shared by all countries, the thresholds, and the validation metric used to tune them. France has its own rules because it has no labels; they are in [06-france.md](06-france.md). The rescue passes that add records after this layer are in [07-rescue.md](07-rescue.md), and the validation setup is described in full in [08-validation.md](08-validation.md).

## Contents

- [Why the decision layer matters under F0.5](#why-the-decision-layer-matters-under-f05)
- [Order of operations](#order-of-operations)
- [1. One owner per record](#1-one-owner-per-record)
- [2. Decoy-word veto](#2-decoy-word-veto)
- [3. Expected-F0.5 selection per entity](#3-expected-f05-selection-per-entity)
- [4. House-number sibling rule](#4-house-number-sibling-rule)
- [5. Thresholds and how they were chosen](#5-thresholds-and-how-they-were-chosen)
- [6. Test-weighted tuning](#6-test-weighted-tuning)
- [7. What we tried and dropped in this layer](#7-what-we-tried-and-dropped-in-this-layer)
- [Code map](#code-map)

## Why the decision layer matters under F0.5

The score is F0.5 computed per Source 1 entity and then averaged over all entities, singletons included. Written with counts, for one entity:

```
F0.5 = 1.25 * TP / (1.25 * TP + 0.25 * FN + FP)
```

A false positive sits in the denominator with weight 1 and a false negative with weight 0.25. For an entity with three correct matches:

| Change | Precision | Recall | F0.5 | Loss |
|---|---|---|---|---|
| Nothing wrong | 1 | 1 | 1.000 | 0 |
| Miss a fourth true match | 1 | 3/4 | 0.9375 | 0.0625 |
| Add one wrong record | 3/4 | 1 | 0.789 | 0.211 |

One wrong merge costs about 3.4 times as much as one missed match. Singletons are stricter still: an entity with no true matches scores 1 when its list is empty and 0 as soon as anything is assigned to it. In the training data 123,247 of 2,206,821 Source 1 entities (5.6%) are singletons.

Two consequences shaped every rule on this page:

- **Decoys are the main risk.** Many unmatched Source 2/3 records copy a real business and change one detail (a word, the legal form, the house number). Each one that gets merged costs more than a missed copy.
- **The right cut-off depends on the entity.** Whether one more uncertain record is worth adding depends on what the entity already has. A single global threshold cannot express that (Section 3).

## Order of operations

The rules run in this order. Counts are from the v15 test build ([methodology, Appendix B](methodology.md)); the final v21 file adds later change sets on top (see [reproduction step 13](reproduction.md)).

| Step | Rule | Countries | Effect on test (v15) |
|---|---|---|---|
| 1 | Best Source 1 candidate per record, probability at least 0.01 | all | one owner per record |
| 2 | Decoy-word veto | US, India | 1,444 assignments vetoed |
| 3a | Expected-F0.5 selection per Source 1 entity | US, India | 4,963,564 of 5,143,313 kept |
| 3b | Fixed threshold 0.85 on the guarded score | France | see [06-france.md](06-france.md) |
| 4 | House-number sibling rule | all | 4,001 dropped |
| 5 | French exact-address additions | France | 9,360 added |
| 6 | France audit rules | France | 14,970 removed |
| 7 | France audit additions and rescue for blocking misses | France, then US and India | 34,798 added |
| | Result | | 5,817,948 matches, 100,309 empty Source 1 entities |

Every addition in steps 5 to 7 goes only to a record that is still unassigned, so each record keeps exactly one Source 1 entity.

## 1. One owner per record

In the training ground truth no Source 2/3 record is matched to more than one Source 1 entity. We use that as a hard constraint and resolve from the Source 2/3 side: each record keeps only its highest-scoring Source 1 candidate, or nothing.

```sql
SELECT rid, arg_max(s1, (p, -s1)) AS s1, max(p) AS p
FROM predictions GROUP BY rid
```

- This turns entity resolution into an assignment problem. There is no pairwise clustering and no transitive closure to go wrong.
- Ties go to the smaller Source 1 id, so the output is deterministic. On the v15 scores this mattered for exactly 2 Indian records whose two best candidates had identical scores.
- Pairs below 0.01 are dropped here. Everything after this step works on at most one pair per record.

## 2. Decoy-word veto

Decoys often take a real business name and add one word: "... Enterprises", "... Holdings", "... Group". The veto learns those words from labelled data and refuses any assignment whose record name adds one of them to the Source 1 name.

### How the words are learned

[`src/learn_decoy_words.py`](../src/learn_decoy_words.py) looks at best-candidate training pairs outside the validation slice where the record name contains every core word of the Source 1 name, so the record only adds words. It counts each added word separately for true pairs and for records that match nothing.

| Criterion | Value |
|---|---|
| Added to a real name by records that match nothing | at least 200 times |
| Added by true pairs | at most 0.2 per 100 such decoy uses (`MAX_TRUE_SHARE = 0.002`) |
| Scope | per country |
| Result | 65 words for India (for example `public`, `industries`, `enterprises`), 27 for the US (for example `group`, `holdings`, `southside`) |

A record is vetoed when its name contains a learned word for its country that the Source 1 name does not ([`src/decoy_veto.py`](../src/decoy_veto.py), `veto_flags`). France has no list, because France has no labels.

### Evidence

| Measurement | Value |
|---|---|
| Held-out validation, stage 2 scores at 0.75, no synthetic rows | +0.00062 macro F0.5 (95% CI +0.00053 to +0.00072) |
| Precision on that slice | 298 assignments vetoed, 8 of them true matches (97% precise) |
| Stage 1 scores at 0.65, plain and with test-like decoy density simulated | +0.0011 and +0.0017 |
| Public leaderboard | the upload that introduced it (v6) moved from 0.968 to 0.974; it vetoed 28,525 test assignments |

The leaderboard gain was about ten times the local gain. The reason shows up in how often the learned words occur among mid-confidence best candidates (stage 1 probability 0.65 to 0.99):

| Country | Validation slice | Test |
|---|---|---|
| US | 0.68% | 1.74% |
| India | 2.54% | 5.52% |

Test carries 2.2 to 2.6 times as many decoy-word hits as validation. Validation was decoy-poor, so it understated every rule that removes false positives. The test-weighted metric in Section 6 addresses the same problem.

By v15 the veto fired on only 1,444 test assignments. The stage 2 odd-one-out features include the same learned decoy and replacement words, so most of these records probably no longer reach the decision with a high score, and the veto acts as a backstop.

### What we did not veto

`decoy_veto.py` also implements a generic word-swap veto: the record adds a frequent real word and loses another. It is switched off for every country (`SWAP_COUNTRIES = set()`). On the labelled slice, 90.8% (US) and 87.5% (India) of the pairs it would remove were true matches, and turning it on cost 0.0007 (stage 1 scores). In US and Indian data a swapped word is usually noise. France is different, and its type-word swap rule is narrower (see [06-france.md](06-france.md)).

## 3. Expected-F0.5 selection per entity

### The problem with one global threshold

Suppose an entity already has `m` matches we are sure of, and one more record with calibrated probability `p` points at it. Adding the record is worth it only if `p` beats a break-even value that depends on `m`:

| Sure matches already kept (m) | 0 | 1 | 2 | 3 | 10 | many |
|---|---|---|---|---|---|---|
| Break-even probability | 0.500 | 0.727 | 0.759 | 0.771 | 0.791 | 0.800 |

With `m = 0` a miss and a false merge both score 0, so anything above even odds is worth adding. With many sure matches, a false merge costs about four times what a miss costs (the 1 against 0.25 in the formula), so the bar approaches 0.8. A global threshold has to pick one number for all of these cases.

### What the selection does, in plain words

For each Source 1 entity ([`src/expected_f.py`](../src/expected_f.py)):

1. Take the records whose best candidate is this entity and that survived the veto, sorted by probability. At most 12 are considered (the largest entity in the training ground truth has 11 matches).
2. Treat each record as an independent coin flip that comes up "true match" with its probability.
3. For every `k` from 0 to the number of records, imagine keeping the top `k`:
   - the number of true records among the kept `k` follows a Poisson-binomial distribution (the count of heads for coins with different probabilities); so does the number of true records among the ones dropped;
   - for each combination of `i` true kept and `j` true dropped, the entity's F0.5 is `1.25 i / (1.25 i + 0.25 j + (k - i))`, which is 0 when nothing kept is true;
   - keeping nothing scores 1 if nothing is true (the entity was a singleton) and 0 otherwise;
   - average these scores over the two distributions to get the expected F0.5 of keeping `k`.
4. Keep the `k` with the highest expected F0.5.

Under independence the best set is always a top-`k` prefix, so trying every `k` is exact. With at most 12 records per entity this is a small loop per entity.

The selection only reasons about records that chose the entity. True matches that never picked it (blocking misses) are outside its view.

### Worked examples

Computed with `best_count` and `poisson_binomial` from `src/expected_f.py`. The inputs are illustrative probabilities, not records from the data.

| Probabilities of the records pointing at one entity | Expected F0.5 for keeping 0, 1, 2, ... | Kept |
|---|---|---|
| 0.55 | 0.45, **0.55** | 1 |
| 0.45 | **0.55**, 0.45 | 0 |
| 0.6, 0.6 | 0.16, 0.54, **0.6267** | 2 |
| 0.9, 0.6, 0.3 | 0.028, **0.7727**, 0.7516, 0.6397 | 1 |
| 0.99, 0.99, 0.99, 0.75 | 0, 0.6428, 0.8453, **0.9449**, 0.9409 | 3 |
| 0.99, 0.99, 0.99, 0.80 | 0, 0.6383, 0.8416, 0.9418, **0.9514** | 4 |

A 0.6 record is kept when two of them point at an entity and dropped when it sits next to a 0.9 record: once the entity has a likely match, a 0.6 record is below the break-even. A 0.75 record is dropped from an entity with three near-certain matches and a 0.80 record is kept, as the break-even table predicts.

### Settings

| Setting | Value |
|---|---|
| Countries | US and India (France uses a fixed 0.85, see [06-france.md](06-france.md)) |
| Minimum probability to be considered | 0.01 |
| Maximum records per entity | 12 |
| Decoy weight (deflates every record's odds by this factor before the computation) | 1.0, that is no deflation |

France is left out. The computation treats scores as calibrated probabilities, which we could check on US and Indian validation data but not on France. There was also direct evidence that French scores behave differently: a label-shift estimate of the French matched share drifts from 0.630 to 0.581 depending on the threshold used, while the US and Indian estimates are stable to within 0.003.

### What it did, and what we know about it

- On test (v15) it kept 4,963,564 of the 5,143,313 candidate assignments for US and India, dropping 179,749.
- It shipped in v11 together with the first cross-encoder stack. That pair moved local validation from 0.98470 (v10) to 0.98716 (+0.00246). The leaderboard went from 0.974 (v6) to 0.982 with v11, which also carried the v10 changes (odd-one-out stage 2, sibling rule, France guard).
- An earlier measurement on stage 1 scores alone found no gain over the best global threshold (0.97969 against 0.97970). With about 3.5 true matches per entity, most entities sit near the flat end of the break-even table, where one well-placed threshold is already close to optimal.
- We never ran a clean ablation of this step alone on the v11 or later scores, so its separate share of the v11 gain is unknown. Changing its settings later (decoy weight, probability floor) gave no held-out gain.

## 4. House-number sibling rule

Many decoys copy a real business and move its house number by a small offset ("1788 hamilton st" next to the real "1781 hamilton st"). When the same entity also has a correct copy, the decoy is the odd one out.

The rule, applied after the selection above:

> If another record **from the same source** is assigned to the same Source 1 entity and its first house number equals the entity's, then a record whose first house number differs from the entity's is dropped, unless its probability is at least 0.95.

Records without a house number are never dropped by it. "First house number" is the first number in the normalized address (`first_num_equal` from the pair features).

| Measurement | Value |
|---|---|
| Stage 1 validation at 0.65, after the decoy veto, plain and with test-like decoy density simulated | +0.00013 and +0.00045 |
| Share of dropped records that are true matches (validation, stage 1 scores) | 52% to 66% |
| Test assignments dropped (v15) | 4,001 |
| French records dropped in the v17 re-run | 733 |
| Shipped in v10, together with the odd-one-out stage 2 | local 0.98241 to 0.98470 |

The rule pays even though more than half of what it drops is correct, and the break-even table explains why. The dropped records sit on entities that already have another assigned record with the right house number. If that sibling is right, adding one more record needs a probability of about 0.73 or more. A group of records that is true only 52% to 66% of the time is below that line.

Known weaknesses: it penalises a business that really has two addresses, and it assumes the first number in an address is the house number.

## 5. Thresholds and how they were chosen

The main settings in and around the decision layer, and where each came from:

| Setting | Value | Where | How it was set |
|---|---|---|---|
| Minimum probability for a best candidate | 0.01 | step 1 | default; a grid re-tune found no held-out gain |
| Expected-F0.5 records per entity | at most 12 | step 3a | fixed; the largest true entity in train has 11 matches |
| Expected-F0.5 decoy weight | 1.0 | step 3a | grid re-tune, no held-out gain |
| France threshold | 0.85 on min(stage 2, stacked, guard) | step 3b | fixed and conservative, no labels to tune it |
| Sibling rule cut-off | 0.95 | step 4 | grid re-tune, no held-out gain |
| Decoy word | at least 200 decoy uses, at most 0.2% true | word lists | learned on training pairs outside the slice |
| French type word | swapped in at least 150 times | France rules | learned from French pairs, see [06-france.md](06-france.md) |
| Cross-encoder band | rank 1 or 2, stage 2 score in [0.01, 0.99] | stacking | covers the 902,470 uncertain test pairs |
| High-confidence recheck | stage 2 score in (0.99, 0.999], score can only fall | before step 1, US and India ([04-cross-encoders.md](04-cross-encoders.md)) | +0.0001 on validation |
| Rescue acceptance | 0.7 | rescue ([07-rescue.md](07-rescue.md)) | sweep on validation (below) |
| Name-key and hc rescue | 0.7; hc only for records whose candidates all score below 0.3 | rescue | chosen on one half of the slice, scored on the other |
| Reverse blocking | at least 0.6, best competing entity below 0.2 | rescue | same protocol |

The rescue sweep is typical of what tuning looked like near the end: a flat optimum.

| Rescue acceptance threshold | Records added | Of which correct | Test-weighted F0.5 |
|---|---|---|---|
| 0.5 | 1,147 | 1,064 | 0.98899 |
| 0.6 | 1,120 | 1,047 | 0.98899 |
| **0.7 (used)** | 1,095 | 1,033 | **0.98901** |
| 0.8 | 1,058 | 1,005 | 0.98901 |
| 0.9 | 992 | 948 | 0.98897 |
| 0.95 | 939 | 908 | 0.98895 |

Late in the challenge we re-tuned the decision settings together on a grid (expected-F0.5 decoy weight, probability floor, sibling cut-off, rescue threshold). No combination beat the defaults by at least +0.00005 on held-out data, so the defaults stayed. For scale, choosing a threshold on the same slice it is scored on adds roughly 1e-4 of optimism (paired bootstrap), which is why late changes were chosen on one half of the slice and scored on the other.

## 6. Test-weighted tuning

This section is about how the metric was used to tune the decision layer; [08-validation.md](08-validation.md) covers the slice, the out-of-fold setup and the loss decomposition.

### The problem

Early on, local validation and the leaderboard disagreed badly. v3 scored 0.9817 locally and 0.968 on the leaderboard. The gap was not noise: a bootstrap over validation entities gives a standard error of 0.000238, so a gap of 0.0137 is dozens of standard errors wide.

The cause was a difference in decoy density between train and test, which can be computed from the file sizes alone:

| | Train | Test |
|---|---|---|
| Source 2/3 records per Source 1 entity | 4.6765 | 5.7543 |
| True matches per entity | 3.4613 | 3.4613 (assumed equal to train) |
| Unmatched records per entity | 1.2153 | 2.2931 |
| Share of Source 2/3 records that match nothing | 26.0% | about 39.8% |

2.2931 / 1.2153 = **1.887**. Test has about 1.9 times as many unmatched records per entity, and unmatched records are exactly what produces false merges.

The "equal true matches" assumption was checked without labels. A label-shift estimate on stage 1 test scores gives a matched share of 0.6045 (US) and 0.5968 (India), stable across thresholds, against 0.739 on validation. That implies 3.47 true matches per test entity, against 3.46 in train. The test set looks like the training set with extra decoys added.

### The metric

We kept the official macro F0.5 per Source 1 entity, singletons included, with one change: a false positive caused by a record that has **no true Source 1 entity at all** is counted 1.887 times in the denominator.

```
F0.5_weighted = 1.25 * TP / (1.25 * TP + 0.25 * FN + FP_wrong_entity + 1.887 * FP_unmatched)
```

A false positive from a record that belongs to a different entity keeps weight 1. The implementation is `macro_f05` in [`src/train_rescue.py`](../src/train_rescue.py) and `fscore` in [`src/final_steps/lib/eval2.py`](../src/final_steps/lib/eval2.py). We always report the plain F0.5 next to it (for v15: 0.98901 weighted, 0.98923 plain).

The weighting happens only at evaluation time. No synthetic rows are added to training. We had tried the opposite first, adding exact copies of unmatched records to stage 2 training to mimic the test density. It looked like +0.003 locally and was -0.0017 once measured without synthetic rows: only decoys ever had a perfect twin, so the model learned "has a twin, so it is a decoy". Reweighting the metric gets the density right without teaching the model anything false.

### Did it work?

After the switch, local and leaderboard gains moved together, and the remaining gap kept shrinking:

![Local validation against the public leaderboard](../assets/local_vs_leaderboard.png)

| File | Local F0.5 | Public F0.5 | Local minus public |
|---|---|---|---|
| v3 | 0.9817 (older unweighted metric) | 0.968 | 0.0137 |
| v6 + veto | 0.98241 | 0.974 | 0.0084 |
| v11 | 0.98716 | 0.982 | 0.0052 |
| v15 | 0.98901 | 0.986784 | 0.0022 |
| v17 | 0.98901 (US and India unchanged) | 0.9871 | 0.0019 |

Steps agreed too:

- v11 over the 0.974 file: +0.0048 locally, +0.008 on the leaderboard. Same direction, and larger on test, most likely because the cross-encoder helped more on the decoy-heavy test than on validation.
- v21 over v20: +0.00008 predicted from held-out validation, +0.000093 measured on the leaderboard.

### Protocol for late changes

- **Held-out slice.** Every Source 1 entity with `hash(id) % 20 = 0` (DuckDB hash, 110,565 entities) is held out. All Source 2/3 records that have any held-out entity among their candidates are scored, so false merges from records that belong elsewhere, or nowhere, are counted.
- **Out-of-fold only.** Stage 2, the stacker and the acceptance models are trained on the slice with 2-fold cross-validation, and only out-of-fold predictions are evaluated.
- **Split halves.** For the late US and Indian change sets, the threshold was chosen on one half of the slice (`hash(s1) % 40 < 20`) and the gain was reported on the other half. For hc and reverse rescue, every increment was also positive in all four quarters of the slice.
- **Paired comparisons.** Absolute scores have a standard error of about 0.0002, but two decision rules compared on the same entities differ much less noisily (for example 0.80 against 0.75: -0.000174, 95% CI -0.000305 to -0.000029). Changes of 1e-4 are measurable this way.
- **France is out of scope.** The slice covers US and India only. French changes are judged with the label-free evidence in [06-france.md](06-france.md).

## 7. What we tried and dropped in this layer

| Idea | Result |
|---|---|
| Generic word-swap veto (US, India) | removes mostly true matches (87.5% to 90.8% true); -0.0007 |
| Re-tuning floor, decoy weight, sibling cut-off and rescue threshold on a grid | no held-out gain of at least +0.00005 |
| Re-tuning a global threshold for the test density (stage 1 scores) | +0.00013 only; prior correction and per-entity decoding recovered nothing more on stage 1 scores |
| Per-source and per-entity match caps | per-source caps touched only 147 entities; no gain |
| Stage 1 probability floors and caps inside the decision | rejected on validation |
| Synthetic exact-duplicate decoys to mimic test density | a leak: +0.003 locally, -0.0017 measured cleanly |

Three more rule families are implemented in `predict_submission.py` but off by default: `shift` (house number moved by a typical decoy offset), `replacement` (learned replacement words, [`src/decoy_families.py`](../src/decoy_families.py)) and `promoted` (pairs with a low stage 1 score). The decoy offsets and replacement words they use are also inputs to the stage 2 odd-one-out features.

## Code map

| File | What it does here |
|---|---|
| [`src/predict_submission.py`](../src/predict_submission.py) | `write_outputs`: best candidate, veto, expected-F0.5 or France threshold, sibling rule, French additions, France rules, rescue, output files, official validator |
| [`src/expected_f.py`](../src/expected_f.py) | `poisson_binomial`, `best_count`, `select` |
| [`src/decoy_veto.py`](../src/decoy_veto.py) | `veto_flags` (decoy word and word swap), `type_swap_flags` |
| [`src/learn_decoy_words.py`](../src/learn_decoy_words.py) | learns `data/decoy_words.json` |
| [`src/decoy_families.py`](../src/decoy_families.py) | learns replacement words; flags for the optional rule families |
| [`src/address_rules.py`](../src/address_rules.py) | French exact-address additions |
| [`src/france_rules.py`](../src/france_rules.py) | France audit rules and the v17 filter |
| [`src/train_rescue.py`](../src/train_rescue.py) | rescue acceptance model and the test-weighted `macro_f05` |

Run the decision stage alone with `python src/predict_submission.py 0.85` once the scores exist (full order in [reproduction.md](reproduction.md)).
