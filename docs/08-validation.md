# 8. Validation: a local number the leaderboard agrees with

In a 72-hour challenge with a limited number of uploads, the local validation number is the only thing you can iterate against. Early on ours disagreed with the leaderboard by more than a full point. By the end, a change that validation said was worth +0.00008 moved the public leaderboard by +0.000093. This document explains the held-out slice, the reweighted metric that closed the gap, the checks behind it, and where the remaining loss sits.

## 1. Summary

| Piece | Choice |
| --- | --- |
| Held-out slice | every Source 1 entity with `hash(id) % 20 = 0` (DuckDB hash): 110,565 US and Indian entities |
| Records scored | every Source 2/3 record that has any held-out entity among its candidates |
| Metric | official macro F0.5 per Source 1 entity, with false positives from records that match nothing weighted **1.887** |
| Out of fold | stage 2 (2 folds by record), stacker and rescue model (2 folds by Source 1 entity) |
| Late changes | threshold chosen on one half of the slice, gain reported on the other half |
| Checks | independent reference evaluator, official validator, bootstrap and quarter spreads |
| France | no labels: hand-read samples, label-free fingerprints and a Monte Carlo model instead |

## 2. The held-out slice

| Property | Value |
| --- | --- |
| Rule | `hash(source1_id) % 20 = 0`, DuckDB `hash`, DuckDB pinned at 1.5.5 |
| Source 1 entities | 110,565 (the neighbouring buckets 1, 2 and 3 hold 110,119, 110,139 and 110,250) |
| Countries | India 44,350 (40.1%), US 66,215 (59.9%) |
| Singletons (no true match) | 6,201 (5.6%) |
| True pairs | 382,531 (Source 2: 185,172, Source 3: 197,359) |
| Kept by blocking | 375,278 (98.10%) |
| Candidate pairs scored at stage 1 | 3,422,482 from 817,515 records |

**Why records, not just pairs.** Every record that has a held-out entity anywhere in its candidate list is scored, including records whose true entity is outside the slice and records that match nothing. False merges into held-out entities therefore come from the same kinds of records as on test, only in different proportions, which section 3 corrects. Scoring only the true pairs of held-out entities would hide most false positives.

**Who learns from what.** The slice is used in two ways: some models never see it, others are trained on it out of fold.

| Component | Trained on | Validation number comes from |
| --- | --- | --- |
| Stage 1 LightGBM | half of the records whose candidates never touch the slice (7,175,385 pairs, 48.9% positive) | predictions on slice records it never saw |
| Cross-encoders (ce1, ce2, kbase, klarge) | 1,392,272 pairs from the train split, never from the slice | scores on the 82,133 slice band pairs (41.3% true) |
| Decoy-word list | training pairs outside the slice | used as a fixed input |
| Stage 2 LightGBM | slice pairs, 2 folds split by record | out-of-fold predictions |
| Stacker | slice band pairs, 2 folds split by Source 1 entity | out-of-fold predictions |
| Rescue acceptance model | eligible slice pairs, 2 folds split by Source 1 entity | out-of-fold predictions |
| nm1d, hc and reverse models | slice halves, `hash(s1) % 40 < 20` | threshold on one half, gain on the other |

## 3. The metric

### Official definition

Per Source 1 entity, with TP, FP and FN counted over its matched record IDs:

```
F0.5 = 1.25 * P * R / (0.25 * P + R) = 1.25*TP / (1.25*TP + 0.25*FN + FP)
```

The score is the plain average over **all** Source 1 entities. An entity with no true matches scores 1 for an empty prediction and 0 for any prediction. In the denominator a false positive weighs four times as much as a miss (1 against 0.25), so precision dominates.

### The test-weighted version

We split false positives into two kinds:

- **FP_other:** the record belongs to a different Source 1 entity.
- **FP_unmatched:** the record belongs to no Source 1 entity at all. Most of these are decoys.

```
F0.5_w = 1.25*TP / (1.25*TP + 0.25*FN + FP_other + 1.887 * FP_unmatched)
```

Entities with no true matches still score 1 or 0. Written as the SQL used in the gate scripts:

```sql
CASE
  WHEN n_true = 0 THEN greatest(0, 1 - fp_other - 1.887 * fp_unmatched)
  WHEN tp = 0     THEN 0.0
  ELSE 1.25*tp / (1.25*tp + 0.25*(n_true - tp) + fp_other + 1.887*fp_unmatched)
END
```

The plain metric is always reported next to it (v15: 0.98901 test-weighted, 0.98923 plain). How the decision layer is tuned against this metric is in [05-decision-rules.md](05-decision-rules.md).

### Where 1.887 comes from

Test has more records per Source 1 entity than train. Label-shift estimates on the test scores say the extra records are unmatched ones, not extra copies.

| | Train | Test |
| --- | --- | --- |
| Source 1 entities | 2,206,821 | 1,732,544 |
| Source 2 + Source 3 records | 10,320,219 | 9,969,589 |
| Records per entity | 4.6765 | 5.7543 |
| True matches per entity | 3.4613 | 3.4613 (assumed, checked below) |
| **Unmatched records per entity** | **1.2153** | **2.2931** |
| Unmatched share of records | 26.0% | about 39.8% |

2.2931 / 1.2153 = **1.887**. A decoy false positive happens when an unmatched record is merged into an entity, so on test each entity faces 1.887 times as many chances for one. A false positive from a record that belongs to another entity scales with the number of true records per entity, which does not change, so it keeps weight 1.

**Checking the assumption.** Black-box shift estimation (BBSE) and EM on the stage 1 test scores gave a matched share of 0.6045 (US) and 0.5968 (India), against 0.7386 and 0.7398 on validation. BBSE stayed within 0.6033 to 0.6058 (US) and 0.5950 to 0.5982 (India) across 11 thresholds, which is what a clean prior shift looks like. That implies 3.47 true matches per test entity, the same as train's 3.46. For France the same estimate drifted from 0.630 to 0.581 with the threshold: France has a conditional shift, not just a prior shift, so no reweighting can stand in for French labels.

### Why it mattered

An unweighted metric under-counts decoy false merges, so it rewards trading precision for recall. Two early signs:

- v3 scored 0.9817 locally (older slice, unweighted) and 0.968 on the leaderboard.
- The decoy-word veto was worth +0.0006 on leak-free validation and +0.006 on the leaderboard.

A simulation that duplicated 90% of the unmatched validation records at evaluation time (unmatched share 0.2602 to 0.4005) cost about 0.0019 of stage 1 F0.5 (0.97970 to 0.97778), and re-tuning the threshold won back only +0.00013. Density alone was a real, measurable cost that a threshold could not undo. After the switch to the weighted metric, local and leaderboard gains moved together (section 8).

### What the weight does not fix

- **Singletons.** An entity with no true match scores 0 as soon as it has any false positive, weighted or not. The weight cannot express that such an entity is more likely to attract a decoy on test.
- **Decoy type mix.** In the uncertain score band, test carries 2.2 to 2.6 times the decoy-word hit rate of validation (US 1.74% against 0.68%, India 5.52% against 2.54%). The weight corrects how many decoys there are, not what they look like, so validation still understates changes that remove false positives.

## 4. Out-of-fold predictions

Models that train on the slice are evaluated only on predictions from a model that did not see that part of the slice.

| Model | Folds | Split key | Reason for the key |
| --- | --- | --- | --- |
| Stage 2 | 2 | record id parity | all candidates of a record fall in the same fold, so no record is scored by a model that saw it |
| Stacker | 2 | Source 1 id parity | the metric is per entity; no entity is scored by a model that saw it |
| Rescue acceptance | 2 | Source 1 id parity | same |
| nm1d, hc | 2 | `hash(s1) % 40 < 20` | same halves as the late-change protocol |
| Reverse blocking | 2 | `hash(s1) % 40 < 20` | same |

For test, each model is refitted on the whole slice (stage 2, stacker) or on all eligible pairs (acceptance models) with the same settings.

## 5. Late-change protocol

A threshold picked on the same data it is scored on is optimistic. On this slice that optimism was estimated at about 1e-4 with a paired bootstrap (for example, threshold 0.80 against 0.75: -0.000174, 95% interval -0.000305 to -0.000029, with 893 entities changing). That is the same size as the late gains we were chasing. So for the US and Indian change sets after v17:

1. Fit the acceptance model out of fold across the two halves (`hash(s1) % 40 < 20`).
2. Choose the threshold on the fitting half.
3. Report the gain on the other half, and check the reverse direction too.
4. For combined sets, require every increment to be positive in all four quarters (`hash(s1 * 7 + 3) % 4`).
5. A re-tuned setting has to beat the default by at least +0.00005 held-out, or the default stays.

| Change set | Fitting half | Evaluation half | Reported |
| --- | --- | --- | --- |
| nm1d name-key rescue | +0.000143 | +0.000166 | +0.00017 |
| hc and reverse on top of nm1d | | | +0.000075, positive in 4 of 4 quarters |
| Re-tuned decision settings (expected-F0.5 decoy weight, probability floor, sibling cut-off, rescue threshold) | | | below +0.00005: defaults kept |

## 6. Checking the evaluator itself

Before trusting any local number, the metric code was checked against an independent implementation.

| Check | Result |
| --- | --- |
| Reference evaluator in plain Python (dicts and sets, formula exactly as the challenge README states it) against the pipeline's SQL evaluator | largest difference 4.7e-15 over 10 thresholds from 0.50 to 0.95 (float summation order); stage 1 evaluator 5.2e-15 |
| README worked example (3 predicted, 2 true, both found) | P = 2/3, R = 1, F0.5 = 5/7 = 0.714, reproduced |
| Edge cases | empty prediction for a singleton 1.0; singleton with a false positive 0.0; empty prediction for an entity with matches 0.0; half recall 0.8333 |
| Algebra | the SQL form `1.25tp / (1.25tp + 0.25(n_true - tp) + (n_pred - tp))` reduces to `1.25tp / (0.25 n_true + n_pred)`, identical to the README formula |
| Official validator on a validation-slice output file | PASS: 110,565 rows, 6,609 empty, 368,727 links |
| Ground-truth structure | no Source 2/3 record belongs to more than one Source 1 entity; every ID matches `S[123]-<number>`, so the integer ID encoding round-trips without loss |

The metric was ruled out as a cause of the early leaderboard gap.

## 7. How precise is a local number

Measured on an early leak-free model (stage 2 at threshold 0.75):

| Measure | Value |
| --- | --- |
| Bootstrap standard error over entities (2,000 resamples) | 0.000238 |
| Four quarters of the slice | 0.98299, 0.98256, 0.98303, 0.98262 |
| Two halves | 0.98277, 0.98283 |
| Per-entity F distribution | 86.54% exactly 1, 12.95% partial, 0.51% zero |

Absolute levels wobble by a few 1e-4 between subsets, but paired differences between two systems on the same entities are far tighter (section 5). That is why every decision compared systems on the same slice rather than trusting absolute numbers.

**Leaderboard noise.** Assuming the public leaderboard is a random subset of test entities (the challenge only says "a subset"), the difference between public and private scores has a standard deviation of about 0.000135 at a 30% public share (0.000264 in a worst case), and the paired standard error of a difference between two uploads is about 2e-5 to 5e-5. Every leaderboard step in this project, including the last one (+0.000093), is larger than that paired standard error.

## 8. How local gains translated to the leaderboard

### The gap closed

| File | Local metric | Local | Public leaderboard | Local minus leaderboard |
| --- | --- | --- | --- | --- |
| v1 | older slice, unweighted | 0.9537 | 0.940 | 0.0137 |
| v2 | older slice, unweighted | 0.9714 | 0.950 | 0.0214 |
| v3 | older slice, unweighted | 0.9817 | 0.968 | 0.0137 |
| v6 + veto ("0.974 file") | test-weighted | 0.98241 | 0.974 | 0.0084 |
| v11 | test-weighted | 0.98716 | 0.982 | 0.0052 |
| v15 | test-weighted | 0.98901 | 0.986784 | 0.0022 |

The early validation numbers (v1 to v3) were measured on an older `hash % 50` slice of 44,083 entities with the unweighted metric, so they are not comparable with the later rows. The gap shrank at v11 (cross-encoders help more on the decoy-heavy test) and again at v15 (the France audit rules, which validation cannot see).

### Increments lined up

| Step | What validation predicted | Leaderboard change |
| --- | --- | --- |
| v3 to v6 + veto | veto +0.0006 (unweighted, leak-free stage 2) | +0.006 |
| 0.974 file to v11 | +0.0048 | +0.008 |
| v11 to v15 | +0.00185 for US/India (0.98716 to 0.98901); France rules not measurable | about +0.0048 (0.982 was shown to three decimals) |
| v15 to v17 | 0 for US/India; France modelled +0.0010 central, +0.0005 pessimistic | +0.0003 |
| v17 to v20 | US/India +0.00013 (klarge), +0.0001 (recheck), +0.00017 (nm1d); France push not measurable; pre-upload estimate 0.9877 to 0.9882 | +0.00145 (0.988549) |
| **v20 to v21** | **+0.000075 (hc and reverse) plus under +0.00002 (France fill): +0.00008** | **+0.000093 (0.988642)** |

Three patterns show up:

- **Changes that remove decoy false positives gained more on the leaderboard than locally** (veto, cross-encoder stack), and the uploads that also carried French precision fixes (v15, v20) beat their US/India-only estimate by the widest margin. Validation is decoy-poor even after reweighting (section 3), and it cannot see France at all.
- **Recall changes on US and India landed on their estimate.** The v21 rescue sets were predicted at +0.00008 and measured at +0.000093.
- **Label-free French modelling was optimistic.** The v17 French address fix was modelled at +0.0010 and delivered +0.0003, a third of it. The measured +0.0003 sits between the modelled worst case (+0.00020) and the pessimistic scenario (+0.00054).

## 9. Loss decomposition

### Early snapshot

Leak-free stage 2 at threshold 0.75, before the veto, plain metric. Total loss 0.01720.

| Error type | Entities | Share of loss |
| --- | --- | --- |
| Partial, misses only | 13,004 | 54.3% |
| Entity with matches, nothing predicted | 484 | 25.4% |
| Partial, false positives only | 1,057 | 12.1% |
| Singleton with a false positive | 76 | 4.0% |
| Partial, both misses and false positives | 255 | 3.9% |

Micro precision was 0.99616 and micro recall 0.96021 (1,415 false positives, 15,219 misses). Removing every false positive would have added +0.00324; adding every missed match +0.01400. About 81% of the recoverable loss was recall. Entities with exactly one true match were 5.4% of the slice but 22.7% of the loss (mean F 0.92724): one miss takes such an entity to 0.

### At v11

| Class | Value if fixed alone |
| --- | --- |
| Blocking misses (true pair never a candidate) | 0.0061 |
| True pairs scored below the decision line | 0.0073 |
| False positives | 0.002 |

Each figure is the gain from fixing that one class while keeping the rest, so the classes overlap and their sum is larger than the whole remaining loss (1 - 0.98716 = 0.0128 on the test-weighted metric). Recall still dominated, and every later step aimed at one of these rows:

| Class | What went after it | Measured gain |
| --- | --- | --- |
| Below the line | second e5-small cross-encoder | +0.00061 |
| | e5-base cross-encoder | +0.00054 |
| | e5-large cross-encoder | +0.00013 |
| Blocking misses | key-family rescue | +0.00070 |
| | name-key rescue (nm1d) | +0.00017 |
| | hc and reverse rescue | +0.000075 |
| False positives | high-confidence recheck | +0.0001 |
| | France rules, French-adapted cross-encoder | not measurable locally |

### What was left

In the late analysis, address-less records whose name is shared by several businesses accounted for **about 0.005** of local F0.5, when the whole remaining loss was about 0.011 (v15 and v17: 1 - 0.98901). The rest is made of the typical misses: heavily transliterated names whose legal words stay unrecognized, true copies whose house number was also perturbed ("6825 kildare ave" against "6818 kildare ave"), and the part of the 1.87% blocking misses that rescue cannot reach.

## 10. Why the address-less chain loss is irreducible

The case: a Source 2/3 record with no address, whose name (after normalization) belongs to several Source 1 entities in the same country, such as branches of one business. Four facts close it off.

**These records are real copies.** In training, 97.8% of address-less Source 2 records are true copies of some entity, so leaving them unassigned is a pure recall loss. They are not decoys to be ignored.

**The only field that separates branches is missing.** The Source 1 branches differ in their address. The record has none, and its name is the same for every branch. The team looked for another signal that separates the branches of a chain and found none.

**A guess is not worth making under F0.5.** For an entity that already has m correct matches, adding one more record only pays off if that record is correct with probability at least:

| Correct matches already predicted (m) | 0 | 1 | 2 | 3 | 10 | many |
| --- | --- | --- | --- | --- | --- | --- |
| Break-even probability | 0.500 | 0.727 | 0.759 | 0.771 | 0.791 | 0.800 |

A record split evenly between k branches is right with probability about 1/k. For two branches that is 0.5, which only breaks even for an entity with no other match; for three or more it is below every break-even value. Abstaining is the F0.5-optimal decision, and the models abstain.

**The solvable part was taken.** Where the name is distinctive enough that at most 3 entities carry its key, the nm1d rescue with its competitor-margin feature added +0.00017 held-out ([07-rescue.md](07-rescue.md)). A final search for anything worth +0.0005 or more that could still be built found nothing, and the chain loss was confirmed irreducible with the information in the data.

The same break-even table explains an earlier finding: with about 3.5 true matches per entity, the break-even sits near 0.77, so a single global threshold near 0.75 to 0.8 was already close to optimal for stage 1 scores, and expected-F0.5 decoding added nothing on them (0.97969 against 0.97970). It was adopted later together with the cross-encoder stack in v11, a combined step worth +0.00246 (see [09-experiment-log.md](09-experiment-log.md)).

## 11. Known limitations

- **Transliteration map.** It is learned from all matched training pairs, including those of the slice. It only affects non-Latin names, and its size is small (521 words, each seen at least 20 times with a 60% majority), but its effect on validation was never measured.
- **France cannot be scored.** French changes were judged by hand-read samples of 20 to 30 pairs, label-free fingerprints (for example how often names are lowercased or keep dots) and a Monte Carlo model, and they were kept conservative: guard model, fixed 0.85 threshold, no learned rescue ([06-france.md](06-france.md)).
- **Decoy-poor validation.** The weight corrects decoy density, not decoy style (section 3), so false-positive fixes remain under-rated locally.
- **The slice depends on DuckDB's `hash`.** A different DuckDB version can pick a different slice; the version is pinned.
- **Shared entity features across stage 2 folds.** Stage 2 folds split by record, while entity-level aggregates are computed over all records. The leakage risk through this route is low (at least 200 rows per leaf) but was not measured.

See [07-rescue.md](07-rescue.md) for the rescue passes and [09-experiment-log.md](09-experiment-log.md) for every experiment in order. The full methodology is in [methodology.md](methodology.md).
