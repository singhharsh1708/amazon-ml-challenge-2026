# 6. France: matching a country with no labels

The training data covers the US and India. The test set adds France: 259,452 of the 1,732,544 test Source 1 entities (15%) and 1,434,993 of the 9,969,589 test Source 2/3 records. There is not a single French label, so nothing on this page could be tuned on a validation score. French changes were judged on label-free evidence and hand-read samples instead.

This page covers the France-specific rules in the order they were built. The shared decision rules they plug into are in [05-decision-rules.md](05-decision-rules.md), and the full upload history is in [10-leaderboard-journey.md](10-leaderboard-journey.md).

## Contents

- [France at a glance](#france-at-a-glance)
- [Why France needs its own rules](#why-france-needs-its-own-rules)
- [Working principles](#working-principles)
- [1. Two-model guard and a fixed threshold](#1-two-model-guard-and-a-fixed-threshold)
- [2. Label-free fingerprints](#2-label-free-fingerprints)
- [3. The France audit rules](#3-the-france-audit-rules)
- [4. French exact-address additions](#4-french-exact-address-additions)
- [5. The address normalization fix (v17)](#5-the-address-normalization-fix-v17)
- [6. France push: French cross-encoder removals and French rescue](#6-france-push-french-cross-encoder-removals-and-french-rescue)
- [7. Empty-entity fill](#7-empty-entity-fill)
- [8. What each step did on the leaderboard](#8-what-each-step-did-on-the-leaderboard)
- [9. What did not work](#9-what-did-not-work)
- [10. Limitations](#10-limitations)
- [Code map](#code-map)

## France at a glance

| Step | First in | What it does | Size on test |
|---|---|---|---|
| Two-model guard, fixed threshold 0.85 | v10, v11 | keeps a French match only when two stage 2 models agree; learned scores can only lower a French score | applies to every French pair |
| Exact-address additions | v10 | adds unblocked records at a unique address whose name is the entity's initials or made-up words | 9,360 records (v15 build) |
| France audit rules | v13, first uploaded as v15 | removes type-word swaps, adds records that only add trade suffixes, plus smaller removal and addition rules | 14,970 removed, 22,365 added |
| Address normalization fix, re-run and filter | v17 | fixes four French address formats, re-runs France, keeps only well-supported changes | 14,700 added, 3,630 removed |
| France push | v19, uploaded as v20 | a French-adapted cross-encoder removes decoys; a French rescue adds domain-style names | 922 removed, 2,248 added |
| Empty-entity fill | v21 | 24 strict pairs for French entities that would otherwise be empty | 24 added |

## Why France needs its own rules

Country is deliberately not a model feature, so France is scored by models trained only on US and Indian pairs. That is a reasonable start, but French data behaves differently in ways the models cannot see:

| Evidence | US | India | France |
|---|---|---|---|
| Estimated matched share of records, across 11 thresholds | 0.6033 to 0.6058 | 0.5950 to 0.5982 | drifts from 0.6304 to 0.5809 |
| Mid-band test mass against what a US and Indian mixture predicts | consistent | consistent | 1.28 to 1.74 times higher |
| Records in the 0.65 to 0.99 band whose name swaps a real word for another | 3.2% | 2.6% | 20.6% (25,433 records) |
| Learned decoy-word list | 27 words | 65 words | none |

For the US and India the test set looks like the training set with more decoys (a pure prior shift, see [05-decision-rules.md](05-decision-rules.md#6-test-weighted-tuning)). France does not fit that picture: its score distribution changes shape, not just its mix, and name swaps are six to eight times as common as in the other countries. A classifier trained to tell French test pairs from validation pairs reached an AUC of 0.977, against 0.59 for US and 0.57 for Indian test pairs.

French addresses also follow conventions the normalizer was not built for: region and department names, `N°56`, `5B` for "5 bis", and abbreviations such as `Q.` for quai. Section 5 covers the fix.

## Working principles

1. **Model scores never add French matches on their own.** The guard and the cross-encoder stack enter French scores through a minimum, never a maximum. French additions come from explicit rules on names, addresses and scores.
2. **Validate the evidence where labels exist, then read it on France.** Each fingerprint was first measured on the labelled US and Indian slice.
3. **General patterns only.** Rules are written in terms of name words, legal forms, house-number offsets and address formats. No rule lists an individual record or entity.
4. **Read before shipping.** Change sets were sampled and read on the raw test text. Sets that failed were dropped whole or trimmed by a rule, never pair by pair.
5. **No rescue model for France.** The rescue acceptance model has no French labels, so the general rescue pass is switched off for France. French rescue was done later with its own gate (Section 6).

## 1. Two-model guard and a fixed threshold

Some stage 2 improvements that helped the US and India were not trustworthy on France. French pairs therefore need two models to agree:

- the final stage 2 model (55 features, including the odd-one-out features), and
- an older stage 2 model without the odd-one-out features (the v6 model, 27 features).

A French record keeps its best candidate only when both models pick the same Source 1 entity, and its score is the **minimum** of the two. For French pairs in the cross-encoder band the stacked score also enters as a minimum. The final French score is therefore:

```
score = min(stage 2, stacked, guard)   if both models pick the same Source 1 entity
score = 0                               otherwise
```

The expected-F0.5 selection used for the US and India needs calibrated probabilities, which cannot be checked for France. French pairs instead face a fixed, conservative threshold of 0.85 on this score.

One caveat: the guard is the model that was trained with the duplicate-copy augmentation later found to be a leak (see [05-decision-rules.md](05-decision-rules.md#6-test-weighted-tuning)). It can only remove French matches, so it stayed as a second opinion.

## 2. Label-free fingerprints

Without labels we needed properties of a pair that separate decoys from true copies and that can be measured on the labelled US and Indian data first. Four kinds of evidence were used. The first, letter case and dots, is invisible to every model: normalization lowercases the text and strips punctuation before any model reads it.

### 2.1 Lowercase and dots

Some records are written entirely in lowercase. Among pairs whose names differ by one swapped word, wrong pairs on the labelled slice are all-lowercase more than ten times as often as true pairs:

| Address relation (labelled US and Indian slice, best candidates) | Wrong pairs | Lowercase | True pairs | Lowercase |
|---|---|---|---|---|
| same address | 1,476 | 2.5% | 26,953 | 0.2% |
| one house number differs | 10,741 | 2.1% | 2,333 | 0.2% |
| several numbers differ | 4,197 | 2.3% | 2,307 | 0.2% |
| street differs | 1,378 | 2.2% | 6,247 | 0.2% |
| other | 34,861 | 2.4% | 11,974 | 0.1% |
| address missing | 4,417 | 0.5% | 2,798 | 0.1% |

The rate depends heavily on the kind of name difference. True copies with equal names at the same address are lowercase 4.3% of the time, and true run-together names (spaces removed) about 75% of the time. So lowercase is only ever compared within one class of name difference, never across classes.

Dots in the name are the second marker. On the labelled slice they do not separate one-word swaps (about 6% to 8% in both groups), but in France they move together with lowercase, so they were tracked as a supporting signal.

On French test pairs (v11 assignment, pairs at the same address whose names differ by one swapped word), the split is sharp:

| Assigned French pairs, same address, one word swapped | Pairs | Lowercase | Dots |
|---|---|---|---|
| swapped-in word is a learned business-type word | 15,088 | 3.76% | 6.34% |
| swapped-in word is a trade suffix (fils, cie, groupe, ...) | 31,805 | 0.09% | 0.19% |
| swapped word in neither list (the most frequent are abbreviations and typos: svc for service, clbu for club) | 21,851 | 0.11% | 0.15% |
| `compagnie` replacing a word other than `cie` | 332 | 3.92% | 6.02% |
| `compagnie` replacing `cie` | 700 | 0.00% | 0.00% |

Type-word swaps look like the labelled wrong pairs. Trade-suffix swaps and abbreviations look like the labelled true pairs. Section 3 turns that into rules.

### 2.2 Upward house-number offsets

Decoys that change a house number do not change it at random. On the labelled slice, for candidate pairs with equal names on the same street and one differing house number (offset = record's number minus the Source 1 number, 1 to 25 either way):

| Offset | Wrong pairs | True pairs |
|---|---|---|
| up by 1, 2, 3, 4, 5, 7, 9, 11, 13 or 21 | 10,353 | 596 |
| up by any other amount | 47 | 182 |
| down by any amount | 8 | 871 |

Each of the ten decoy offsets holds about 1,000 wrong pairs (1,005 to 1,073), while +6, +8 and +10 hold 12, 5 and 11. Of the 10,408 wrong pairs, only 8 move the number down. True copies move it in both directions with similar frequency, mostly by 1 or 2.

The same fingerprint gives probabilities directly. For equal names, a number moved **up** by a decoy offset is a true match in 4.5% of 13,516 US pairs and 15.2% of 1,264 Indian pairs. Moved **down** by the same amounts, it is a true match in 99.4% of 677 US pairs and 99.5% of 216 Indian pairs.

French candidate pairs (v11 scores) with the same shape show the identical offset set:

| Offset, French candidate pairs | Unassigned | Assigned |
|---|---|---|
| up by one of the ten decoy offsets | 70,836 | 240 |
| up by any other amount | 1,840 | 440 |
| down by any amount | 3,465 | 1,483 |

Each decoy offset holds 6,810 to 7,205 unassigned French pairs, against 90 to 197 for the other upward offsets. The same decoy pattern that shaped the training data is at work in France, and the models were already rejecting most of its output. The assigned pairs lean downward, which is the copy shape.

### 2.3 The symmetry check

Because decoys only move numbers up and copies move them both ways, the direction of house-number changes inside any set of pairs says whether the set is copy-shaped or decoy-shaped. The check needs no labels, and it shaped both additions and removals:

- **Additions** that allow a different house number must be copy-shaped: the France rules only add records whose number moved down, or moved up by an amount that is not a decoy offset.
- **Removals** must not be copy-shaped. A broader removal set of 3,952 pairs from the French cross-encoder (Section 6) failed here. Beyond the 1,001 pairs it shared with the narrow set, 2,571 of its pairs had a different address; of their 1,907 house-number changes, 200 went up by a decoy offset and 1,328 went down. That is the copy pattern, so the set was not used.

### 2.4 Hand reads

Change sets were sampled at random and read on the raw test text, and each pair was sorted into same business, different business or unsure. Twenty to thirty pairs per group catch a bad set. In the France push a fresh sample was read after each trim. Hand reads carried the final decision whenever the fingerprints were ambiguous.

## 3. The France audit rules

The first France rules came from an audit of assigned and guard-blocked French test pairs on the v11 scores. [`src/france_rules.py`](../src/france_rules.py) classifies each pair by how the names differ (equal, legal form only, word order, typo, spacing, acronym, added words, one swapped word, disjoint) and how the addresses differ (same, region only, one number changed and by how much, street changed, missing), and builds two lists from those classes.

### Type-word swaps are decoys

A French decoy often keeps the name and address of a real business and swaps its business-type word. An illustrative (made-up) example:

```
Source 1:  Club Lumiere SARL     12 Rue Victor Hugo, Nantes
record:    Ecole Lumiere SARL    12 Rue Victor Hugo, Nantes     type-word swap, treated as a decoy
record:    Lumiere et Fils SARL  12 Rue Victor Hugo, Nantes     trade suffix in place of the type word, treated as noise
```

The list of type words is learned from French pairs themselves, not written by hand. A word qualifies when it appears as the swapped-in word at least 150 times among best-candidate pairs whose house number also differs, excluding the trade suffixes. On the v11 scores this gives 49 words from 67,087 such pairs. Among assigned one-word swaps at the same address (trade words excluded), the most frequent swapped-in words were club (1,483), ecole (1,144), amicale (1,088), comite (1,038) and amis (690).

The removal rule: an assigned pair at the **same address** whose names differ by **one swapped word**, where the new word is a learned type word, is removed. These pairs carry the wrong-pair fingerprint (3.76% lowercase against 0.09% for trade-suffix swaps, Section 2.1). Hand reads agreed: when the re-run was audited later (Section 5), a random sample of pairs that only v15 contained held 17 "different business" pairs, and 12 of them were type-word swaps at the same address.

Two smaller removal rules follow the same logic:

- **`compagnie` variants.** `compagnie` replacing `cie` is an expansion (0% lowercase over 700 pairs) and is kept. `compagnie` replacing any other word looks like a type-word swap (3.92% lowercase) and is removed.
- **Duplicated "france".** An assigned pair with equal names whose house number moved up by a decoy offset and whose raw record name repeats "france" more often than the Source 1 name does.

This rule is much narrower than a generic word-swap veto. In US and Indian data, most word swaps are true matches (87.5% to 90.8% on the labelled slice), and a blanket swap veto was rejected for both. The French rule only fires on a learned type word, at an identical address, with exactly one word swapped.

### Trade suffixes are noise

Records that add or substitute trade words (fils, cie, groupe, services, developpement, associes, france, frs, freres, st) look like copies: the trade-suffix swaps sit at 0.09% lowercase, at or below the labelled true-pair rate. The rule adds such a record when it is unassigned, both models point at the same entity, the address is the same, every added word is a trade word, and any word it replaced is common in French Source 1 names (at least 1,000 names contain it).

This reverses an earlier hypothesis. Counting how often words are added versus dropped on test flagged `developpement` (added 4,373 times, dropped 65) and `groupe` (4,610 against 601) as possible French decoy words. The fingerprints and hand reads pointed the other way, so they became noise words instead.

### Other additions

| Rule | Condition | Why it is safe |
|---|---|---|
| Guard-blocked, same business | both models pick the entity, stacked score at least 0.85, guard below 0.85, names equal up to legal form, word order, typo or spacing, same address | the guard was the only objection |
| Acronym or unique address | the address belongs to only one French Source 1 entity, and the record's name is its initials or made of words no French Source 1 name uses; score at least 0.1 | a scrambled or abbreviated name at a one-business address |
| House-number change, copy-shaped | names equal up to order, typo or spacing, the number moved down or up by a non-decoy amount, no other Source 1 entity with the same name at another number on that street; score at least 0.5 | the symmetry check (Section 2.3) |

### Counts

| List | Rule | Pairs (v11 inputs) |
|---|---|---|
| Removals | type-word swaps | 15,088 |
| | `compagnie` variants | 332 |
| | duplicated "france" | 59 |
| | **total** | **15,479** |
| Additions | trade-suffix noise | 14,696 |
| | acronym or unique address | 5,883 |
| | guard-blocked, same business, stacked score at least 0.85 | 2,207 |
| | copy-shaped house-number changes | 1,142 |
| | **total** | **23,928** |

Applied to the v15 assignment, the lists removed 14,970 existing matches and added 22,365 pairs; the other 1,563 proposed pairs were already assigned. The released `france_rules.py` recomputes both lists from the current scores. On the v15 scores it proposes 16,711 removals and 23,399 additions, so a rebuild with the released code differs from the v15 file for 4,049 French records.

These rules were first uploaded in v15, together with a second e5-small cross-encoder, the Kaggle e5-base cross-encoder and the rescue pass. Local validation (US and India only) rose from 0.98716 to 0.98901, and the leaderboard from 0.982 to 0.986784. The local-to-leaderboard gap shrank from about 0.0052 to 0.0022. Local validation cannot see France, so the France rules are the most likely source of that extra gain, though the new cross-encoders may also have helped more on the decoy-heavy test than locally, as they did at v11.

## 4. French exact-address additions

Some French records carry a scrambled or invented name at a real address. Blocking leans on name keys, so it often misses them. [`src/address_rules.py`](../src/address_rules.py) builds an address key for every French record (region words removed, tokens sorted, must contain a number) and adds an unassigned record to a Source 1 entity when:

- its key equals the key of **exactly one** French Source 1 entity, and
- its name is either that entity's initials, or made only of words that never occur in any French Source 1 name.

These pairs lie outside the blocking set. The rule added 9,360 test records in the v15 build and 6,542 in the v17 re-run.

## 5. The address normalization fix (v17)

### What was wrong

The address normalizer was written for US and Indian addresses. On French addresses it produced tokens that hid true matches:

- Source 1 addresses end with a region ("Pays de la Loire"). Records carry the region, the department instead ("Loire-Atlantique"), or nothing. Every variant added mismatching tokens.
- Transliteration turned `N°56` into the single token `ndeg56`, and the house number was lost.
- `5B` and `19BIS` stayed glued, so they never matched `5 bis` and `19 bis`.
- French street abbreviations (`Q.`, `Crs`, `Bld`) were not mapped to one form.

### The four fixes

`normalize_address_france` in [`src/normalize.py`](../src/normalize.py) runs only for French records:

| Fix | Rule | Example |
|---|---|---|
| Region names | drop a comma-separated part that is only a region, a department or "France" (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire, Nord, Pas-de-Calais, Gironde, Loire-Atlantique, France) | `..., Nantes, Pays de la Loire` to `... nantes` |
| Number sign | remove `N°` and `Nº` before a number | `N°56` to `56` |
| Number suffixes | split a suffix glued to the number; `b` becomes bis and `t` becomes ter | `19BIS` to `19 bis`, `5B` to `5 bis`, `12T` to `12 ter` |
| Street types | map French abbreviations to one form | `q` to quai, `crs` to cours, `psg` and `pass` to passage, `res` to residence, `bld`, `bvd` and `boul` to blvd, `appt`, `app` and `appartement` to apt |

### Before and after

Normalized text for three record and Source 1 pairs (the "after" column is what `normalize_address` in the current code returns):

| Raw address | Before the fix | After the fix |
|---|---|---|
| record: N°56 AVENUE DE VILLENEUVE, SAINT-NAZAIRE, Pays de la Loire | `ndeg56 ave de villeneuve st nazaire pays de la loire` | `56 ave de villeneuve st nazaire` |
| Source 1: 56 Avenue de Villeneuve, Saint-Nazaire, Pays de la Loire | `56 ave de villeneuve st nazaire pays de la loire` | `56 ave de villeneuve st nazaire` |
| record: 5B R. Maurice Duval, Pays de la Loire, Nantes | `5b r maurice duval pays de la loire nantes` | `5 bis r maurice duval nantes` |
| Source 1: 5 Bis Rue Maurice Duval, Nantes, Pays de la Loire | `5 bis r maurice duval nantes pays de la loire` | `5 bis r maurice duval nantes` |
| record: 58 Q. LERAY, PORNIC, Loire-Atlantique | `58 q leray pornic loire atlantique` | `58 quai leray pornic` |
| Source 1: 58 Quai Leray, Pornic, Pays de la Loire | `58 quai leray pornic pays de la loire` | `58 quai leray pornic` |

In each pair the two addresses were different strings before the fix and are identical after it.

| Scope of the change | Count |
|---|---|
| French test addresses changed | 1,233,001 of 1,694,445 |
| of which Source 1 | 259,452 (all of them) |
| of which Source 2 | 471,692 |
| of which Source 3 | 501,857 |
| US and Indian test records whose normalized output changed | 0 of 10,007,688 |

The training data has no French records, so no model and no validation number changes.

### The French re-run

With the v15 models unchanged, the French records went through the pipeline again on the fixed normalization:

| Stage | Result |
|---|---|
| Blocking | 3,700,328 candidate pairs for 1,431,134 French records (before: 3,929,813 for 1,431,690) |
| Recall of v15's French matches | 838,880 of 845,271 are among the new candidates (835,911 among the old ones) |
| Stage 1, stage 2, guard | all new pairs rescored |
| Cross-encoders | not re-run; 147,634 of the 205,609 French band pairs had v15 cross-encoder scores and were re-stacked with the v15 stacker; the rest kept their stage 2 score |
| Decision | 862,476 best candidates at 0.85 or above, 733 dropped by the sibling rule, 6,542 exact-address additions |
| France rules, recomputed on the new scores | 24,476 removed (23,901 type-word swaps, 534 `compagnie` variants, 41 duplicates), 12,217 added |
| Result | 856,026 French pairs (model 837,267, France rules 12,217, exact address 6,542) |

Fewer candidate pairs, yet more of v15's matches among them. Most likely the dropped region words removed common, uninformative keys, while visible house numbers gave true copies rarer keys to share.

### Auditing the changes

Against v15: 840,214 pairs in both runs, 15,812 only in the new run, 5,057 only in v15. We read 30 random pairs from each group on the raw test text:

| Group | Same business | Different | Unsure |
|---|---|---|---|
| Only in the new run (15,812) | 25 | 3 | 2 |
| Only in v15 (5,057) | 11 | 17 | 2 |

Most of the new same-business pairs involve exactly what the fixes address (`N°`, `No` and `#` prefixes, `3B` for 3 bis, `53BIS` for 53 bis) or a trade suffix such as Et Fils, Cie or Groupe in place of the type word. 12 of the 17 "different" v15-only pairs are type-word swaps at the same address.

Why the v15-only pairs dropped out of the new run:

| Reason | Pairs |
|---|---|
| removed by the recomputed France rules (11 of 11 sampled were different businesses) | 1,645 |
| new score below 0.85 | 2,058 |
| blocked by the guard | 973 |
| no longer a candidate | 313 |
| other | 68 |

Running the rule classifier over all changed pairs isolated two weak groups:

- new pairs whose house number or street differs from the Source 1 address: 12 sampled, 9 different, 1 same, 2 unsure;
- v15 pairs lost to the new run's score, guard or blocking although the names agree: 12 sampled, all 12 the same business.

### The filter against v15

`python src/france_rules.py partial` combines the two runs using only those classes:

1. **Drop** the 1,112 new pairs (not in v15) whose house number or street differs between record and Source 1.
2. **Put back** the v15 pairs that the new run lost through its score, guard or blocking (not through the France rules), when the names are equal, a typo, spaceless, an acronym or differ only by a trade-suffix word, and the address is the same or missing: 1,449 pairs, of which 1,427 are put back and 22 skipped because the record is already assigned elsewhere.

The result has 856,341 French pairs, one Source 1 entity per record. Against v15: 841,641 in both, 14,700 added, 3,630 removed, and 16,869 French Source 1 rows changed. v17 is v15 with only these French rows replaced; no US or Indian row changed.

### Expected and measured effect

France cannot be scored locally, so the effect was modelled with a Monte Carlo over the 19,058 affected French entities (200 draws, per-entity F0.5 averaged over all 1,732,544 entities). Pairs in both files count as true. Each changed pair is true with the probability of its audited group:

| Group | Central | Pessimistic |
|---|---|---|
| new pairs kept | 0.929 | 0.80 |
| new pairs with a different address | 0.143 | 0.35 |
| removed by the France rules | 0.042 | 0.20 |
| put back by the filter | 0.976 | 0.85 |
| other v15 drops | 0.35 | 0.55 |

A worst case additionally treats every trade-suffix swap as a decoy (0.10).

| Scenario | v17 (filtered) | New run without the filter |
|---|---|---|
| Central | +0.00106 | +0.00084 |
| Pessimistic | +0.00054 | +0.00040 |
| Worst case | +0.00020 | +0.00004 |

The filter beats the unfiltered run in every scenario, so v17 shipped with it. On the leaderboard v17 moved from 0.986784 to 0.9871, about +0.0003. Because v17 changed only French rows, this is the one clean leaderboard measurement of a French change. It landed between the pessimistic and worst-case scenarios, about a third of the central estimate. We do not know which assumption was optimistic; hand-read samples of 30 pairs per group leave wide margins.

### A side finding: blocking was not repeatable

Re-running the old blocking on unchanged French data returned the same number of pairs (3,929,813), but 38,703 records got a different candidate set. 99,264 of the 99,276 pairs found only in the re-run tied with the record's lowest kept score to within 1e-9. DuckDB adds the IDF doubles in a different order on each parallel run, so equal scores differed in the last bits and the tie-break by Source 1 id never applied. `src/block_candidates.py` now rounds each IDF times 10^9 to an integer and sums integers. On 71,532 French records blocked three times, the old query gave 3,140 and 1,983 differing pairs between runs and the new one gave 0. The v17 French candidates were produced with the old query.

## 6. France push: French cross-encoder removals and French rescue

The last French change set had two halves, built and gated separately.

### Removals from a French-adapted cross-encoder

ce1 (the first e5-small cross-encoder, see [04-cross-encoders.md](04-cross-encoders.md)) was trained further on 100,000 French pairs from the re-run. The labels come from our own rules, never from ground truth or outside data:

- **positives:** French pairs with stage 2 and guard scores of at least 0.98 that the name-and-address review rule calls the same business;
- **negatives:** pairs scored at most 0.02, plus synthetic decoys made from Source 1 records by the patterns above (a business-type word swapped, the house number moved by a decoy offset, the legal form changed, a word added).

Training took 1,631 s on the Apple M2 GPU (62 pairs/s). The model flagged 1,001 assigned French pairs as type-word-swap decoys, all with a cross-encoder logit of at most -4.02.

The gate:

| Check | Result |
|---|---|
| Trim | 79 pairs removed from the set: swapped words within edit distance 2, or abbreviations (Etablissements and Ets, Saint and St), which read as typo copies. **922 remain.** |
| Fingerprint | the 922 are lowercase in 4.45% and dotted in 4.34% of record names, against 0.02% and 0.05% for the 23,969 other type-word pairs that stay in the file |
| Symmetry | among the 1,001 flagged pairs, house numbers differ in only 8 of the 932 where both sides have one, and none moves up by a decoy offset; these are same-address swaps |
| Hand reads | 20 of 20 random removals were decoys; a fresh 20 from the trimmed set gave 18 decoys, 1 typo copy, 1 unsure |
| Hand-labelled pairs | of 60 hand-labelled French pairs, the removals take out the one labelled different and none labelled same |

The fingerprint is not fully independent of the model. The cross-encoder reads lowercased text and cannot see case, but the business-type words used to build its synthetic decoys were chosen with a word-level lowercase filter (at least 2%, in [`frce/b1_build.py`](../src/final_steps/france_push/frce/b1_build.py)). For this set the hand reads and the symmetry check are the stronger evidence.

A broader removal set of 3,952 pairs from the same model failed the symmetry check (Section 2.3): its extra pairs were copy-shaped. It was not used.

### French rescue

Some unassigned French records are a run-together, domain-style or hashtag form of a French Source 1 name at the same address, for example (illustrative) `lumiereclub.fr` or `#ClubLumiere` for "Club Lumiere SARL". The general rescue pass ([07-rescue.md](07-rescue.md)) is off for France, so these were proposed by a French-only search that required equal house numbers: 2,273 pairs.

| Check | Result |
|---|---|
| Trim | 17 non-domain names that the French cross-encoder scores below 0 and 8 multi-word names that share no word with the Source 1 name besides legal forms were removed. **2,248 remain.** |
| Hand reads | 20 of 20 random additions correct; a fresh 20 from the trimmed set, 19 of 20 |

Together the two halves touch 3,115 Source 1 entities (0.18%) and change no US or Indian row.

## 7. Empty-entity fill

In v17, 15,580 of the 259,452 French Source 1 entities (6.0%) had no match. Many of those are probably real singletons: 5.6% of training entities have no true match. So the fill had to be strict.

Candidates were unassigned French records whose best candidate is an empty entity with a stage 2 score of at least 0.8, that the guard also picks, and that the French decision missed only because the lower of the two scores fell below 0.85. The gate read all 31 proposals and 30 pairs from a wider pool, and dropped every pair with a conflict: another Source 1 entity with the same name at the same address, or a reading as a type-word swap. **24 pairs** remained, all scoring at least 1.03 with the French cross-encoder, none looking like a decoy. They affect 20 entities, so the expected effect is below +0.00002. A wider fill of 62 pairs failed its audit (conflicting Source 1 entities at the same address).

## 8. What each step did on the leaderboard

| Upload | Score seen (27 Sep, IST) | French content | Public F0.5 | Change |
|---|---|---|---|---|
| v6 + veto | | France scored like the US and India at threshold 0.75, no guard | 0.974 | |
| v11 | about 10:00 | two-model guard, stacked scores can only lower French scores, exact-address additions | 0.982 | +0.008, whole file |
| v15 | 16:35 | France audit rules | 0.986784 | +0.0048, whole file |
| v17 | about 18:15 | address fix, re-run and filter; French rows only | 0.9871 | about +0.0003, France only |
| v20 | about 21:05 | France push (922 removals, 2,248 additions), bundled with US and Indian changes | 0.988549 | +0.00145, whole bundle |
| v21 (final) | 23:36 | 24 empty-entity fills, bundled with US and Indian rescue | 0.988642 | +0.000093, whole file |

Only v17 isolates France. The v20 bundle also added e5-large to the stack, the high-confidence recheck and the name-key rescue, and was not split on the leaderboard, so the France push's share of its +0.00145 is unknown. It can be bounded: the push touches 3,115 of 1,732,544 entities, so even if every one of them went from 0 to 1 the macro score could move by at most 0.0018.

## 9. What did not work

| Idea | Why it was dropped |
|---|---|
| A blanket veto on French word swaps | as a group, the swaps behaved like true matches; the narrower type-word rule came later from the audit |
| Treating `groupe` and `developpement` as French decoy words | suggested by add and drop counts; the fingerprints and hand reads said they are noise |
| Per-country score normalization, self-training | no gain |
| The 3,952-pair removal set from the French cross-encoder | copy-shaped house-number changes among its extra pairs (1,328 down, 200 up by a decoy offset) |
| A 62-pair empty-entity fill | conflicting Source 1 entities at the same address |
| The general rescue model on France | no French labels to fit or check its acceptance threshold |

## 10. Limitations

- **Precision rests on reading.** Twenty to thirty hand reads per set catch a bad set; they cannot certify a good one to high precision. The v17 result (about a third of the central estimate) is a reminder.
- **Fingerprints separate groups, they do not label pairs.** They are class-dependent (Section 2.1) and correlational. They were used to accept or reject whole rule-defined sets, never to pick individual pairs.
- **The French cross-encoder learns from our own rules.** Its labels come from high- and low-confidence pairs and synthetic decoys, so it can inherit the rules' blind spots, and its type-word list was picked with a lowercase filter. The trims, the symmetry check and the hand reads were the check on that.
- **The guard carries the old leak.** It can only remove French matches, but it was trained with synthetic exact copies.
- **Part of v17 is not bit-for-bit repeatable.** Its French candidates came from the old blocking query, and the re-run reused the v15 cross-encoder scores (see [reproduction.md](reproduction.md), step 12).

## Code map

| File | What it does |
|---|---|
| [`src/normalize.py`](../src/normalize.py) | `normalize_address_france`: the four French address fixes |
| [`src/predict_submission.py`](../src/predict_submission.py) | guard, France minimum with the stacked score, fixed 0.85 threshold, applies the France rules |
| [`src/france_rules.py`](../src/france_rules.py) | name and address classes, proposals, removals, and the `partial` filter against an earlier file |
| [`src/address_rules.py`](../src/address_rules.py) | French exact-address additions |
| [`src/block_candidates.py`](../src/block_candidates.py) | blocking with integer IDF sums |
| [`src/final_steps/assemble/france_audit/`](../src/final_steps/assemble/france_audit/) | the audit scripts behind Sections 2 and 3 (`j12_fp_trainclass.py` for the labelled lowercase rates, `j15_typeswap_asg.py` and `j25_removal.py` for the French fingerprints, `j21_delta.py` for the house-number offsets) |
| [`src/final_steps/france_rerun/`](../src/final_steps/france_rerun/) | the v17 France re-run |
| [`src/final_steps/france_push/`](../src/final_steps/france_push/) | French cross-encoder (`frce/`), French rescue (`frrescue/`) and their gate (`gate/`, fingerprints in `g2_check.py`) |
| [`src/final_steps/france_fill/`](../src/final_steps/france_fill/) | empty-entity fill and its gate |
