# 7. Rescue: recovering pairs that blocking missed

Blocking decides the recall ceiling. A true pair that never becomes a candidate can never be matched, no matter how good the models behind it are. This document covers the four rescue passes that search again, with different keys, for records that end the main pipeline unassigned, and the small models that decide which of those new pairs to accept.

All numbers are on the US and Indian validation slice (test-weighted macro F0.5, see [08-validation.md](08-validation.md)) unless marked as test counts.

## 1. Why a separate rescue pass

| Fact | Value |
| --- | --- |
| True training pairs kept by blocking | 98.13% |
| True entity ranked first by blocking | 97.99% |
| True training pairs that never reach a model | 143,070 of 7,638,365 |
| Weakest cell (Source 3, India) | 97.626% kept |
| Missed true pairs on the validation slice | 3,300 (Source 2) and 3,953 (Source 3) |
| Training records with no candidate at all | 32,571 |
| Value of fixing every blocking miss at v11 | 0.0061 of local F0.5 |

Two observations shaped the design.

**A longer candidate list does not help** (see [02-blocking.md](02-blocking.md) for the pruning study). Almost every true pair that blocking keeps is already ranked first (97.99% against 98.13%). The unpruned top 10 kept only about 0.0004 more of the true training pairs than the pruned lists, at 101,866,356 train candidate pairs instead of 17,769,228. The missing pairs are not sitting just below the cut. Most of them share no rare key with their entity at all.

**The misses have recognisable shapes.** Reading them gave five recurring patterns: domain-style names (`omedicine.com`), words glued together or split apart (`wisedata` against `wise data`), typos in rare name words, a name word dropped from a record that has no address, and invented names that keep part of the real address. Each rescue pass or key family below targets one of these shapes.

## 2. Where rescue sits in the pipeline

| Order | Pass | Records it looks at | Scored by | Accepted when |
| --- | --- | --- | --- | --- |
| 1 | Key-family rescue (v14 onward) | US/India records still unassigned after the decision layer | ce1 + ce2 cross-encoders, 13-feature LightGBM | acceptance probability at least 0.7 |
| 2 | Name-key rescue, nm1d (v20) | US/India records with no address and no blocking candidate | ce1 + ce2, 7-feature LightGBM | acceptance probability at least 0.7 |
| 3 | hc rescue (v21) | US/India records with no address whose blocking candidates all score below 0.3 at stage 1 | same as nm1d | acceptance probability at least 0.7 |
| 4 | Reverse blocking (v21) | Search from the Source 1 side, 20 records per entity | own stage 1, ce2, own stage 2 | score at least 0.6 and best competing entity below 0.2 |

Common rules for every pass:

- It only ever adds a record that is still unassigned at that point. It never moves or removes an existing match.
- Passes 1 to 3 drop pairs that blocking or an earlier pass already proposed, so they only bring genuinely new pairs.
- Each record keeps at most one Source 1 entity, its best accepted candidate.
- France is excluded from the learned passes, because their acceptance models need labels and France has none. French records get their own label-free rescue (section 9 and [06-france.md](06-france.md)).

## 3. Pass 1: six extra key families

`src/rescue_candidates.py` builds an independent candidate search with six key families. Each key is hashed together with the family name and the country, so keys only ever join records of the same country.

| Family | Pattern it targets | Record key | Source 1 key | Selection cap (Source 1 document frequency) |
| --- | --- | --- | --- | --- |
| `aexact` | invented name, real address | all address tokens, sorted and joined (only when there are at least 3 tokens and one contains a digit) | same | 3 |
| `corenum` | spacing and legal-form noise | core name with spaces removed, plus each house number | same | 5 |
| `cat` | glued or split words, domain-style names | sorted letters of the spaceless core name (7+ characters) and of each core word of 8+ characters | sorted letters of every concatenation of 2 or more of the first 7 name words (7+ characters) | 10 |
| `typo` | one-letter typo in a rare word | one exact name word plus the sorted letters (or a one-letter deletion of them) of an out-of-vocabulary word of 4+ characters among the first 5 core words | same, over all words | 10 |
| `anum` | partial address | a house number plus the sorted letters of one alphabetic street word (4+ letters) | same | 1, or 3 when the pair shares 2 or more such keys |
| `aalpha` | partial address without a number | pairs of alphabetic address words among the first 8 | same | 1 |

Because the `typo` key compares sorted letters with one letter deleted on either side, it matches one inserted, deleted, substituted or swapped letter in one word, as long as another word of the name matches exactly.

### Gates and caps

- **Rare keys only.** The Source 1 key index keeps keys carried by at most 100 entities; pairs are formed only through keys carried by at most 10; the final selection applies the per-family caps in the table above.
- **Out-of-vocabulary gate.** Every family except `aexact` only runs for records whose core name contains a word of 3 or more characters that appears in no Source 1 name of the same country. Blocking joins on keys shared with Source 1, so a word that no Source 1 name contains could never have produced a candidate. The gate points the search at exactly those records.
- **Address check for name families.** A `cat`, `typo` or `corenum` pair is kept only if record and entity share an address token (3+ characters, or a number), or if the record has no address.
- **At most 5 candidates per record,** ranked by the number of families that matched, then the strongest family (`aexact` > `corenum` > `cat` > `anum` > `aalpha` > `typo`), then the lowest key frequency, then Source 1 id.

### Illustrative keys

These are made-up records, already normalized, that show how each family fires.

```
aexact   record  "northstar ventures | 12 mg rd indiranagar bangalore"
         entity  "priya textiles     | mg rd 12 indiranagar bangalore"
         key     "12 bangalore indiranagar mg rd"          (same address, any order)

corenum  record  "wisedata labs      | 61 elm st"     key  "wisedatalabs|61"
         entity  "wise data labs llc | 61 elm st"     key  "wisedatalabs|61"

cat      record  "omedicine.com"  ->  "omedicine"          sorted letters "cdeeiimno"
         entity  "o medicine pvt ltd"  words "o" + "medicine"  ->  "cdeeiimno"

typo     record  "kelvn logistics"    anchor "logistics" + "eklnv" (sorted "kelvn")
         entity  "kelvin logistics"   anchor "logistics" + "eklnv" ("eiklnv" minus "i")
```

## 4. The acceptance model

A candidate from a rare key is still often a decoy, so every rescue pair is re-read by the two e5-small cross-encoders (ce1 and ce2, see [04-cross-encoders.md](04-cross-encoders.md)) and then judged by a small LightGBM model (`src/train_rescue.py`).

| Group | Features |
| --- | --- |
| Cross-encoders | ce1 logit, ce2 logit |
| Key families | one flag per family (6), number of families that matched |
| Name | RapidFuzz token-set ratio, ratio of the names with spaces removed |
| Address | token-set ratio (-1 when the record has no address), address-missing flag |

13 features, 15 leaves, learning rate 0.05, at least 40 rows per leaf, 300 rounds.

**Training data.** The same six families are run for the training split, and pairs whose Source 1 entity is in the validation slice are labelled from the ground truth. A pair is eligible when its record is still unassigned on validation (not assigned by the decision layer and with no candidate scoring 0.85 or above after stacking). Records whose true pair was already a blocking candidate but that were never scored are excluded, because they are not blocking misses. That leaves **16,040 eligible pairs, 1,133 of them true**.

**Evaluation.** 2-fold cross-validation split by Source 1 entity, so no entity is scored by a model that saw it. Out-of-fold AUC: **0.9984**. The out-of-fold probabilities are then applied to the validation assignment and scored with the full metric.

### Threshold sweep

Validation, v15 assignment, test-weighted F0.5 before rescue 0.98831.

| Threshold | Records added | Correct | Precision | Test-weighted F0.5 | Gain |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 1,147 | 1,064 | 92.8% | 0.98899 | +0.00068 |
| 0.6 | 1,120 | 1,047 | 93.5% | 0.98899 | +0.00068 |
| **0.7 (used)** | **1,095** | **1,033** | **94.3%** | **0.98901** | **+0.00070** |
| 0.8 | 1,058 | 1,005 | 95.0% | 0.98901 | +0.00070 |
| 0.9 | 992 | 948 | 95.6% | 0.98897 | +0.00066 |
| 0.95 | 939 | 908 | 96.7% | 0.98895 | +0.00064 |

The curve is flat between 0.6 and 0.85 (within 0.00002), so the choice of 0.7 is not fragile. Precision climbs with the threshold, but F0.5 peaks where the extra recall still outweighs the extra false merges. At about 94% precision each added record sits well above the F0.5 break-even of 0.73 to 0.80 for an entity that already has matches (see [08-validation.md](08-validation.md#10-why-the-address-less-chain-loss-is-irreducible)).

### On test

| Step | Count |
| --- | --- |
| Rescue pairs scored (France excluded) | 381,243 |
| Records accepted at 0.7 | 12,451 |
| of which India | 7,410 |
| of which US | 5,041 |

The acceptance model that produced these test additions was fitted on the eligible pairs of the v11 validation assignment; the validation figures above repeat the out-of-fold evaluation on the v15 assignment.

## 5. Pass 2: name keys for address-less records (nm1d)

**The gap.** A record with no address loses every address key and every name-by-address cross key. If its name also drops a word or abbreviates one, blocking has nothing left to join on. These records end up with no candidate at all.

**Population.** US and Indian Source 2/3 records that have no address, no blocking candidate, and no assignment in the build so far.

**Keys.** Built from the Source 1 names: the name words without legal forms and stop words (`pvt`, `ltd`, `llc`, `inc`, `the`, `and`, `m`, `s`, and similar), de-duplicated and sorted.

| Key type | Rule | Example for a made-up entity "Kumar Steel Traders Pvt Ltd" (words: kumar, steel, traders) |
| --- | --- | --- |
| full | all words | `kumar steel traders` |
| one-word deletion | every key with one word removed (names of 2 to 8 words) | `steel traders`, `kumar traders`, `kumar steel` |
| first letter | one word of 3+ characters reduced to its first letter | `steel traders\|k`, `kumar traders\|s`, `kumar steel\|t` |

The same keys are built for the record. A key is used only when **at most 3 Source 1 entities of the record's country** carry it, which keeps generic names ("Steel Traders") out. Pairs already proposed by blocking or by pass 1 are dropped, and each record keeps its **top 5 candidates by Jaro-Winkler** similarity of the full names.

**Acceptance model.** ce1 and ce2 score every pair. A 7-feature LightGBM then decides: both cross-encoder logits, Jaro-Winkler, token-set ratio, the ratio of the names with spaces removed, how many candidates the record has, and the margin between this candidate's averaged cross-encoder score and the record's next best candidate. The margin is there for the hardest case: a record with no address that fits two entities equally well is a coin flip, and the model should refuse it.

**Threshold protocol.** The validation slice is split in two halves by `hash(s1) % 40 < 20`. The model is trained out of fold across the halves; the acceptance threshold is chosen on the fitting half and the gain is reported on the other half. The check is also run the other way round.

| Measure | Value |
| --- | --- |
| Gain on the evaluation half | +0.000166 |
| Gain on the fitting half | +0.000143 |
| Reported held-out gain | +0.00017 |
| Slice additions at 0.7 | 261 |
| of which correct | 244 (93.5%) |
| Test additions | **3,169** |
| Test additions the slice predicts | 3,470 |
| Name-key pairs in the final candidate file | 22,504 |

The test count landing close to what the slice predicts is a useful sanity check that the population on test is the same kind as on validation.

**Variants that did not make the final file.** Several variants were built and validated: deletion keys only with 3 candidates per record (`nm1b`), the full key set with 5 (`nm1c`, `nm1d`) at different thresholds, the same search for records that do have an address (`hca`), and house-number-plus-street-word keys for records with an address but no candidates (`adr`). `nm1d` at 0.7 was the one kept.

## 6. Pass 3: hc, address-less records with only weak candidates

nm1d covers records with no candidate. **hc** covers the neighbouring case: address-less US and Indian records that do have blocking candidates, but whose best stage 1 score is below 0.3. The blocking candidates are probably wrong, and the true entity was never proposed.

hc reuses the nm1d keys, the same key frequency limit of 3, the same top-5 selection, the same cross-encoder scoring and the same kind of 7-feature acceptance model, with its own fit and the same 0.7 threshold.

On test it adds **2,252** records. Its hc candidate pool contributes 24,871 pairs to the final candidate file.

## 7. Pass 4: reverse blocking from the Source 1 side

Forward blocking asks, for each record, "which entities share my rare keys?" and keeps the top 10. Reverse blocking asks, for each Source 1 entity, "which records look most like me?" and keeps 20. A record whose forward list is crowded out by other entities can still be one of the best matches of its own entity.

| Step | Detail |
| --- | --- |
| Candidates | 20 records per Source 1 entity |
| Reverse stage 1 | LightGBM on 21 features: blocking score, shared keys, rank, how many entities share the record's and the entity's core name, how many reverse lists the record appears in, the record's best reverse score, the entity's best and second-best scores, name token-set, token-sort and spaceless ratios, exact core-name equality, address presence, address token-set ratio, shared address tokens, shared house numbers, name lengths, a country flag, and the score relative to the record's best. Keeps the pairs above a cut set at the top 1% of training pairs. |
| Cross-encoder | ce2 scores the kept pairs |
| Reverse stage 2 | LightGBM on 24 features: the ce2 logit, how many unassigned records share the record's or the entity's core name, the record's forward candidate count and best forward score, the reverse score relative to the entity's best, the gap between the entity's best and second-best records, plus most of the stage 1 features |
| Competitor check | the best stage 2 score of any other entity for the same record |
| Accept | stage 2 score at least 0.6 **and** best competing entity below 0.2 |

The competitor condition protects precision: a record is only taken when one entity clearly wants it and no other entity is a plausible owner. On test it accepts **563** records. The reverse search contributes 204,736 pairs to the final candidate file.

## 8. The gate for hc and reverse

hc and reverse were measured on top of nm1d, on the four quarters of the validation slice (`hash(s1 * 7 + 3) % 4`), against the configurations base, nm1d, nm1d + hc, nm1d + reverse and nm1d + hc + reverse.

| Result | Value |
| --- | --- |
| Gain of hc and reverse over nm1d, held-out halves | +0.000075 |
| Quarters where every increment is positive | 4 of 4 |
| Test additions: hc | 2,252 |
| Test additions: reverse | 563 |
| Records found by both | 324 |
| Records added in total | **2,491** |

At the v20 decision they had been held back as below the bar; measured together on top of nm1d they passed, and they went into v21, the final upload. The leaderboard moved by +0.000093 against the +0.00008 the held-out numbers predicted for v21 (see [08-validation.md](08-validation.md)).

## 9. France

The learned passes exclude France because their acceptance models are trained on US and Indian labels. French records get two label-free rescues instead:

- **French exact-address additions** (part of the decision layer): an unassigned French record whose normalized address matches exactly one Source 1 entity is added when its name is that entity's initials or made only of words that never occur in French Source 1 names. These pairs lie outside the blocking set (9,360 in v15; 6,542 in the v17 French re-run).
- **French rescue (France push, v19 onward):** unassigned French records whose name is a run-together, domain-style or hashtag form of a French Source 1 name at the same address with equal house numbers. 2,273 were proposed; the gate removed 17 non-domain names the French-adapted cross-encoder scored below 0 and 8 multi-word names that share no word with the entity beyond legal forms, leaving **2,248**. Hand reads: 20 of 20 random additions correct, and 19 of 20 in a fresh sample from the trimmed set.
- **Empty-entity fill (v21):** 24 strict pairs for French Source 1 entities that would otherwise stay empty.

## 10. How additions are merged

Pass 1 additions are applied by the decision stage. The late change sets are applied by the final chain (`src/final_steps/assemble/apply_changes.py`): the France push removals first, then all addition sets together:

1. Union of the addition sets (French rescue, nm1d, hc and reverse, France fill).
2. A record that appears in more than one set keeps one pair, chosen deterministically (lowest Source 1 id).
3. Any record already assigned in the file is skipped.
4. The script asserts that no record ends up with two Source 1 entities.
5. Every added pair is also written to the candidate file, so the submitted candidate set always contains every match.

## 11. All rescue passes at a glance

| Pass | Version | Population | Validation effect | Test additions |
| --- | --- | --- | --- | --- |
| Key families + acceptance | v14 onward | unassigned US/India records | +0.00070 (0.98831 to 0.98901) | 12,451 |
| nm1d name keys | v20 | address-less, no candidate | +0.00017 held-out | 3,169 |
| hc | v21 | address-less, weak candidates | with reverse: +0.000075 held-out | 2,252 |
| Reverse blocking | v21 | all US/India, from the entity side | (combined with hc) | 563 (324 shared with hc) |
| French rescue | v19 onward | French, label-free | hand reads 20/20, 19/20 | 2,248 |
| French empty-entity fill | v21 | French entities with no match | below +0.00002 expected | 24 |

In the final candidate file, rescue accounts for 445,514 pairs from the key-family pools (including the French rescue pool), 22,504 name-key pairs, 24,871 hc pairs and 204,736 reverse-blocking pairs.

## 12. What rescue cannot fix

- **Record-to-record rescue** of blocking misses was tried before the key families and rejected on validation evidence.
- **Re-tuning the rescue threshold** together with the other decision settings gave no held-out gain of at least +0.00005, so 0.7 stayed.
- **Address-less records with chain names.** When a record has no address and its name is shared by several Source 1 entities (branches of the same business), the name keys either match every branch or are too frequent to be used at all, and nothing in the record separates the branches. The acceptance models correctly refuse these. This is the largest remaining loss, about 0.005 of local F0.5, and [08-validation.md](08-validation.md#10-why-the-address-less-chain-loss-is-irreducible) explains why it is irreducible with the information in the data.
