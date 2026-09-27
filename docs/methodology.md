# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Inno8  
**Team Members:** Drashtant Mevada (Team Leader), Jenil Gajera, Ramani Dwarkesh, Singh Harsh Rahulkumar  
**Submission Date:** 2026-09-27

---

## 1. Summary

This section is the short, self-contained description of the solution (approach, models, experiments, conclusion). Sections 2 to 8 give the details and the evidence behind each number.

**Task.** For each of the 1,732,544 test Source 1 businesses (US, India, France) list its matching Source 2 and Source 3 records. The score is macro F0.5 per Source 1 entity, so a wrong merge costs more than a missed one.

**Approach.** In the training data every Source 2/3 record belongs to at most one Source 1 entity, so we resolve from the Source 2/3 side: each record picks its single best Source 1 candidate or stays unmatched, and a Source 1 entity's matches are the records assigned to it. The pipeline:

1. **Normalization.** Non-Latin scripts are transliterated and mapped back to English words with a dictionary learned from the training pairs; legal forms, DBA prefixes, leetspeak, street types, ordinals and state names are canonicalized; French addresses get four extra fixes (region names, `N°`, `5B` to `5 bis`, street abbreviations).
2. **Blocking.** IDF-weighted rare keys (name and address tokens and bigrams, spaceless name, name x address cross keys), at most 10 Source 1 candidates per record. It keeps the true entity for 98.13% of training ground-truth pairs.
3. **Two LightGBM stages.** Stage 1 scores 60 pair features (RapidFuzz similarities, house numbers, legal forms, blocking score). Stage 2 adds record and entity context and 28 "odd one out" features that flag the record whose house number or extra word differs from the other records pointing at the same entity. For France, which has no labels, an older stage 2 model acts as a guard.
4. **Cross-encoders.** Four fine-tuned multilingual transformers (two e5-small, e5-base, e5-large) re-read the 902,470 uncertain test pairs, and a LightGBM stacker combines them with the stage 1 and stage 2 scores. The two e5-small models also recheck the 429,125 very confident US and Indian pairs (0.99 < p <= 0.999) and can only lower their score.
5. **Decision.** Best candidate per record, learned decoy-word veto, a per-entity choice of the records that maximises expected F0.5 (US, India), a fixed 0.85 threshold for France, and a house-number sibling rule.
6. **Targeted corrections.** Audited France rules; a French-adapted cross-encoder that removes 922 type-word-swap decoys; rescue passes for records that blocking missed (extra key families, a name-key search, reverse blocking from the Source 1 side), each accepted by a small model; 2,248 French rescue additions and 24 French pairs for otherwise empty Source 1 entities. Every late change was kept only after a gate: a held-out gain on the validation slice for US and India, and hand-read samples plus label-free fingerprints for France.

**Models.** LightGBM (stage 1, stage 2, France guard, stacker, acceptance models; MIT) and fine-tuned multilingual text encoders: `intfloat/multilingual-e5-small` (118M), `-base` (278M) and `-large` (560M), all MIT; if used in the final file, `BAAI/bge-reranker-v2-m3` (568M, Apache 2.0) as a fifth cross-encoder. Compute: an Apple M2 laptop (16 GB) plus free Kaggle T4 and P100 notebooks. No external data, APIs or lookups.

**Validation.** 110,565 held-out US and Indian Source 1 entities (`hash(id) % 20 = 0`), macro F0.5 with false positives from records that have no true entity weighted 1.887, because test has 1.887 times as many such records per entity as train. With this weighting, local gains and leaderboard gains moved together.

**Experiments and results.**

| File | Main change | Local validation (test-weighted F0.5) | Public leaderboard |
| --- | --- | --- | --- |
| v1 | full-scale blocking and LightGBM | older metric | 0.940 |
| v2 | cross and spaceless keys, stage 2 | older metric | 0.950 |
| v3 | transliteration map, legal-form and house-number features | older metric | 0.968 |
| v6 + veto | learned decoy-word veto | 0.98241 | 0.974 |
| v11 | e5-small cross-encoder stack, expected-F0.5 selection | 0.98716 | 0.982 |
| v15 | second e5-small, e5-base (Kaggle), France rules, rescue | 0.98901 | 0.986784 |
| v17 | French address fix and French re-run (France only) | 0.98901 | 0.9871 |
| v20 | e5-large in the stack (+0.00013 over three cross-encoders), high-confidence recheck (+0.0001), France push, name-key rescue (+0.00017 held-out) | increments shown | 0.988549 |
| **v21 (final)** | hc and reverse rescue (+0.000075 held-out), 24 French empty-entity fills | increments shown | 0.988642 (v21, the last upload; v20 scored 0.988549) |

If the last upload is v22, it is the v21 chain with `bge-reranker-v2-m3` added to the stacker as a fifth cross-encoder; everything else is identical.

What did not work: synthetic exact-duplicate decoys in training (a leak: +0.003 locally, -0.0017 once measured cleanly), per-country score normalization, self-training, a 95-feature stacker (+0.00004), a third e5-small cross-encoder, re-tuning the decision settings (no held-out gain), and broader French change sets that failed their audit.

**Conclusion.** A one-owner assignment from the Source 2/3 side, heavy investment in blocking recall, and a validation metric reweighted to the test set's decoy density gave steady, leaderboard-confirmed progress from 0.940 to 0.988549. Small multilingual cross-encoders applied only to the uncertain band were the largest late gain. The remaining loss is mostly recall on records that carry no address and a generic name, and precision on French records, where no labels exist.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the provided files:

- **Scale.** Train: 2,206,821 Source 1, 5,034,616 Source 2 and 5,285,603 Source 3 records. Test: 1,732,544, 4,887,273 and 5,082,316. Exhaustive pairwise comparison is impossible, so blocking decides both runtime and the recall ceiling.
- **Match structure.** The ground truth holds 7,638,365 matched pairs. No Source 2/3 record is matched to more than one Source 1 entity. About 27% of Source 2 and 25% of Source 3 records match nothing. 123,247 Source 1 entities (5.6%) are singletons; the rest have 1 to 11 matches, 3.67 on average (3.46 per entity over all entities).
- **Countries.** Train has US and India only. Test adds France: 259,452 of the 1,732,544 test Source 1 records (India 809,986, US 663,106) and 1,434,993 of the 9,969,589 test Source 2/3 records. France has no labels, so nothing in the learned models uses country as a feature.
- **Test is more decoy-heavy than train.** Train has 4.6765 Source 2/3 records per Source 1 entity, test 5.7543. If test has the same 3.4613 true matches per entity as train, unmatched records per entity rise from 1.2153 to 2.2931, a factor of **1.887**, and about 39.8% of test Source 2/3 records match nothing (26.0% in train). This single number drives our validation metric (Section 5).
- **Source 1 is clean, Sources 2/3 are noisy.** Source 1 names are plain ASCII. In Sources 2/3, 28% (Source 2) and 19% (Source 3) of Indian business names are written in Devanagari, Bengali, Tamil, Telugu or Gujarati, and about 7% of US names carry diacritics or character substitutions. About 3% of Source 2/3 addresses are missing (`None`, `NULL` or empty).
- **Name noise.** Legal suffix swaps (Pvt/Private, Ltd/Limited, LLC, Inc, SARL, SAS), filler words (Services, Center, Group), prefixes (M/s, Shri, `***`), DBA prefixes with invented names ("Kelovantagehalo Co DBA: Heritage Midstream Inc."), word reordering, typos, leetspeak ("ster1ing") and domain-style names ("omedicine.com").
- **Address noise.** Street and unit abbreviations (Rd/Road, St/Street, Ngr/Nagar), house-number formats (`#61`, `H.NO 61`, `Door No 61`), leading zeros (`00691`), ordinal words ("Second Street"), state names vs codes vs native script (Maharashtra, MH, महाराष्ट्र), missing, reordered and injected components.
- **Decoys.** Many unmatched Source 2/3 records copy a real business and change one detail: the house number by a small offset (1788 vs 1781), the legal form ("Soutien Amicale SNC" next to the real "Soutien Amicale SAS"), or one added word ("... Enterprises", "... Holdings"). These are the main source of false merges, and they are the "odd one out" among the records that point at the same Source 1 entity.

### 2.2 Solution Strategy

**Approach Type:** Hybrid: blocking, two-stage gradient-boosted pair classifier, cross-encoder re-scoring of the uncertain band, stacking, and a one-owner assignment with expected-F0.5 selection.  
**Core Innovation:**

1. Resolving from the Source 2/3 side, which enforces the one-owner property for free and removes pairwise clustering.
2. Name-by-address cross keys and a transliteration dictionary learned from the training pairs, which make blocking recall 98.13%.
3. Context the pair cannot see on its own: a second model sees how many other records already point at the same Source 1 entity and whether this record is the odd one out among them (different house number, unexplained extra word, new suffix).
4. Cross-encoders used only where they matter: the 902,470 test pairs in the uncertain band, not all 19.2M candidates.
5. A per-entity expected-F0.5 decision instead of one global threshold.
6. A validation metric reweighted to the test set's decoy density, which is what made local gains line up with leaderboard gains.

Pipeline (each step is one script in `src/`, see Appendix A):

1. Normalize all six source files.
2. Blocking: up to 10 Source 1 candidates per Source 2/3 record.
3. Pair features and stage 1 LightGBM.
4. Odd-one-out features and stage 2 LightGBM; for France a second, older stage 2 model acts as a guard.
5. Cross-encoders (ce1, ce2, kbase, klarge) on the uncertain band, combined by a LightGBM stacker.
6. High-confidence recheck: US and Indian top-1 pairs with a stage 2 score in (0.99, 0.999] are rescored by ce1 and ce2 and a stacker, and keep the lower of the two scores.
7. Decision: best Source 1 per record, decoy-word veto, expected-F0.5 selection per Source 1 (US, India), threshold 0.85 (France), house-number sibling rule, French exact-address additions.
8. France audit rules (removals and additions).
9. Rescue: extra key families for records that blocking missed, scored by the cross-encoders and a small acceptance model.
10. French rows from the v17 re-run: French address normalization fix, French records re-run through steps 2 to 8 with the v15 models, and a filter against v15.
11. France push: a French-adapted cross-encoder removes type-word-swap decoys (922 pairs) and a French rescue adds domain-style and run-together names at the same address (2,248 pairs).
12. Name-key, hc and reverse rescue for US and Indian records that are still unassigned (3,169 + 2,491 pairs).
13. France empty-entity fill: 24 strict pairs for French Source 1 entities that would otherwise stay empty.

Steps 1 to 10 are the scripts in `src/`; steps 6 and 11 to 13 are in `src/final_steps/` (Section 4.9).

---

## 3. Candidate Generation (Blocking)

**Normalization** (shared by every stage): non-Latin scripts are transliterated to ASCII with `anyascii` (ISC licence). Transliterated words are then mapped back to English with a dictionary learned from the training data: in 954,525 matched pairs where the Source 2/3 name was non-Latin and had as many words as the Source 1 name, words were aligned by position, and every transliterated word whose most frequent aligned English word covered at least 60% of at least 20 occurrences was kept (521 entries, for example `praivet` to private, `teknolojij` to technologies, `tredimg` to trading). For transliterated Indian names this raises exact agreement of the core name with the true Source 1 name from 15.3% to 89.5%. Text is lowercased, `&` becomes "and", DBA prefixes are cut to the real name, domain suffixes are stripped, leetspeak digits inside words are mapped back to letters, and consecutive duplicate words are removed. A "core" name drops legal forms and filler words in English, French and their common transliterations. Addresses drop placeholder and house-number marker tokens, strip leading zeros, map ordinal words and street types to one short form, and map US and Indian state names (including transliterated native-script names) to their codes. In the final file, French addresses get four more fixes (region names, the `N°` sign, number suffixes such as `5B` and `19BIS`, French street-type abbreviations; Section 4.8).

- **Blocking keys used** (all scoped to the record's country):
  - name core tokens and address tokens (unigrams);
  - adjacent name token pairs and adjacent address token pairs, order-insensitive;
  - the core name with spaces removed ("wisedata" vs "wise data");
  - name token x address token cross keys, for example `primary|chelsea`.
- **Scoring.** A candidate's blocking score is the sum of the IDF (computed on the split's own Source 1 file) of all keys it shares with the record. Unigrams are used as join keys only when at most 50 Source 1 records carry them, all other keys when at most 500 do. Each record keeps at most 10 candidates, and only those scoring at least half of its best candidate's score.
- **Engineering.** Entity IDs and keys are hashed to integers and the join runs in DuckDB over 40 slices of the Source 2/3 records with capped memory and spill, so the whole train or test set blocks on a 16 GB laptop.
- **Candidate pairs generated:** 17,769,228 for train (10,287,648 records) and 19,225,118 for test (9,947,553 records). The v15 `candidate_pairs.tsv` holds 19,246,911 pairs: the blocking set plus the pairs added by the French exact-address rule, the France audit rules and the rescue pass. The v17 candidate file holds 19,250,779 pairs: the v15 file plus the 3,868 French pairs that v17 matches and the v15 file did not contain. Every submitted match is also in the candidate file.
- **Final candidate file (v21 `candidate_pairs.tsv`):** 19,258,685 pairs over all 1,732,544 Source 1 rows: the v17 file plus the pairs added by the late rescue and France steps (Section 4.9) that it did not already contain. We checked that all 5,833,349 submitted matches are in it (0 missing).

| Candidates per Source 1 entity (v21) | Value |
| --- | --- |
| Mean | 11.12 |
| Median | 9 |
| 90th / 99th percentile | 18 / 36 |
| Maximum | 4,433 |
| Source 1 entities with no candidate | 45 |
| Entities with 1 to 5 candidates | 217,764 |
| Entities with 6 to 10 candidates | 814,838 |
| Entities with 11 or more candidates | 699,897 |

- **How we ensured true matches were not lost:** recall was measured against the full training ground truth after every change, and new keys were designed from the misses.

| Blocking version | True entity ranked first | True entity kept |
| --- | --- | --- |
| v1: unigram and ordered bigram keys, top 10 | 88.8% | 94.45% |
| v2: + spaceless-name and name x address cross keys | 94.4% | 97.05% |
| v3: + learned transliteration map | 96.2% | 98.2% |
| Final: + score-ratio pruning (score at least 0.5 x best) | 97.99% | 98.13% |

Score-ratio pruning cut the train candidate set to 1.73 pairs per record. The remaining misses (domain-style names, dropped name words with no address, invented names with partial addresses, glued or split words, typos in rare words) are targeted by the rescue pass in Section 4.7.

---

## 4. Matching Model

### 4.1 Stage 1: pair classifier

**Features used** (60, computed for every record-candidate pair):

- **Name features:** RapidFuzz ratio, token-sort, token-set and partial ratio and Jaro-Winkler on the core names; ratio and token-set on the full normalized names; name token Jaccard; first-token equality; name lengths; whether the record name was transliterated.
- **Address features:** RapidFuzz ratio, token-sort, token-set and partial ratio; address token Jaccard; house-number features (numbers in common, all record numbers present in the candidate, first numbers equal, difference of the first numbers, record numbers missing from the candidate, number counts); missing-address flags on both sides; address lengths.
- **Legal-form features:** legal forms from all three countries are mapped to shared codes (Pvt/Private/Ltd/Limited to one code, Inc/Incorporated, Corp/Corporation, LLC, LLP, SARL, SAS, SASU, EURL, SA, SNC, SCI), and we flag when both names carry a legal form and they disagree. Among hard training pairs this clash occurs in 6% of wrong pairs but in only 0.02% (India) and 0.5% (US) of true pairs, and it is how many French decoys differ from the real entity ("Chevaux Ecole SASU" vs "Chevaux Ecole E.U.R.L.").
- **Blocking features:** blocking score, number of shared keys, rank, ratio and gap to the record's best score, the record's margin between its first and second candidates, and how many records list this Source 1 entity as a candidate or as their top candidate.
- **Relative features:** for name token-set, name ratio, Jaro-Winkler, full-name ratio, address token-set, address ratio and their sum, the gap to the best value among the record's candidates and the rank among them.
- **Other:** source (2 or 3). Country is deliberately not a feature, so France is scored like the training countries.

**Model type:** LightGBM binary classifier (learning rate 0.1, 127 leaves, min_data_in_leaf 100, feature and bagging fraction 0.8, L2 1.0, seed 42, up to 1,000 rounds with early stopping on a 5% holdout; all 1,000 were used). Training uses half of the Source 2/3 records whose candidates never touch a validation entity: 7,175,385 pairs, 48.9% positive. On its own, stage 1 reaches a plain macro F0.5 of 0.97970 on the 110,565 validation entities (threshold 0.65).

### 4.2 Stage 2: record, entity and odd-one-out context

A second LightGBM model (learning rate 0.05, 63 leaves, min_data_in_leaf 200, feature fraction 0.9, bagging fraction 0.8, L2 1.0, 400 rounds) takes the stage 1 probability plus 54 features that a single pair cannot see:

- **Record and entity context (26):** gap to the record's best and second-best candidates, number of candidates, how many other records have this Source 1 entity as their best candidate, how many do so with probability at least 0.5, their summed and maximum probability, the gap to the entity's best name-and-address similarity, and a subset of the stage 1 pair features.
- **Odd-one-out features (28, `src/odd_features.py`):** for each pair we look at the "siblings", the other records whose best candidate is the same Source 1 entity with probability at least 0.5, and ask whether this record is the odd one out: name and address similarity to the siblings relative to the Source 1 entity, whether its first house number differs from the entity's by one of the offsets decoys use, whether the entity's house number is shared by siblings but not by this record, whether it adds name words that no sibling carries, a new unit suffix, or one of the learned decoy or replacement words.

Stage 2 is trained only on pairs whose Source 1 entity is in the validation slice, with 2-fold cross-validation split by record; the out-of-fold predictions are what every downstream validation number uses, and a model fitted on the whole slice scores the test set.

**France guard.** France has no labels, and some stage 2 improvements that help US and India were not trustworthy there. A French record is therefore kept only when the final stage 2 model and an older stage 2 model without the odd-one-out features (the v6 model, 27 features, trained with the duplicate augmentation described in Section 6) pick the same Source 1 entity, and its score is the minimum of the two. The guard can only remove French matches, never add them.

### 4.3 Cross-encoders on the uncertain band

LightGBM on hand-built features plateaued on pairs where the names differ in ways the features cannot express. We therefore fine-tune transformer cross-encoders that read both records at once. The input is `name_full | address` for the record and for the Source 1 candidate (normalized text), encoded as a sentence pair, mean-pooled, followed by one linear output unit, trained with binary cross-entropy.

- **Where they are applied (the band).** On test: every pair ranked first or second for its record by stage 2 with a score in [0.01, 0.99], 902,470 pairs (France 223,482, India 383,737, US 295,251). On validation: slice pairs with an out-of-fold stage 2 score in [0.002, 0.998], 82,133 pairs, 41.3% true.
- **Training pairs.** 1,392,272 pairs (45.5% true) from the train split, never from the validation slice: every pair with a stage 1 score in [0.003, 0.997] (1,304,531 rows) plus a 1.5% random sample of the others, shuffled once and stored as one table.

| Tag | Base model | Training rows | Settings | Hardware |
| --- | --- | --- | --- | --- |
| ce1 | intfloat/multilingual-e5-small | rows 0 to 250,000 | word embeddings frozen, fp32, AdamW lr 3e-5 (head lr x20), batch 64, max length 128, 1 epoch | Apple M2, MPS |
| ce2 | ce1 continued | rows 250,000 to 750,000 | as ce1, lr 2e-5 | Apple M2, MPS |
| kbase | intfloat/multilingual-e5-base | all 1,392,272 | unfrozen, fp16, AdamW lr 3e-5 (head lr x20), batch 128, max length 128, 1 epoch | Kaggle T4 GPU |
| klarge | intfloat/multilingual-e5-large | 900,000 | word embeddings frozen, fp16, batch 32, max length 128, 1 epoch | Kaggle T4 GPU |
| kbge (only if v22 is the final file) | BAAI/bge-reranker-v2-m3 | from the same training table | trained with `src/final_steps/kaggle/gpu_ce.py` | Kaggle P100 GPU |

Ranking quality on the 82,133 validation band pairs (AUC):

| Score | AUC |
| --- | --- |
| stage 1 | 0.9415 |
| stage 2 (out of fold) | 0.9714 |
| ce1 | 0.9269 |
| ce2 | 0.9457 |
| kbase | 0.9591 |

On its own the best cross-encoder is still below stage 2, but it is wrong on different pairs, which is what the stacker exploits.

### 4.4 Stacker

A small LightGBM (15 leaves, learning rate 0.05, 300 rounds, min_data_in_leaf 100, seed 1) reads logit(stage 2), the three cross-encoder logits, logit(stage 1), address_missing, source and the stage 2 rank. It is trained on the validation band and evaluated with 2-fold cross-validation split by Source 1 entity.

| Stacker inputs (besides stage 1/2 and pair flags) | Out-of-fold AUC on the band |
| --- | --- |
| none | 0.9714 |
| ce1 | 0.9777 |
| ce1 + ce2 | 0.9807 |
| ce1 + ce2 + kbase (v15, v17) | 0.9828 |
| ce1 + ce2 + kbase + klarge (v19 to v21) | +0.00013 test-weighted F0.5 over the three-model stack |

A model fitted on the whole band scores the 902,470 test band pairs, and the stacked score replaces the stage 2 score for those pairs. The final chain uses the four-model stack (the stacking step in `src/final_steps/`); a v22 file would add kbge as a fifth input with the same stacker settings. On test it moved 44,615 pairs above 0.85 and 37,592 below. For France the score is the minimum of stage 2 and the stacked score, applied before the guard, so the stacker can only remove French matches.

### 4.5 Decision layer

1. **Best candidate.** Each Source 2/3 record keeps only its highest-scoring Source 1 candidate (one owner per record).
2. **Decoy-word veto.** Words learned from training pairs outside the validation slice: a word is a decoy word when records that match nothing added it to a real business name at least 200 times, while true pairs added it at most 0.2 times per 100 such decoy uses (65 words for India such as `public`, `industries`, `enterprises`; 27 for US such as `group`, `holdings`, `southside`). A record whose name adds one of these words to the candidate's name is not assigned. It fired on 1,444 test assignments.
3. **Expected-F0.5 selection per Source 1 entity (US, India).** Instead of a global threshold, for each Source 1 entity we sort the records that chose it by probability, treat them as independent Bernoulli variables (Poisson-binomial), and keep the top k records where k maximises the expected F0.5 of that entity, including the chance that it is a singleton. Records need a probability of at least 0.01 to be considered (decoy weight 1.0, at most 12 records per entity). It kept 4,963,564 of 5,143,313 candidate assignments on test.
4. **France:** fixed threshold 0.85 on the guarded score.
5. **House-number sibling rule.** If another record from the same source is assigned to the same entity and shares the entity's first house number, a record with probability below 0.95 whose first house number differs from the entity's is dropped (4,001 test assignments).
6. **French exact-address additions** (`src/address_rules.py`): an unassigned French record whose normalized address (region words removed, tokens sorted, containing a number) equals the address of exactly one Source 1 entity is added when its name is either that entity's initials or made only of words that never occur in French Source 1 names (a scrambled or invented name). These pairs lie outside the blocking set (9,360 test records).

### 4.6 France audit rules

France is 15% of test and has no labels, so its rules were built from an audit of assigned and guard-blocked French test pairs and checked with label-free evidence (for example, true copies and decoys differ in how often their text is lowercased or keeps dots, which we measured on the labelled train split). `src/france_rules.py` produces two lists from the stacked scores and the assignment:

- **Removals, 15,479 pairs:** 15,088 type-word swaps (the record replaces the business-type word of the Source 1 name, a common French decoy pattern), 332 "compagnie" variants and 59 duplicate French assignments.
- **Additions, 23,928 pairs** that the guard had blocked: 14,696 where the record only adds generic noise words (fils, cie, services, ...), 5,883 acronym or unique-address matches, 2,207 where the name differs only by legal form, word order, typo or spacing at the same address with a stacked score of at least 0.85, and 1,142 house-number changes that are not the offsets decoys use.

These counts were computed on the v11 inputs; applied to the v15 assignment they removed 14,970 existing matches and added 22,365 pairs (the other 1,563 proposed pairs were already assigned). The released `src/france_rules.py` recomputes both lists from the current scores and assignment: on the v15 scores it proposes 16,711 removals and 23,399 additions, so a rebuild with the released code differs from the v15 file for 4,049 French records. For the final file v17 the rule lists were recomputed with the released code on the re-run French scores (Section 4.8).

### 4.7 Rescue for blocking misses

Records that end up unassigned get a second, independent candidate search with six key families (`src/rescue_candidates.py`): exact sorted address, spaceless core name plus house number, glued or split name words, one-letter typos in rare name words, house number plus address word, and pairs of address words. Keys must be rare (Source 1 document frequency at most 100, lower per family), pairs already in the blocking set are dropped, and each record keeps at most 5 candidates. Each candidate is scored by ce1 and ce2, and a LightGBM acceptance model (13 features: both cross-encoder logits, family flags, name and address similarity, missing address; 15 leaves, 300 rounds) decides; a record is assigned to its best candidate when the acceptance probability is at least 0.7. France is excluded because the acceptance model has no French labels.

On validation the acceptance model was trained on 16,040 eligible pairs (1,133 true) with 2-fold cross-validation (out-of-fold AUC 0.9984). At 0.7 it adds 1,095 records, 1,033 of them correct, and raises the test-weighted F0.5 from 0.98831 to 0.98901. On test it scored 381,243 pairs and accepted 12,451 records (India 7,410, US 5,041). The acceptance model that produced these test additions was fitted on the eligible pairs of the v11 validation assignment; the validation figures above repeat the out-of-fold evaluation on the v15 assignment.

**Threshold selection method:** where labels exist (US and India), thresholds and rule settings were compared on the validation slice with the test-weighted macro F0.5 of Section 5; the rescue threshold sweep in Appendix B is one example (0.7 is the best value, and the score is flat within 0.00002 between 0.6 and 0.85). France has no labels and keeps a fixed, conservative threshold of 0.85.

### 4.8 French address fix and the French rows (v17)

**What the fix changes.** The address normalization was built for US and Indian addresses and handled several French formats badly. `normalize_address_france` in `src/normalize.py` applies four fixes, only to French records:

1. **Region names.** A comma-separated part that is only a region, a department or "France" (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire, Nord, Pas-de-Calais, Gironde, Loire-Atlantique, France) is removed. Source 1 addresses end with the region, while records may carry the department instead (Loire-Atlantique for Pays de la Loire) or no region at all, so these words only added mismatching tokens.
2. **Number sign.** `N°` and `Nº` in front of a house number are removed. Before the fix, transliteration turned `N°56` into the single token `ndeg56` and the house number was lost.
3. **Number suffixes.** A suffix glued to the house number is split off: `19BIS` becomes `19 bis`, `5B` becomes `5 bis`, `12T` becomes `12 ter`.
4. **Street types.** French abbreviations map to one form: `q` to quai, `crs` to cours, `psg` and `pass` to passage, `res` to residence, `bld`, `bvd` and `boul` to blvd, `appt`, `app` and `appartement` to apt, and a few more.

| Raw address | Before the fix | After the fix |
| --- | --- | --- |
| record: N°56 AVENUE DE VILLENEUVE, SAINT-NAZAIRE, Pays de la Loire | `ndeg56 ave de villeneuve st nazaire pays de la loire` | `56 ave de villeneuve st nazaire` |
| Source 1: 56 Avenue de Villeneuve, Saint-Nazaire, Pays de la Loire | `56 ave de villeneuve st nazaire pays de la loire` | `56 ave de villeneuve st nazaire` |
| record: 5B R. Maurice Duval, Pays de la Loire, Nantes | `5b r maurice duval pays de la loire nantes` | `5 bis r maurice duval nantes` |
| Source 1: 5 Bis Rue Maurice Duval, Nantes, Pays de la Loire | `5 bis r maurice duval nantes pays de la loire` | `5 bis r maurice duval nantes` |
| record: 58 Q. LERAY, PORNIC, Loire-Atlantique | `58 q leray pornic loire atlantique` | `58 quai leray pornic` |
| Source 1: 58 Quai Leray, Pornic, Pays de la Loire | `58 quai leray pornic pays de la loire` | `58 quai leray pornic` |

The fix changes 1,233,001 of the 1,694,445 French test addresses (all 259,452 in Source 1, 471,692 in Source 2, 501,857 in Source 3). The new code path runs only for French records: for all 10,007,688 US and Indian test records the normalized output is identical to v15's, and the training data has no French records, so no model and no validation number changes.

**French re-run.** With the v15 models unchanged, the French records went through the pipeline again on the fixed normalization:

- Blocking: 3,700,328 candidate pairs for 1,431,134 French records (3,929,813 for 1,431,690 before). 838,880 of v15's 845,271 French matches are among the new candidates, against 835,911 among the old ones.
- Stage 1, stage 2 and the France guard rescore all new pairs. The cross-encoders were not re-run: of the 205,609 French band pairs, the 147,634 that have v15 cross-encoder scores are re-stacked with the v15 stacker, and the others keep their stage 2 score.
- Decision as in Section 4.5: 862,476 best candidates at 0.85 or above, 733 dropped by the sibling rule, 6,542 exact-address additions. The France rules of Section 4.6, recomputed on the new scores, remove 24,476 pairs (23,901 type-word swaps, 534 "compagnie" variants, 41 duplicates) and add 12,217.
- Result: 856,026 French pairs (model 837,267, France rules 12,217, exact address 6,542). Against v15: 840,214 in both, 15,812 only in the new run, 5,057 only in v15.

**Audit.** France has no labels, so we read random pairs on the raw test text (30 per group):

| Group | Same business | Different | Unsure |
| --- | --- | --- | --- |
| Only in the new run (15,812) | 25 | 3 | 2 |
| Only in v15 (5,057) | 11 | 17 | 2 |

Most of the same-business new pairs involve what the fixes address (`N°`, `No` and `#` prefixes, `3B` = 3 bis, `53BIS` = 53 bis) or a trade suffix such as Et Fils, Cie or Groupe in place of the type word. 12 of the 17 "different" v15-only pairs are type-word swaps at the same address. By reason, the v15-only pairs were removed by the France rules (1,645; 11 of 11 sampled were different), fell below 0.85 on the new score (2,058) or guard (973), were no longer candidates (313), or other (68). A rule classifier over all changed pairs then isolates two weak groups: new pairs whose house number or street differs from the Source 1 address (12 sampled: 9 different, 1 same, 2 unsure), and v15 pairs lost to the new run's score, guard or blocking although the names agree (12 sampled: all 12 the same business).

**Filter against v15** (`python src/france_rules.py partial`, Appendix A):

1. Drop the 1,112 new pairs (not in v15) whose house number or street differs between record and Source 1.
2. Put back the v15 pairs that the new run lost through its score, guard or blocking (not through the France rules) when the names are equal, a typo, spaceless, an acronym or differ only by a trade-suffix word, and the address is the same or missing: 1,449 pairs, of which 1,427 are put back and 22 skipped because the record is already assigned to another Source 1 entity.

The result has 856,341 French pairs, one Source 1 entity per record. Against v15: 841,641 in both, 14,700 added, 3,630 removed, and 16,869 French Source 1 rows change. v17 is v15 with only these French rows replaced.

**Expected effect.** France cannot be scored locally, so the effect is modelled, not measured: a Monte Carlo over the 19,058 affected French Source 1 entities (200 draws, per-entity F0.5 averaged over all 1,732,544 Source 1 entities). Pairs in both files count as true; each changed pair is true with the probability of its audited group (new pairs kept 0.929, new pairs with a different address 0.143, removed by the France rules 0.042, put back 0.976, other v15 drops 0.35). The pessimistic scenario uses 0.80, 0.35, 0.20, 0.85 and 0.55, and the worst case treats the trade-suffix swaps as decoys (0.10).

| Scenario | v17 (filtered) | New run without the filter |
| --- | --- | --- |
| Central | +0.00106 | +0.00084 |
| Pessimistic | +0.00054 | +0.00040 |
| Worst case | +0.00020 | +0.00004 |

These count a Source 1 entity with no predicted and no true matches as 1; counting it as 0 gives +0.00098, +0.00051 and +0.00019 for v17. So v17 is expected to gain about +0.0010 over v15 (+0.0005 pessimistic), and the filter beats the unfiltered run in every scenario.

**Repeatable blocking.** Re-running the old blocking on unchanged French data returned the same number of pairs (3,929,813), but 38,703 records got a different candidate set, and 99,264 of the 99,276 pairs found only in the re-run tie with the record's lowest kept score to within 1e-9. The cause is that DuckDB adds the IDF doubles in a different order on each parallel run, so equal scores differ in the last bits and the tie-break by Source 1 id never applies. `src/block_candidates.py` now rounds each IDF times 10^9 to an integer and sums integers. On a sample of 71,532 French records blocked three times, the old query gave 3,140 and 1,983 differing pairs between runs and the new one 0; old and new differ in 7,102 pairs, 7,098 of them ties at the record's lowest kept score. The v17 French candidates were produced with the old query.

### 4.9 Final steps (v19 to v21)

The final file is built by one chain (`src/final_steps/`, run order in its `README.md`): the four-model stack, the high-confidence recheck, the decision stage of `src/predict_submission.py` (sibling rule, exact-address additions, expected-F0.5 selection), the France audit sets and US/India rescue additions, the v17 French rows, and then the four change sets below. The chain rebuilds v21 exactly. Every change set was accepted by a gate run separately from the code that produced it. Every addition goes to a record that is still unassigned at that point, and each record keeps exactly one Source 1 entity.

**High-confidence recheck (+0.0001 local).** Stage 2 is very confident on most pairs, but some decoys still reach 0.99 and above. The 429,125 US and Indian pairs that are a record's best candidate with a stage 2 score in (0.99, 0.999] are scored by ce1 and ce2, a stacker with the same inputs as Section 4.4 (without kbase and klarge) turns this into a probability, and the pair keeps the lower of its old score and the stacked score. The recheck can only lower a score. Measured on the validation slice it adds +0.0001.

**Fourth cross-encoder (+0.00013 local).** klarge (`multilingual-e5-large`, Section 4.3) joins the stack as a fourth input, measured as +0.00013 test-weighted F0.5 over the three-model stack.

**France push (922 removals, 2,248 additions).** France has no labels, so both sets come from label-free evidence and were cut by hand-read samples and fingerprints:

- *French-adapted cross-encoder (removals).* ce1 was trained further on 100,000 French pairs from the test re-run whose labels come from our own rules, not from ground truth: positives are French pairs with stage 2 and guard scores of at least 0.98 that a name-and-address review rule calls the same business; negatives are pairs scored at most 0.02 plus synthetic decoys made from Source 1 records (a business-type word swapped, the house number moved by a decoy offset, the legal form changed, a word added). It flagged 1,001 assigned French pairs as type-word-swap decoys (all in the file, all French; cross-encoder logit at most -4.02). The gate removed 79 of them whose swapped words are within edit distance 2 or are abbreviations (Etablissements/Ets, Saint/St), because those read as typo copies, leaving **922**. Fingerprints: the removed pairs are lowercase in 4.45% and carry dots in 4.34% of record names, close to the decoy pattern and far from the other type-word pairs kept (0.02% and 0.05%). Hand reads: 20 of 20 random removals were decoys; a fresh 20 from the trimmed set gave 18 decoys, 1 typo copy and 1 unsure. Of 60 hand-labelled French pairs, the removals take out the one labelled different and none labelled same.
- *French rescue (additions).* Unassigned French records whose name is a run-together, domain-style or hashtag form of a French Source 1 name at the same address (equal house numbers) were proposed (2,273). The gate removed 17 non-domain names that the French cross-encoder scores below 0 and 8 multi-word names that share no word with the Source 1 name besides legal forms, leaving **2,248**. Hand reads: 20 of 20 random additions correct; a fresh 20 from the trimmed set 19 of 20.
- Together they touch 3,115 Source 1 entities (0.18%) and change no US or Indian row.

**Name-key rescue, nm1d (+0.00017 held-out).** Some US and Indian records have no address and no useful blocking candidate. For these we build name keys from the Source 1 names: the sorted name words without legal and stop words, the same with one word deleted, and the same with one word reduced to its first letter. A key is used only when at most 3 Source 1 entities of the record's country carry it. Candidates are scored by ce1 and ce2 and a LightGBM acceptance model, and a record still unassigned in the final build is added to its best candidate at acceptance probability 0.7. The threshold was chosen on one half of the validation slice (`hash(s1) % 40 < 20`) and scored on the other: +0.000166 on the evaluation half, +0.000143 on the fitting half (261 slice additions, 244 true). On test it adds **3,169** records, close to the 3,470 the slice predicts.

**hc and reverse rescue (+0.000075 held-out on top of nm1d).** *hc* applies the same kind of search and acceptance model (threshold 0.7) to records whose existing candidates all score below 0.3. *Reverse blocking* runs the search from the Source 1 side, 20 records per Source 1 entity, with its own stage 1 and stage 2 models, and adds a record when its score is at least 0.6 and its best competing entity scores below 0.2. On top of nm1d the two together gain +0.000075 on held-out halves, with every increment positive in all four quarters of the slice. On test: 2,252 hc additions and 563 reverse additions, 324 of which hc already adds, so **2,491** records in total.

**France empty-entity fill (24 strict pairs).** French Source 1 entities that no record is assigned to. The candidates are unassigned French records whose best candidate is such an entity with a stage 2 score of at least 0.8, that the guard also picks, and that the French decision missed only because the lower of the two scores was below 0.85. The gate read all 31 proposals and 30 of a wider pool, dropped every pair with a conflict (another Source 1 of the same name at the same address, or a type-word reading), and kept **24** pairs: all 24 score at least 1.03 with the French cross-encoder, and none looks like a decoy. They affect 20 Source 1 entities, so the expected effect is below +0.00002.

**Optional fifth cross-encoder (v22).** `BAAI/bge-reranker-v2-m3` (568M, Apache 2.0), fine-tuned on a Kaggle P100 with `src/final_steps/kaggle/gpu_ce.py`, enters the stacker as a fifth input. It is part of the final file only if v22 is the last upload.

---

## 5. Validation Methodology

- **Held-out slice.** Every Source 1 entity with `hash(id) % 20 = 0` (DuckDB hash, 110,565 entities) is held out. All Source 2/3 records that have any held-out entity among their candidates are scored, so false merges from records belonging to other entities, or to none, are counted. Stage 1 and the cross-encoders never see a pair whose record touches the slice. Stage 2, the stacker and the rescue model are trained on the slice itself with 2-fold cross-validation, and only out-of-fold predictions are evaluated.
- **Metric.** Macro F0.5 per Source 1 entity exactly as the challenge defines it, singletons included, with one change: a false positive caused by a record that has no true Source 1 entity at all is counted with weight **1.887** in the F0.5 denominator; a false positive from a record that belongs to another entity keeps weight 1.
- **Why.** Section 2.1 shows that test has 1.887 times as many unmatched records per Source 1 entity as train, and those unmatched records (mostly decoys) are exactly what produces false merges. An unweighted local metric under-counts them, so it rewards changes that trade precision for recall. Our early uploads showed this: validation said 0.9817 for v3 and the leaderboard said 0.968. After switching to the weighted metric, local and leaderboard gains moved together (v11: +0.0048 locally, +0.008 on the leaderboard over the 0.974 file). We report the plain F0.5 alongside it.
- **Scope.** The slice covers US and India only. France has no labels, so French changes were judged on label-free evidence and an audit of French test pairs, and they are kept conservative (guard, fixed 0.85 threshold, rescue disabled). The v17 French changes and the France push were judged the same way (Sections 4.8 and 4.9).
- **Late changes.** For the US and Indian change sets of Section 4.9, the acceptance threshold was chosen on one half of the slice (`hash(s1) % 40 < 20`) and the gain is reported on the other half, so the held-out numbers are not tuned on the data they are measured on.

---

## 6. Results & Error Analysis

- **F_0.5 Score (macro):** 0.98901 test-weighted (0.98923 plain) on the 110,565 validation entities for v15 and v17, which share their US and Indian rows. The final file adds measured increments on top: +0.00013 (fourth cross-encoder), +0.0001 (high-confidence recheck), +0.00017 held-out (nm1d) and +0.000075 held-out (hc and reverse). Public leaderboard: v15 0.986784, v17 0.9871, v20 0.988549. Final (v21): 0.988642 (v21, the last upload; v20 scored 0.988549).

| Version | What changed | Local, test-weighted F0.5 | Public leaderboard |
| --- | --- | --- | --- |
| v1 | first full-scale pipeline | older metric | 0.940 |
| v2 | cross and spaceless keys, relative features, stage 2 | older metric | 0.950 |
| v3 | learned transliteration map, legal-form clash, house-number distance | older metric | 0.968 |
| v6 + veto ("0.974 file") | learned decoy-word veto, threshold 0.75 | 0.98241 | 0.974 |
| v10 | odd-one-out stage 2, house-number sibling rule, France guard, French exact-address additions | 0.98470 | n/a |
| v11 | + ce1 stacker, expected-F0.5 selection | 0.98716 | 0.982 |
| v12 | + ce2 | 0.98777 | n/a |
| v13 | + France audit rules (no effect on US/India) | 0.98777 | n/a |
| v14 | + rescue | 0.98847 | n/a |
| v15 | + kbase (Kaggle e5-base) in the stacker | 0.98901 | 0.986784 |
| v17 | French address normalization fix, French re-run, filter against v15 (US and India unchanged) | 0.98901 | 0.9871 |
| v19 | + klarge (four-CE stack), high-confidence recheck, France push (922 removals, 2,248 additions) | +0.00013, +0.0001 | n/a |
| v20 | + name-key rescue nm1d (3,169 additions) | +0.00017 held-out | 0.988549 |
| **v21 (final)** | + hc and reverse rescue (2,491 additions), 24 French empty-entity fills | +0.000075 held-out | 0.988642 (v21, the last upload; v20 scored 0.988549) |
| v22 (only if uploaded last) | v21 chain with bge-reranker-v2-m3 as a fifth cross-encoder | | |

The v15 file assigns 5,817,948 of the 9,969,589 test Source 2/3 records (France 58.9%, India 57.9%, US 58.8%) and leaves 100,309 of the 1,732,544 Source 1 entities empty. The final v17 file assigns 5,829,018 records (France 59.7%, India 57.9%, US 58.8%) and leaves 100,036 Source 1 entities empty. The final v21 file assigns 5,833,349 records, leaves 99,991 Source 1 entities empty and has 19,258,685 candidate pairs. All three pass the official `validate_submission.py`.

**What worked** (validation gains where measured): cross keys and spaceless keys (blocking top 10 recall 94.45% to 97.05%); the learned transliteration map (exact core-name agreement for transliterated Indian names 15.3% to 89.5%); stage 2 context (0.9664 to 0.9714 on the older unweighted metric); odd-one-out stage 2 with the sibling rule (0.98241 to 0.98470; the French parts of v10 do not touch this US/India number); the cross-encoder stack with expected-F0.5 selection (+0.00246); a second cross-encoder (+0.00061); the Kaggle e5-base cross-encoder (+0.00054 before rescue); rescue (+0.00071); the French address fix with the filter against v15 (modelled +0.0010 central, +0.0005 pessimistic, Section 4.8; the leaderboard moved +0.0003); the fourth cross-encoder (+0.00013); the high-confidence recheck (+0.0001); name-key rescue (+0.00017 held-out); hc and reverse rescue (+0.000075 held-out). v17 to v20 moved the leaderboard by +0.00145.

**What did not work** (all measured on the validation slice unless stated):

- **Exact-duplicate decoy augmentation (a leak).** We added synthetic unmatched copies of records to stage 2 training. Only decoys ever have a perfect twin, so the model learned "has a twin, so it is a decoy". It looked like +0.003 locally, but re-scored without synthetic rows it was -0.0017. Removed from stage 2; every later validation number is computed on data with no synthetic rows. The France guard (Section 4.2) is the earlier model that was trained with it, and it only affects French test pairs.
- **Per-country quantile normalization** of the scores: no gain.
- **Self-training** (refitting on the model's own confident predictions): no gain.
- **A larger stacker** (95 features instead of 8): +0.00004 once the cross-encoders were in, not worth the complexity.
- **A third e5-small cross-encoder (CE3)** trained on the remaining 642,272 rows: better alone than ce2 (band AUC 0.9525), but on top of kbase it moved the stacked band AUC only from 0.9828 to 0.9830 and gave no F0.5 gain, so it was dropped.
- **Re-tuning the decision settings** (expected-F0.5 decoy weight, probability floor, sibling cut-off, rescue threshold) on a grid: no held-out gain of at least +0.00005, so the defaults stayed.
- **Broader French change sets.** A larger removal set from the French cross-encoder (3,952 pairs) and a wider empty-entity fill (62 pairs) failed their audits (copy-shaped house-number patterns, conflicting Source 1 entities at the same address) and were not used.
- **"Orphan copies" as the explanation of the leaderboard gap.** We suspected the test contained unmatched copies of true records that our model would merge. The address-less rate of test records matches that of decoys, not of copies, so the hypothesis was rejected and the gap was addressed with the reweighted metric instead.

**Common false positives (wrong merges):** decoys with the same name and a different legal form or house number ("Soutien Amicale SNC, 9 Rue Hans Christian Andersen" next to the real "Soutien Amicale SAS, 2 Rue Hans Christian Andersen"; "toledo womens health, 1788 hamilton st" assigned to "1781 hamilton st"), suffix variants of a real name ("summit foundation cii" vs "summit foundation ii"), and records with no address whose name matches several entities in different cities. Most false merges come from records that have no true Source 1 entity at all, which is why the metric weights them.

**Common false negatives (missed matches):** at v11 the remaining validation loss split into blocking misses (0.0061), true pairs scored below the decision line (0.0073) and false positives (0.002), so recall dominates. Typical misses are heavily transliterated names whose legal words stay unrecognized ("al tek teknaljis piraivet limitet" for "Al Tech Technologies"), true matches whose house number was also perturbed ("6825 kildare ave" vs "6818 kildare ave"), generic names with a missing address, and the 1.87% of true pairs that blocking does not return (the rescue pass recovers part of them).

---

## 7. Models, Licences and Compute

### 7.1 Models and licences

| Component | Model | Licence | Parameters | Trained on / where |
| --- | --- | --- | --- | --- |
| Stage 1 | LightGBM GBDT, 60 features, 1,000 trees of 127 leaves | MIT | tree model | train split outside the slice; Apple M2 CPU |
| Stage 2 | LightGBM GBDT, 55 features, 400 trees of 63 leaves | MIT | tree model | validation slice; Apple M2 CPU |
| France guard | earlier LightGBM stage 2, 27 features | MIT | tree model | validation slice; Apple M2 CPU |
| ce1, ce2 | intfloat/multilingual-e5-small, fine-tuned | MIT | 117,653,760 + 385 (output unit) | train split outside the slice; Apple M2 GPU (MPS) |
| kbase | intfloat/multilingual-e5-base, fine-tuned | MIT | 278,043,648 + 769 (output unit) | train split outside the slice; Kaggle T4 GPU |
| Stacker | LightGBM, 8 features, 300 trees of 15 leaves | MIT | tree model | validation band; Apple M2 CPU |
| Rescue acceptance | LightGBM, 13 features, 300 trees of 15 leaves | MIT | tree model | validation slice; Apple M2 CPU |
| klarge | intfloat/multilingual-e5-large, fine-tuned | MIT | about 560M | train split outside the slice; Kaggle T4 GPU |
| High-confidence recheck stacker | LightGBM, 7 features (ce1, ce2, stage 1, stage 2, pair flags), 300 trees of 15 leaves | MIT | tree model | validation band; Apple M2 CPU |
| French-adapted cross-encoder | ce1 (multilingual-e5-small) trained further | MIT | as ce1 | 100,000 French test pairs with rule-made labels (Section 4.9); Apple M2 GPU (MPS) |
| nm1d, hc acceptance; reverse stage 1 and 2 | LightGBM | MIT | tree models | validation slice, threshold picked on one half; Apple M2 CPU |
| kbge (if used in the final file) | BAAI/bge-reranker-v2-m3, fine-tuned | Apache 2.0 | about 568M | train split outside the slice; Kaggle P100 GPU |

The model hub lists totals of 117,654,272 and 278,044,162 for the two E5 checkpoints; the difference to the figures above is an integer position-id buffer (512 and 514 entries), not trainable weights. The largest model is about 568M parameters (560M without kbge), well under the 8B limit. Libraries: anyascii (ISC), LightGBM (MIT), RapidFuzz (MIT), DuckDB (MIT), PyTorch (BSD-3), transformers (Apache 2.0), plus numpy, pyarrow and scikit-learn; exact versions are pinned in `requirements.txt`.

### 7.2 Compute

- **Apple M2 laptop, 16 GB RAM**, for everything except kbase, klarge and kbge. Every DuckDB step caps its own memory (1.2 to 4 GB) and spill, and heavy joins run in slices. Cross-encoder training on the M2 GPU (MPS): ce1 250,000 pairs in 4,013 s (62 pairs/s), ce2 500,000 pairs in 7,324 s (69 pairs/s). Scoring the 902,470 test band pairs takes 1,268 to 1,274 s per e5-small model (about 740 to 755 pairs/s).
- **One free Kaggle notebook with an NVIDIA Tesla T4 (14.56 GiB usable)** for kbase: one epoch over 1,392,272 pairs, 10,878 steps at 245 pairs/s; the test band was scored in 468 s and the validation band in 40 s. The notebook only received our own training pairs and band tables, built from the challenge files.
- **Free Kaggle T4** also for klarge (multilingual-e5-large, frozen word embeddings, batch 32, 900,000 pairs), and a **free Kaggle P100** for kbge (bge-reranker-v2-m3, only if used in the final file). Both notebooks received only our own training pairs and band tables.
- The French-adapted cross-encoder was trained on the M2 GPU: 100,000 pairs in 1,631 s (62 pairs/s).
- No paid compute and no hosted inference services.

### 7.3 Fair play

- Only the provided challenge data is used. There are no external databases, business registries, APIs, geocoding services or internet data at any point, and no lookup of business identities.
- The only outside artefacts are pretrained weights of public multilingual text encoders (intfloat/multilingual-e5-small, -base and -large, MIT licence; BAAI/bge-reranker-v2-m3, Apache 2.0, if used in the final file), downloaded as initialisation and fine-tuned on pairs from the provided training split. Constraint 5 of the challenge README allows MIT or Apache 2.0 models up to 8 billion parameters; ours are 118M, 278M, 560M and 568M.
- One model, the French-adapted cross-encoder, was trained further on French test pairs. France has no training data, so its labels come from our own scores and rules (high-confidence pairs as positives, low-score pairs and synthetic decoys as negatives), never from ground truth or outside data. On Kaggle, internet access was used only to install the Python packages and download these weights.
- `anyascii` only transliterates characters with a fixed table; it looks nothing up.
- Every learned word list (transliteration map, decoy words, replacement words) and every model is fitted on the training split and its ground truth. The test set has no labels; it is used without labels in the same way as the training records (IDF of blocking keys on the test Source 1 file, per-record and per-entity context).
- Nothing is hard-coded per record or entity. Because France has no training data, the France rules were designed by inspecting unlabelled French test pairs; they are written as general patterns (legal forms, business-type and noise words, house-number offsets, address normalization) and list no individual record or entity. The v17 filter compares the re-run with our own earlier output (v15) using the same kind of general name and address rules; it uses no labels and no outside data. The late change sets (Section 4.9) are produced by models and general rules; the hand-read samples only decided whether a whole set, or a rule-defined part of it, was kept.

---

## 8. Conclusion

Treating the problem as a one-owner assignment, spending most effort on blocking recall, and measuring locally with a metric reweighted to the test set's decoy density gave steady, leaderboard-confirmed progress from 0.940 to 0.986784 (v15), 0.9871 (v17) and 0.988549 (v20). Small multilingual cross-encoders, applied only to the uncertain band and stacked with the gradient-boosted models, were the largest late gain, and free Kaggle GPUs were enough to add e5-base and e5-large. The last steps were narrow corrections, each kept only after a gate: a recheck of very confident pairs, French decoy removals and French rescue from a French-adapted cross-encoder, and name-key and reverse rescue for records blocking never returned. The remaining loss is mostly recall on address-less records with generic names, and precision on French records, where no labels exist.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains `src/`, `README.md` and `requirements.txt`. The README gives the exact run order; in short:

| Step | Scripts |
| --- | --- |
| Normalize | `learn_translit.py`, `build_normalized.py` (uses `normalize.py`) |
| Blocking | `block_candidates.py train`, `block_candidates.py test`, `evaluate_blocking.py` |
| Stage 1 | `build_features.py train`, `build_features.py test`, `train_matcher.py` |
| Word lists | `learn_decoy_words.py`, `decoy_families.py` |
| Stage 2 | `build_odd_features.py train`, `build_odd_features.py test` (uses `odd_features.py`), `train_stage2.py` |
| Cross-encoders | `build_ce_pairs.py`, `train_cross_encoder.py`, `score_cross_encoder.py` (uses `cross_encoder.py`) |
| Stacker | `train_stacker.py` |
| Rescue | `rescue_candidates.py`, `score_rescue.py`, `train_rescue.py` |
| Output | `predict_submission.py 0.85` (uses `decoy_veto.py`, `expected_f.py`, `address_rules.py`, `france_rules.py`), then `package_submission.py` |
| France, v17 | French re-run of blocking, scoring and the decision on the fixed normalization, then `france_rules.py partial` (filter against v15) |
| Final steps (v19 to v21) | `src/final_steps/`: four-model stack, high-confidence recheck, decision and merge of the France audit sets, rescue and v17 French rows, France push, name-key/hc/reverse rescue, France fill; exact commands in `src/final_steps/README.md` |

`predict_submission.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official validator on them. The README lists the v17 French steps and what in them is not yet repeatable from `src/` alone. `src/final_steps/` builds the final file from the scores and change sets those steps produce.

### B. Additional Results

Rescue acceptance threshold on validation (v15 assignment, test-weighted F0.5 before rescue 0.98831):

| Threshold | Records added | Of which correct | Test-weighted F0.5 |
| --- | --- | --- | --- |
| 0.5 | 1,147 | 1,064 | 0.98899 |
| 0.6 | 1,120 | 1,047 | 0.98899 |
| 0.7 (used) | 1,095 | 1,033 | 0.98901 |
| 0.8 | 1,058 | 1,005 | 0.98901 |
| 0.9 | 992 | 948 | 0.98897 |
| 0.95 | 939 | 908 | 0.98895 |

How the v15 test file was assembled:

| Step | Effect on test |
| --- | --- |
| Blocking | 19,225,118 candidate pairs for 9,947,553 records |
| Stacker | 902,470 band pairs rescored |
| Decoy-word veto | 1,444 assignments vetoed |
| Expected-F0.5 selection (US, India) | 4,963,564 of 5,143,313 kept |
| House-number sibling rule | 4,001 dropped |
| French exact-address additions | 9,360 added |
| Before France rules and rescue | 5,798,120 matches |
| France removals | 14,970 removed |
| France additions and rescue | 34,798 added (rescue proposed 12,451) |
| Final | 5,817,948 matches; 100,309 empty Source 1 entities |

How the final v17 file was assembled from v15 (France only):

| Step | Effect on French test records |
| --- | --- |
| Address fix | 1,233,001 of 1,694,445 addresses changed |
| Blocking | 3,700,328 candidate pairs for 1,431,134 records |
| Stacker (v15 cross-encoder scores reused) | 147,634 of 205,609 band pairs rescored |
| Threshold 0.85 | 862,476 best candidates kept |
| House-number sibling rule | 733 dropped |
| Exact-address additions | 6,542 added |
| France rules on the new scores | 24,476 removed, 12,217 added; 856,026 pairs |
| Filter against v15 | 1,112 dropped, 1,427 put back; 856,341 pairs |
| Merged into v15 | 16,869 French Source 1 rows changed, 0 US or Indian rows |
| Final | 5,829,018 matches; 100,036 empty Source 1 entities; 19,250,779 candidate pairs |

How the final v21 file was assembled (the chain in `src/final_steps/`):

| Step | Effect on test |
| --- | --- |
| Four-model stack (ce1, ce2, kbase, klarge) | 902,470 band pairs rescored |
| High-confidence recheck (US, India) | 429,125 pairs in (0.99, 0.999] rescored, score can only fall |
| Decision, France audit sets, rescue, v17 French rows | as for v15 and v17 above |
| France push | 922 French pairs removed, 2,248 added |
| Name-key rescue (nm1d) | 3,169 added |
| hc and reverse rescue | 2,491 added (2,252 hc, 563 reverse, 324 shared) |
| France empty-entity fill | 24 added |
| Final | 5,833,349 matches; 99,991 empty Source 1 entities; 19,258,685 candidate pairs; every match in the candidate file |
