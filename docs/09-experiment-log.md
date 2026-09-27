# 9. Experiment log

Every significant experiment behind the final submission, in the order it happened, with its result and what we did with it. Rejected ideas are here too, with the evidence that sank them, because they shaped the solution as much as the ones that stayed.

**Context.** The challenge window ran from 25 Sep 2026 00:00 to 27 Sep 2026 23:59 IST. Team Inno8 (Drashtant Mevada, Jenil Gajera, Ramani Dwarkesh, Harsh Singh) made its first commit on 26 Sep at 03:41, about 28 hours in, so everything below happened in roughly 44 hours. Compute was one Apple M2 laptop (16 GB, MPS) plus free Kaggle GPU notebooks (T4 and P100).

**How to read it.** "Local" means test-weighted macro F0.5 on the US and Indian validation slice unless noted (see [08-validation.md](08-validation.md)); "LB" means the public leaderboard. Decisions are **Kept**, **Rejected**, or **Check** (a diagnostic that changed how we worked rather than the model). Times are IST.

## Leaderboard uploads

| Upload | When | Main change | Local | LB | Standing |
| --- | --- | --- | --- | --- | --- |
| v1 | before v6 | full-scale blocking and one LightGBM | 0.9537 (older slice, unweighted) | 0.940 | |
| v2 | before v6 | cross and spaceless keys, stage 2 | 0.9714 (older slice, unweighted) | 0.950 | |
| v3 | before v6 | transliteration map, legal-form and house-number features | 0.9817 (older slice, unweighted) | 0.968 | |
| v6 + veto | early 27 Sep | learned decoy-word veto | 0.98241 | 0.974 | |
| v11 | 27 Sep, about 10:00 | first cross-encoder stack, expected-F0.5 selection | 0.98716 | 0.982 | rank 470 |
| v15 | 27 Sep, 16:35 | cross-encoder ensemble, France audit rules, rescue | 0.98901 | 0.986784 | |
| v17 | 27 Sep, about 18:15 | French address normalization fix | 0.98901 (France only changed) | 0.9871 | about rank 230 at 18:19 |
| v20 | 27 Sep, about 21:05 | e5-large, high-confidence recheck, France push, name-key rescue | +0.00013, +0.0001, +0.00017 on US/India | 0.988549 | about rank 100 to 110 against the 18:19 board |
| **v21 (final)** | **27 Sep, 23:36** | **hc and reverse rescue, 24 French fills** | **+0.00008 predicted** | **0.988642** | just below 18:19's #100 |

Leaderboard snapshot at 18:19 on 27 Sep (8,269 teams): #1 0.991829, #50 0.989515, #100 0.988705. The story of each upload is told in [10-leaderboard-journey.md](10-leaderboard-journey.md).

## Phase 1. The full-scale pipeline (from 26 Sep, 03:41)

| # | Experiment | Result | Decision |
| --- | --- | --- | --- |
| 1.1 | Pin DuckDB (1.5.5) and read the data root from an environment variable, so the hash-based validation split is identical on every teammate's machine | hash-based splits reproducible across machines | Kept |
| 1.2 | Blocking v1: IDF-weighted unigram and ordered bigram keys, top 10 candidates per record | true entity ranked first 88.8%, kept 94.45% | Kept as baseline |
| 1.3 | One LightGBM on RapidFuzz name and address features; each record goes to its best candidate above a threshold | local 0.9537, LB 0.940 (v1) | Kept as baseline |
| 1.4 | Spaceless-name keys ("wisedata" against "wise data") and name-by-address cross keys | ranked first 94.4%, kept 97.05% | Kept |
| 1.5 | Stage 2 LightGBM with record and entity context and relative features | 0.9664 to 0.9714 (older metric); LB 0.950 at threshold 0.75 (v2) | Kept |
| 1.6 | Transliteration map learned from 954,525 aligned training pairs (521 entries such as `praivet` to private) | exact core-name agreement for transliterated Indian names 15.3% to 89.5%; blocking ranked first 96.2%, kept 98.2% | Kept |
| 1.7 | Legal-form clash feature (shared codes across countries) and house-number distance | with 1.6: local 0.9817, LB 0.968 at threshold 0.80 (v3) | Kept; the 0.0137 local-to-LB gap became the next problem |

## Phase 2. Decoys, a leak and the veto (night of 26 Sep to morning of 27 Sep)

| # | Experiment | Result | Decision |
| --- | --- | --- | --- |
| 2.1 | Score-ratio pruning: keep a candidate only if it scores at least half of the record's best | ranked first 97.99%, kept 98.13%; train candidates 101,866,356 to 17,769,228 pairs for about 0.0004 recall; 1.73 pairs per record | Kept |
| 2.2 | **Larger K** (deeper candidate lists) | the unpruned top 10 keeps only about 0.0004 more true pairs at almost six times the pairs; 97.99% of true entities are already ranked first, so the misses do not sit at rank 11 | Rejected; misses attacked with new keys instead ([07-rescue.md](07-rescue.md)) |
| 2.3 | **Exact-duplicate decoy augmentation**: synthetic unmatched copies of 90% of the unmatched validation records added to stage 2 training (v6 model) | looked like +0.003 locally. Only decoys ever had a perfect twin, so the model learned "has a twin, so it is a decoy". 1,592,192 of the 5,014,674 out-of-fold rows were synthetic; re-scored with no synthetic rows the change was -0.0017 | Shipped in the v6 upload, then rejected for the main model once the leak was found. The v6 model survives only as the France guard, which can only remove French matches |
| 2.4 | **Learned decoy-word veto**: words that unmatched records add to a real name at least 200 times, with at most 0.2 true uses per 100 decoy uses (65 India words such as `enterprises`, 27 US words such as `holdings`) | 97% precise out of sample; +0.0006 on leak-free validation but +0.006 on the LB; removed 28,525 test assignments in the 0.974 file | Kept (v6 + veto, LB 0.974) |
| 2.5 | Word-swap veto for US and India (record replaces a word of the Source 1 name) | 87.5% to 90.8% of the vetoed pairs were true matches; -0.0007 on validation | Rejected (disabled) |
| 2.6 | Word-swap veto for French organisation words | the swaps behaved like true matches | Rejected |
| 2.7 | Monotone constraints on LightGBM (for robustness under shift) | 0.98132 to 0.97377 in-domain | Rejected |
| 2.8 | Importance weighting with a domain classifier, tested as a US to India transfer proxy for France | 0.9333 to 0.9366, with effective sample sizes of 0.02 to 0.4 | Not adopted |
| 2.9 | **Self-training**: refit on the model's own confident predictions | proxy 0.9333 to 0.9362; on the validation slice no gain, and it risks confirming the model's own decoy errors | Rejected |

## Phase 3. Validation review and shift diagnosis (early 27 Sep)

With the local number above 0.98 and the LB at 0.974, the team audited the evaluation itself before building further.

| # | Experiment | Result | Decision |
| --- | --- | --- | --- |
| 3.1 | Independent reference evaluator (plain Python) against the pipeline's SQL evaluator; README worked example; official validator on a slice file | largest difference 4.7e-15 at 10 thresholds; example 5/7 = 0.714 reproduced; validator PASS (110,565 rows) | Check: the metric is not the cause of the gap |
| 3.2 | Byte-level reproduction of the 0.974 file from the committed code | models, predictions and output byte-identical | Check: the uploaded file is what the code makes |
| 3.3 | **ID, split and label leakage checks.** (a) synthetic twins (see 2.3); (b) transliteration map scope; (c) decoy-word learner scope; (d) threshold chosen on the scored slice; (e) ID format and ground-truth structure | (a) confirmed, inflated the out-of-fold score by +0.0012 to +0.0014; (b) the map uses all training pairs including the slice: a real leak of unmeasured size, non-Latin names only; (c) the slice is excluded; (d) about 1e-4 optimism; (e) every ID parses as `S[123]-<number>` (largest 999,999,995) so the integer encoding is lossless, and no record belongs to two entities | Check: twins removed; (b) documented; (d) led to the half-split protocol for late changes |
| 3.4 | Label-shift estimation (BBSE and EM) on stage 1 test scores | US and India: clean prior shift, matched share 0.6045 and 0.5968 against about 0.74 on validation, 3.47 true matches per test entity (train 3.46). France: estimate drifts from 0.630 to 0.581, a conditional shift | Check: reweight US/India; treat France separately |
| 3.5 | Density simulation: duplicate 90% of unmatched validation records at evaluation time | costs about 0.0019 of stage 1 F0.5 (0.97970 to 0.97778); re-tuning the threshold recovers +0.00013; expected-F0.5 decoding recovers nothing (0.97969 against 0.97970 on plain scores) | Check: a threshold cannot undo the density shift |
| 3.6 | Public LB noise estimate (random-subset assumption) | public minus private standard deviation about 0.000135 at a 30% public share; paired standard error between two uploads 2e-5 to 5e-5 | Check: every LB step so far was signal |
| 3.7 | **Test-weighted metric**: false positives from unmatched records weighted 1.887 (first used to pick thresholds, then as the headline local number) | the 0.974 file scores 0.98241 on it; the weight is backed by 3.4 | Kept: every later local number uses it |
| 3.8 | **Per-country quantile normalization** of the scores | no gain | Rejected |
| 3.9 | Removing the count features | rejected on validation evidence | Rejected |
| 3.10 | **French decoy-word veto** on `groupe`, `developpement`, `holding`, `participations`, found by add/drop asymmetry on test (`developpement` added 4,373 times, dropped 65) | 11,122 French assignments in the 0.974 file add one of these words; a bet worth about +0.0013 if all are decoys and -0.0004 if all are true | Not adopted: hand reads of French pairs later showed Groupe, et Fils and Cie as trade suffixes in true copies, and the France rules treat them as allowed noise words |
| 3.11 | House-number sibling rule on stage 1 scores | +0.00013 plain, +0.00045 at test density, but 52% to 66% of dropped records were true matches | Deferred; adopted in v10 together with the odd-one-out stage 2 |

## Phase 4. Odd-one-out context and the first cross-encoder (27 Sep morning)

| # | Experiment | Result | Decision |
| --- | --- | --- | --- |
| 4.1 | v10: 28 odd-one-out features in stage 2 (is this record the one whose house number or extra word differs from its siblings), the sibling rule for records below 0.95, the France guard, French exact-address additions | 0.98241 to 0.98470 | Kept (not uploaded) |
| 4.2 | ce1: `multilingual-e5-small` fine-tuned as a cross-encoder on 250,000 pairs on the M2 (4,013 s), stacked with stage 1 and stage 2 on the uncertain band | band AUC 0.9714 to 0.9777 | Kept |
| 4.3 | Expected-F0.5 selection per Source 1 entity on the stacked scores | with 4.2: 0.98470 to 0.98716 (+0.00246); LB 0.982, +0.008 over the 0.974 file against +0.0048 locally; rank 470 | Kept (v11) |
| 4.4 | **Per-entity match caps** per source | would bind on only 147 Source 1 entities | Rejected |
| 4.5 | Record-to-record rescue of blocking misses | rejected on validation evidence | Rejected |
| 4.6 | **Orphan-copy hypothesis**: the LB gap comes from test records that are unmatched copies of true records, which the model would merge | the address-less rate of test records matches decoys, not copies | Rejected; the gap was handled by the weighted metric |

## Phase 5. Cross-encoder ensemble, France rules and rescue (27 Sep, 10:00 to 16:35)

| # | Experiment | Result | Decision |
| --- | --- | --- | --- |
| 5.1 | ce2: ce1 trained further on 500,000 more pairs (7,324 s on the M2) | stacked band AUC 0.9807; +0.00061 (0.98777) | Kept (v12) |
| 5.2 | 95-feature stacker instead of 8 inputs | +0.00004 once ce2 was in | Rejected: not worth the complexity |
| 5.3 | France audit rules: remove type-word swaps, add pairs blocked by the guard that only add noise words, acronyms or unique-address matches; checked with label-free fingerprints (true copies are lowercase 0.2% of the time in train, decoys 2.5%) | no effect on US/India (0.98777); on test 15,479 proposed removals and 23,928 additions on the v11 inputs | Kept (v13; measured on the LB with v15) |
| 5.4 | Key-family rescue with a cross-encoder acceptance model | acceptance AUC 0.9984 out of fold; +0.0007 (0.98847) | Kept (v14) |
| 5.5 | kbase: `multilingual-e5-base`, all weights trainable, 1,392,272 pairs on a free Kaggle T4 (245 pairs/s) | alone AUC 0.9591, stacked 0.9828; +0.00054 before rescue; local 0.98901; LB 0.986784 | Kept (v15) |
| 5.6 | CE3: a third e5-small on the remaining 642,272 pairs | alone 0.9525 (better than ce2), but stacked AUC only 0.9828 to 0.9830 and no F0.5 gain | Rejected |
| 5.7 | klarge, first attempt: `multilingual-e5-large` at batch 64 on the T4 | out of memory | Re-run with frozen word embeddings, batch 32, 900,000 pairs |

## Phase 6. French address normalization (27 Sep, 16:35 to 18:15)

| # | Experiment | Result | Decision |
| --- | --- | --- | --- |
| 6.1 | Re-run the French records with the old normalization first | reproduces the v15 French scores exactly | Check: the re-run is trustworthy |
| 6.2 | Four French address fixes: region and department names, the `N°` sign, number suffixes (`5B` to `5 bis`), street-type abbreviations | 1,233,001 of 1,694,445 French test addresses change; US and Indian output identical for all 10,007,688 records | Kept |
| 6.3 | Blocking determinism: re-block unchanged French data | 38,703 records got a different candidate set, because DuckDB sums floating-point IDF weights in a different order on each parallel run. Rounding each IDF times 10^9 to an integer took the differing pairs between runs from 3,140 and 1,983 to 0 | Kept (fix; the v17 candidates were made with the old query) |
| 6.4 | French re-run on the fixed normalization | 856,026 French pairs; hand reads of 30 pairs per group: new-only pairs 25 same, 3 different; v15-only pairs 11 same, 17 different | Check: build a filter |
| 6.5 | Filter against v15: drop 1,112 new pairs whose house number or street differs, put back 1,427 v15 pairs lost through score, guard or blocking | modelled +0.00106 central, +0.00054 pessimistic, +0.00020 worst case, beating the unfiltered run (+0.00084, +0.00040, +0.00004) in every scenario; LB +0.0003, a third of the central estimate | Kept (v17) |

## Phase 7. Final steps (27 Sep, 18:15 to 23:36)

| # | Experiment | Result | Decision |
| --- | --- | --- | --- |
| 7.1 | klarge (e5-large, Kaggle T4) as a fourth stacker input | +0.00013 over three cross-encoders | Kept (v19) |
| 7.2 | High-confidence recheck: the 429,125 US and Indian best pairs with stage 2 score in (0.99, 0.999] are rescored by ce1, ce2 and a small stacker and keep the lower score | +0.0001 | Kept (v19) |
| 7.3 | French-adapted cross-encoder (ce1 trained further on 100,000 French pairs with rule-made labels, 1,631 s on the M2) to find type-word-swap decoys | flagged 1,001; the gate removed 79 typo-like swaps, leaving 922; removed names are lowercase 4.45% and dotted 4.34% of the time against 0.02% and 0.05% for kept type-word pairs; hand reads 20 of 20 decoys, then 18 of 20 in a fresh sample | Kept (v19) |
| 7.4 | Broader French removal set from the same model | 3,952 pairs; failed its audit | Rejected |
| 7.5 | French rescue: run-together, domain-style or hashtag names at the same address | 2,273 proposed, 2,248 after the gate; hand reads 20 of 20, then 19 of 20 | Kept (v19) |
| 7.6 | Name-key rescue (nm1d) for address-less records with no candidate | +0.000166 on the evaluation half, +0.000143 on the fitting half; 261 slice additions, 244 true; 3,169 on test against 3,470 predicted | Kept (v20; LB 0.988549, +0.00145 over v17 against an estimate of 0.9877 to 0.9882) |
| 7.7 | **Re-tuning the decision settings** on a grid (expected-F0.5 decoy weight, probability floor, sibling cut-off, rescue threshold) | no held-out gain of at least +0.00005 | Rejected: defaults kept |
| 7.8 | hc rescue and reverse blocking | held back at the v20 decision as below the bar; re-measured on top of nm1d: +0.000075 held-out, every increment positive in all four quarters; 2,252 + 563 test additions, 324 shared, 2,491 records | Kept (v21) |
| 7.9 | France empty-entity fill | all 31 proposals and 30 of a wider pool read by hand; 24 strict pairs kept (20 entities, expected under +0.00002); the wider 62-pair fill failed its audit | Strict set kept (v21), wide set rejected |
| 7.10 | Last search for anything worth +0.0005 or more that could still be built | nothing; the address-less chain loss (about 0.005) confirmed irreducible | Check |
| 7.11 | Fifth cross-encoder: `BAAI/bge-reranker-v2-m3` fine-tuned on 200,000 pairs on a free Kaggle P100 | local 0.98841 with five cross-encoders against 0.98844 with four | Rejected; v21 stays the final upload |
| 7.12 | Upload v21 | LB 0.988642: +0.000093 against the +0.00008 the held-out numbers predicted | Final |

## Rejected ideas in one place

| Idea | Why it looked promising | What killed it |
| --- | --- | --- |
| Exact-duplicate decoy augmentation | raises decoy density in training toward test's | a leak: only decoys had twins; +0.003 became -0.0017 when measured cleanly (2.3) |
| Larger K in blocking | more candidates, more recall | the unpruned lists add about 0.0004 recall at almost six times the pairs; misses need new keys, not depth (2.2) |
| Word-swap veto, US and India | decoys often swap one word | 87.5% to 90.8% of swaps were true; -0.0007 (2.5) |
| Word-swap veto for French organisation words | same idea for France | the swaps behaved like true matches (2.6) |
| Monotone constraints | robustness under shift | 0.98132 to 0.97377 in-domain (2.7) |
| Domain-classifier importance weighting | adapt to France without labels | small proxy gain, tiny effective sample size (2.8) |
| Self-training | adapt to France without labels | no gain on validation; confirms its own decoy errors (2.9) |
| Per-country quantile normalization | align score distributions across countries | no gain (3.8) |
| French decoy-word veto on groupe-type words | strong add/drop asymmetry on test | these words turned out to be trade suffixes in true French copies (3.10) |
| Per-entity match caps | no training entity has more than 5 Source 2 or 6 Source 3 matches | would bind on only 147 entities (4.4) |
| Record-to-record rescue | another route to blocking misses | rejected on validation evidence (4.5) |
| Orphan-copy explanation of the LB gap | test has more records per entity | test's address-less rate matches decoys, not copies (4.6) |
| 95-feature stacker | more context for the stacker | +0.00004 (5.2) |
| Third e5-small cross-encoder | best single e5-small | no F0.5 gain on top of kbase (5.6) |
| Broader French change sets | more decoys removed, more entities filled | 3,952 removals and a 62-pair fill failed their audits: copy-shaped house-number patterns, conflicting entities at the same address (7.4, 7.9) |
| Re-tuned decision settings | a grid search might beat the defaults | no held-out gain of at least +0.00005 (7.7) |
| Fifth cross-encoder (bge-reranker-v2-m3) | a different model family | 0.98841 against 0.98844 (7.11) |

## What we would tell another team

1. **Never augment with exact copies of negatives.** If only one class can have a perfect twin, the model learns the twin, and your validation lies to you. Always re-score on data with no synthetic rows.
2. **Rebuild the metric around the test set's density.** Count records per entity in train and test, check that the shift is a prior shift, and weight the false positives that scale with it. Our weight of 1.887 is what made local and leaderboard gains line up.
3. **Under F0.5, recall is still most of the loss.** At 0.98, about 81% of the recoverable loss was missed matches. Blocking recall and rescue deserve as much attention as the models.
4. **Spend model capacity where the uncertainty is.** The cross-encoders only re-read the 902,470 uncertain test pairs, not all 19.2M candidates, which is why they fit on a laptop and a free GPU.
5. **Gate every late change.** Choose on one half, report on the other, require consistency across quarters; for unlabeled data, read samples by hand and look for fingerprints before shipping.
6. **Make blocking deterministic.** Floating-point sums in a parallel query are not repeatable; integer scores are.
7. **Know what is irreducible.** Address-less records with chain names cost about 0.005, and no signal in the data separates them. Abstaining is the right call, so stop spending time there.

See [07-rescue.md](07-rescue.md) for the rescue passes, [08-validation.md](08-validation.md) for the validation design, and [methodology.md](methodology.md) for the full method.
