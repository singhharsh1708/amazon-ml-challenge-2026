# Documentation

The full write-up of team Inno8's solution to the Amazon ML Challenge 2026 business entity resolution task. The [project README](../README.md) is the short version; this folder has the detail behind every number in it.

## Where to start

| If you want | Read |
| --- | --- |
| The story in ten minutes | [Project README](../README.md), then [10. Leaderboard journey](10-leaderboard-journey.md) |
| The method, end to end | Deep dives [1](01-problem-and-data.md) to [8](08-validation.md) in order |
| The most reusable ideas | [4. Cross-encoders](04-cross-encoders.md), [5. Decision rules](05-decision-rules.md), [8. Validation](08-validation.md) |
| Every experiment, kept or rejected | [9. Experiment log](09-experiment-log.md) |
| To run it yourself | [Reproduction guide](reproduction.md), then [src/final_steps/README.md](../src/final_steps/README.md) |
| The official solution document | [Methodology](methodology.md) |

## Deep dives

| # | Document | What it covers | Headline numbers |
| --- | --- | --- | --- |
| 1 | [The problem and the data](01-problem-and-data.md) | Task, metric, data sizes, name and address noise, decoys, France, train versus test | 1,732,544 test businesses; test has 1.887 times as many unmatched records per business as train |
| 2 | [Blocking](02-blocking.md) | Normalization, the learned transliteration map, key families, IDF scoring, pruning, recall by version, running on a 16 GB laptop, deterministic scores | 98.13% of true training pairs kept at 1.73 candidates per record |
| 3 | [The two LightGBM stages](03-lightgbm-stages.md) | 60 pair features, the 55-feature context model, odd-one-out features, the duplicate-copy leak, the France guard | uncertain-band AUC 0.9415 (stage 1) to 0.9714 (stage 2) |
| 4 | [Cross-encoders](04-cross-encoders.md) | Four fine-tuned multilingual E5 models on the uncertain band, the stacker, the high-confidence recheck, M2 against T4, free Kaggle GPUs | stacked AUC 0.9834 on 82,133 validation pairs |
| 5 | [Decision rules](05-decision-rules.md) | One owner per record, the decoy-word veto, expected-F0.5 selection per business, the house-number sibling rule, thresholds | the veto upload gained +0.006 on the leaderboard |
| 6 | [France](06-france.md) | The two-model guard, label-free fingerprints, audit rules, the address fix and re-run, the French-adapted cross-encoder, the empty-business fill | 259,452 test businesses with no labels |
| 7 | [Rescue](07-rescue.md) | Four searches for records that blocking missed, their acceptance models and the gates they passed | 12,451, 3,169 and 2,491 test records added |
| 8 | [Validation](08-validation.md) | The held-out slice, the test-weighted metric, out-of-fold training, checks on the evaluator, loss decomposition | a change predicted at +0.00008 scored +0.000093 |
| 9 | [Experiment log](09-experiment-log.md) | Every significant experiment in the order it happened, with its result and the decision | from the first commit on 26 Sep, 03:41 IST |
| 10 | [Leaderboard journey](10-leaderboard-journey.md) | The nine uploads with a recorded public score, the timeline, local against public scores, what each step taught us | 0.940 to 0.988642 |

## Reference documents

- [methodology.md](methodology.md): the solution document prepared for the challenge submission, with the method, experiments and results in one place.
- [reproduction.md](reproduction.md): environment, data location, the full run order and what each script does.
- [../src/final_steps/README.md](../src/final_steps/README.md): commands for the chain that builds the final file.
- [../research/README.md](../research/README.md): the GPU training kit, the Kaggle runner and the France re-run tools.

## Figures

All charts are drawn from hard-coded numbers by [`research/make_figures.py`](../research/make_figures.py) (run it from the repository root with matplotlib installed).

| Chart | Shows | Used in |
| --- | --- | --- |
| [leaderboard_journey.png](../assets/leaderboard_journey.png) | Public F0.5 of each upload with a recorded score, with the last four against the 18:19 IST leaderboard | README, 10 |
| [local_vs_leaderboard.png](../assets/local_vs_leaderboard.png) | Local validation against the public score per upload, and the gap between them | README, 5, 10 |
| [local_gains_by_step.png](../assets/local_gains_by_step.png) | Local F0.5 gain of each step from v10 to v21 | 10 |
| [cross_encoder_auc.png](../assets/cross_encoder_auc.png) | Band AUC of each model alone, and of the stack as models are added | README, 4, 10 |
| [remaining_loss.png](../assets/remaining_loss.png) | Where the local loss sat at v11: blocking misses, true pairs below the line, false positives | 2, 10 |
| [candidates_per_entity.png](../assets/candidates_per_entity.png) | Candidates per Source 1 business in the final candidate file, by bucket and by country | 2, 10 |

## Conventions

- **Local** means test-weighted macro F0.5 on the 110,565 held-out US and Indian Source 1 businesses, unless a page says otherwise. Local numbers for v1 to v3 come from an older, unweighted metric on a different slice and are not comparable with later ones.
- **Public** or **LB** means the public leaderboard score.
- Counts are on the test set unless marked as train or validation.
- Times are IST. The challenge ran from 25 Sep 2026 00:00 to 27 Sep 2026 23:59.
- Version numbers skip because many builds were only evaluated locally. The uploads with a recorded public score are v1, v2, v3, v6 + veto, v11, v15, v17, v20 and v21.

## Glossary

| Term | Meaning |
| --- | --- |
| Source 1 | The clean reference list: one row per business (also called an entity) |
| Sources 2 and 3 | Noisy records to be matched to Source 1; each belongs to at most one business, or to none |
| Decoy | A record that copies a real business and changes one detail, so it matches nothing |
| Singleton | A Source 1 business with no true match in Sources 2 and 3 |
| Blocking | Picking a short list of candidate businesses for each record before any model runs |
| Stage 1, stage 2 | The two LightGBM models: pair features first, then record, entity and odd-one-out context |
| Odd-one-out features | Stage 2 features that compare a record with its siblings, the other records whose best candidate is the same business |
| Band | The uncertain pairs rescored by the cross-encoders: ranked first or second for their record, stage 2 score in [0.01, 0.99] on test |
| ce1, ce2, ce3 | Cross-encoders fine-tuned from `multilingual-e5-small` on the Apple M2 (ce3 was dropped) |
| kbase, klarge, kbge | Cross-encoders fine-tuned on free Kaggle GPUs from `multilingual-e5-base`, `multilingual-e5-large` and `bge-reranker-v2-m3` (kbge was dropped) |
| Stacker | The small LightGBM that combines stage 1, stage 2 and cross-encoder scores on the band |
| High-confidence recheck | Rescoring US and Indian pairs with a stage 2 score in (0.99, 0.999]; it can only lower a score |
| Test-weighted F0.5 | The official metric, with a false positive from a record that matches nothing counted 1.887 times |
| Expected-F0.5 selection | Keeping, for each business, the set of records that maximises its expected F0.5 |
| Decoy-word veto | Refusing an assignment when the record adds a learned decoy word (such as `enterprises` or `holdings`) to the business name |
| Sibling rule | Dropping a record with probability below 0.95 whose first house number differs from the business's, when another record from the same source shares it |
| France guard | An older stage 2 model that must pick the same business before a French match is kept |
| Type-word swap | A French record that replaces the business-type word of the name (Club for Ecole, for example) at the same address; treated as a decoy |
| France push | The late French change set: 922 removals from a French-adapted cross-encoder and 2,248 French rescue additions |
| Rescue | A second candidate search for records that end up unassigned |
| nm1d | Name-key rescue for address-less US and Indian records with no useful candidate |
| hc | Rescue for address-less records whose blocking candidates all score below 0.3 |
| Reverse blocking | A candidate search run from the Source 1 side, 20 records per business |
| Held-out half | Half of the validation slice that was not used to pick a change's threshold; late gains are reported on it |
| The 0.974 file | v6 with the decoy-word veto at threshold 0.75: public 0.974, local 0.98241 |
