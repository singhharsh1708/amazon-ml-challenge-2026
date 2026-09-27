# 1. The problem and the data

Part 1 of the technical deep dive into team Inno8's solution for the Amazon ML Challenge 2026 business entity resolution task. Next: [2. Blocking](02-blocking.md). The full methodology is in [methodology.md](methodology.md), and the run order is in [reproduction.md](reproduction.md).

## Summary

Three sources describe the same businesses in the US, India and, in the test set only, France. They share no identifiers. Source 1 is a clean reference list. Sources 2 and 3 hold noisy copies of those businesses: transliterated names, swapped legal forms, reformatted or missing addresses. Mixed in with the copies are **decoys**, records built to look like a real business while differing from it in one detail. For every Source 1 business we list the Source 2 and Source 3 records that describe it. The score is F0.5 per business, averaged over all businesses, so a wrong merge costs more than a missed match.

Four facts about the data drove most of our design:

1. **Scale.** 2.2M reference businesses and 10.3M noisy records in train. Comparing every pair is out of the question, so blocking sets both the runtime and the recall ceiling.
2. **One owner per record.** No Source 2/3 record in the ground truth belongs to more than one Source 1 business, which lets us resolve from the record side as an assignment problem.
3. **Decoys, and more of them in test.** Test has 1.887 times as many unmatched records per business as train. A validation metric that ignores this rewards the wrong changes.
4. **France.** 15% of the test businesses come from a country with no training labels, a different language, different legal forms and a different address format.

## Contents

- [1.1 Context](#11-context)
- [1.2 The task](#12-the-task)
- [1.3 The metric](#13-the-metric)
- [1.4 The data at a glance](#14-the-data-at-a-glance)
- [1.5 Noise patterns](#15-noise-patterns)
- [1.6 Decoys](#16-decoys)
- [1.7 The France twist](#17-the-france-twist)
- [1.8 Train versus test](#18-train-versus-test)
- [1.9 What this meant for the design](#19-what-this-meant-for-the-design)

---

## 1.1 Context

| | |
| --- | --- |
| Format | Online challenge, 25 Sep 2026 00:00 to 27 Sep 2026 23:59 IST |
| Team | Inno8: Drashtant Mevada (leader), Jenil Gajera, Ramani Dwarkesh, Harsh Singh |
| Our start | First commit on 26 Sep at 03:41 IST, about 28 hours into the window |
| Compute | One Apple M2 Mac (16 GB RAM, MPS GPU) and free Kaggle notebooks (T4, P100) |
| Rules that shaped the design | Provided data only: no business registries, geocoding, APIs or internet data. Pretrained models allowed only under MIT or Apache 2.0 and up to 8B parameters |
| Final public score | 0.988642 macro F0.5 |

The 16 GB laptop matters for everything that follows. The blocking and feature steps are written in DuckDB with explicit memory caps and slicing so that 10M-record joins fit on it (see [2.9](02-blocking.md#29-engineering-on-a-16-gb-laptop)).

## 1.2 The task

### Inputs

| File | Columns | Train rows | Test rows |
| --- | --- | --- | --- |
| Source 1 (reference) | `entity_id`, `business_name`, `business_address`, `country` | 2,206,821 | 1,732,544 |
| Source 2 (noisy) | same four columns | 5,034,616 | 4,887,273 |
| Source 3 (noisy) | same four columns | 5,285,603 | 5,082,316 |
| Ground truth (train only) | `source1_entity_id`, `matched_entity_ids` (comma-separated) | 2,206,821 rows, 7,638,365 matched pairs | none |

IDs have the form `S1-<n>`, `S2-<n>` and `S3-<n>`. Our code turns each ID into one 64-bit integer (source digit times 10^10 plus `n`, so `S2-12345` becomes `20000012345`), which keeps every join on integers.

### Outputs

| File | Columns | Content |
| --- | --- | --- |
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | Exactly one row per test Source 1 business (1,732,544 rows). The list may be empty. Duplicate rows or duplicate IDs inside a list are rejected |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | The exact set of pairs fed to the matching model. It is not scored on the leaderboard; it documents blocking. In our final file every submitted match is in it |

An official validator (`validate_submission.py`) checks both files. Unknown IDs in a match list are not rejected; they simply count as false positives.

## 1.3 The metric

### Definition

For one Source 1 business with true match set `T` and predicted set `P`:

```
precision = |P ∩ T| / |P|
recall    = |P ∩ T| / |T|
F0.5      = 1.25 * precision * recall / (0.25 * precision + recall)
          = 1.25 * TP / (0.25 * |T| + |P|)
```

The final score is the plain mean of F0.5 over **all** Source 1 businesses, singletons included.

| Case | Per-business score |
| --- | --- |
| Singleton (no true matches), empty prediction | 1.0 |
| Singleton, any prediction at all | 0.0 |
| Non-singleton, empty prediction | 0.0 (precision is undefined; the challenge text does not say, and our evaluators use 0) |
| Worked example from the challenge: 3 predicted, 2 correct, both true matches found | 0.714 |

An independent evaluator written with plain Python sets agreed with our SQL implementation to within 4.7e-15 at ten thresholds, so the metric code itself was never a source of error.

### What F0.5 does to decisions

Take a business with 3 true matches:

| Prediction | F0.5 | Loss |
| --- | --- | --- |
| All 3 | 1.000 | 0 |
| All 3 plus one wrong record | 0.789 | 0.211 |
| 2 of the 3 | 0.909 | 0.091 |
| 2 of the 3 plus one wrong record | 0.667 | 0.333 |

Here a wrong record costs more than twice what a missing one does. The same arithmetic gives a break-even probability for adding one more record to a business that already has `m` sure matches. Below it, adding the record lowers expected F0.5:

| Sure matches already predicted (`m`) | 0 | 1 | 2 | 3 | 10 | very large |
| --- | --- | --- | --- | --- | --- | --- |
| Minimum calibrated probability to add a record | 0.500 | 0.727 | 0.759 | 0.771 | 0.791 | 0.800 |

Three consequences:

- **Precision first, but not a single threshold.** The right cut depends on how many matches a business already has. That is why the final decision layer picks, for each business, the set of records that maximises its expected F0.5 ([5. Decision rules](05-decision-rules.md)).
- **Singletons are fragile.** 123,247 training businesses (5.6%) have no match at all. One false merge takes such a business from 1.0 to 0.0.
- **Every business weighs the same.** A business with one true match is all-or-nothing on recall. On the validation slice at one early checkpoint, businesses with exactly one true match were 5.4% of the slice but carried 22.7% of the loss.

### Our local version of the metric

We held out every training business with `hash(id) % 20 = 0` (DuckDB's hash, 110,565 businesses) and scored every record that has any of them among its candidates, so false merges from other businesses, or from none, are counted. On top of the exact challenge metric we weight a false positive by **1.887** when the record causing it has no true business at all. Section [1.8](#18-train-versus-test) explains where that number comes from, and [8. Validation](08-validation.md) covers the protocol in full.

## 1.4 The data at a glance

### Match structure (training ground truth)

| Property | Value |
| --- | --- |
| Matched (Source 1, record) pairs | 7,638,365 |
| Records matched to more than one Source 1 business | 0 |
| Source 2 records that match nothing | 26.6% |
| Source 3 records that match nothing | 25.4% |
| Singleton Source 1 businesses | 123,247 (5.6%) |
| Matches per non-singleton business | 1 to 11, 3.67 on average |
| Matches per business over all businesses | 3.46 |
| Most matches from one source for one business | 5 from Source 2, 6 from Source 3 |

The zero in the second row is the most useful fact in the dataset. Because every record has at most one owner, we never cluster records together. Each Source 2/3 record picks its single best Source 1 candidate or stays unmatched, and a business's matches are the records that picked it.

### Countries

| Country | Train Source 1 | Test Source 1 | Test Source 2+3 records | Test records per Source 1 |
| --- | --- | --- | --- | --- |
| US | 1,323,633 (60.0%) | 663,106 (38.27%) | 3,817,031 | 5.756 |
| India | 883,188 (40.0%) | 809,986 (46.75%) | 4,717,565 | 5.824 |
| France | none | 259,452 (14.98%) | 1,434,993 | 5.531 |

Test is more Indian than train, and it adds a whole country with no labels. No learned model uses country as a feature, so France is scored with the same features as the training countries.

### Scripts and missing fields

Share of records whose business name contains non-ASCII characters (other scripts, diacritics or special characters), and share with no usable address (empty after normalization, for example `None`, `NULL` or blank):

| | Train Source 2 | Train Source 3 | Test Source 2 | Test Source 3 |
| --- | --- | --- | --- | --- |
| Non-ASCII names, India | 27.9% | 18.5% | 27.6% | 18.2% |
| Non-ASCII names, US | 6.7% | 6.8% | 6.3% | 6.4% |
| Non-ASCII names, France | none in train | none in train | 24.5% | 23.9% |
| Missing address, all countries | 3.36% | 3.33% | 2.65% | 2.68% |

Indian names in Sources 2 and 3 are often written in Devanagari, Bengali, Tamil, Telugu or Gujarati. Source 1 is clean: every US and Indian Source 1 name is plain ASCII, no Source 1 address is missing in train or test, and the only non-ASCII Source 1 names are French ones with accents (40,789 names, 15.7% of French Source 1).

## 1.5 Noise patterns

These are the ways a true copy in Source 2 or 3 differs from its Source 1 business. Each row links to where the pipeline handles it.

### Names

| Pattern | Example | Handled by |
| --- | --- | --- |
| Legal suffix swaps | Pvt / Private, Ltd / Limited, LLC, Inc, SARL, SAS | Legal forms dropped from the "core" name; mapped to shared codes for a legal-form clash feature |
| Filler words | Services, Center, Group | Dropped from the core name |
| Prefixes | M/s, Shri, `***` | Dropped from the core name (symbols, single letters and stop words) |
| DBA prefix with an invented name | "Kelovantagehalo Co DBA: Heritage Midstream Inc." | Everything before "DBA" or "doing business as" is cut |
| Leetspeak | "ster1ing" | Digits inside mostly-alphabetic words mapped back to letters |
| Domain-style names | "omedicine.com" | Domain suffix and `www.` stripped |
| Non-Latin scripts | Devanagari, Bengali, Tamil, Telugu, Gujarati | `anyascii` transliteration, then a 521-entry word map learned from training pairs |
| Word reordering, typos | Same words in another order; one wrong letter in a rare word | Order-insensitive keys; token-sort and token-set similarity features; typo keys in the rescue pass |

### Addresses

| Pattern | Example | Handled by |
| --- | --- | --- |
| Street and unit abbreviations | Rd / Road, St / Street, Ngr / Nagar | One short form per street type |
| House-number formats | `#61`, `H.NO 61`, `Door No 61` | Marker tokens dropped, so all three become `61` |
| Leading zeros | `00691` | Stripped |
| Ordinal words | "Second Street" | Mapped to `2nd` |
| State names vs codes vs native script | Maharashtra, MH, महाराष्ट्र | All mapped to the state code |
| Missing, reordered or injected components | City before street, an extra locality, no address at all | Token-set features; blocking on unordered keys; missing-address flags |

[2.2](02-blocking.md#22-normalization-highlights-that-matter-for-blocking) shows the normalizer's actual output on inputs like these.

## 1.6 Decoys

### What they are

Many unmatched Source 2/3 records copy a real business and change exactly one detail:

| Decoy family | Example |
| --- | --- |
| House number moved by a small offset | "toledo womens health, 1788 hamilton st" next to the real business at "1781 hamilton st" |
| Legal form changed | "Soutien Amicale SNC, 9 Rue Hans Christian Andersen" next to the real "Soutien Amicale SAS, 2 Rue Hans Christian Andersen" |
| One word added | "... Enterprises", "... Holdings" appended to a real name |
| Suffix variant of a real name | "summit foundation cii" vs "summit foundation ii" |
| Business-type word swapped (mostly France) | "solidarite club" vs "solidarite sportive" at the same address |

Decoys are the main source of false merges. Most wrong merges in our error analysis come from records that have no true Source 1 business at all, not from confusing two real businesses.

### How decoys differ from true copies

A decoy can be *closer* to the real business than a noisy true copy is. The decoy "Soutien Amicale SNC" differs from the real "Soutien Amicale SAS" in one token, while the true copy "al tek teknaljis piraivet limitet" differs from "Al Tech Technologies" in almost every token. No single similarity score separates the two classes. The signals that do work are specific:

| Signal | True copies | Decoys |
| --- | --- | --- |
| House number | Same number, different format (`#61`, `H.NO 61`, `00691`) | A different number, typically a small offset (1788 vs 1781) |
| Legal form | Different spelling of the same form (Pvt / Private) | A different form. On hard training pairs a legal-form clash appears in 6% of wrong pairs but only 0.02% (India) and 0.5% (US) of true pairs |
| Added words | Filler words. Among validation pairs with a stage 1 score of at least 0.65, records that add `partners`, `council`, `society`, `trust` or `district` are 98.6% to 100% true matches | Learned decoy words. In the same band, a record that adds `group` was a true match 0 times in 76 (US) and 0 in 284 (India) |
| Missing address | Address-less records are almost all true copies: 97.8% in Source 2 and 97.7% in Source 3 (train) | Decoys almost always carry an address |
| Casing and punctuation | In train, 0.2% of true copies are written in lowercase | 2.5% of decoys are lowercase, and they keep dots more often |
| Relation to the other records of the same business | Agrees with them | Is the "odd one out" among them: different house number, unexplained extra word, new suffix |

Two lessons from working with these signals:

- **Added words are not uniformly suspicious.** A learned word list works where a generic "extra word means decoy" rule does not. Our decoy-word veto uses words that unmatched records added to a real name at least 200 times while true pairs added them at most 0.2 times per 100 such decoy uses: 65 words for India (`public`, `industries`, `enterprises`, ...) and 27 for the US (`group`, `holdings`, `southside`, ...).
- **Real decoys do not come in pairs.** We once added synthetic exact copies of unmatched records to stage 2 training. Only the synthetic decoys had a perfect twin, so the model learned "has a twin, so it is a decoy". On real data twins do not separate the classes: among swap-pattern pairs on validation, the swapped-in word reappears in another record of the same business for 11.8% of US decoys and 11.0% of US true matches (7.8% vs 15.4% in India). The augmentation looked like +0.003 locally and measured -0.0017 once scored without synthetic rows, so it was removed ([3. The two LightGBM stages](03-lightgbm-stages.md) tells the full story).

The last row of the table is the idea behind the stage 2 "odd one out" features ([3. The two LightGBM stages](03-lightgbm-stages.md)): a record is compared not only with its candidate business but with the other records that chose the same business.

### True copies that look like decoys

The reverse case limits recall. Some true copies carry exactly the perturbations decoys use:

- a house number that was also perturbed ("6825 kildare ave" for "6818 kildare ave");
- heavy transliteration with unrecognised legal words ("al tek teknaljis piraivet limitet" for "Al Tech Technologies");
- a generic name with no address, where nothing in the record tells branches of the same chain apart.

## 1.7 The France twist

France appears only in test: 259,452 Source 1 businesses (14.98% of test) and 1,434,993 Source 2/3 records, with no labels anywhere.

### What is different about France

| Aspect | US and India | France |
| --- | --- | --- |
| Labels | Yes | None |
| Legal forms | Pvt, Ltd, LLC, Inc, Corp, LLP | SARL, SAS, SASU, EURL, SA, SNC, SCI |
| Name script | Indian names often in Indian scripts | Accents: 15.7% of Source 1 names and about 24% of Source 2/3 names are non-ASCII |
| Address ending | State name or code | Source 1 ends with the region (Pays de la Loire); records carry the department (Loire-Atlantique) or nothing |
| House numbers | `#61`, `H.NO 61`, leading zeros | `N°56`, `5B` for 5 bis, `19BIS`, `12T` for 12 ter |
| Common decoy patterns | Added decoy word, house-number offset | Business-type word swapped at the same address, legal form changed |

Example of a true French copy that the original normalizer mishandled:

```
record:   N°56 AVENUE DE VILLENEUVE, SAINT-NAZAIRE, Pays de la Loire
Source 1: 56 Avenue de Villeneuve, Saint-Nazaire, Pays de la Loire
```

Transliteration turned `N°56` into the single token `ndeg56`, so the house number was lost. The French address fix ([methodology 4.8](methodology.md#48-french-address-fix-and-the-french-rows-v17)) changed 1,233,001 of the 1,694,445 French test addresses, including every one of the 259,452 Source 1 addresses, because all of them end with a region name.

### France is a different distribution, not just a new country

We fitted a label-shift model (BBSE and EM on stage 1 scores) to estimate the share of matched records in test:

- **US and India** give consistent estimates across 11 score thresholds (US 0.6033 to 0.6058, India 0.5950 to 0.5982). The test is the training distribution with fewer matches: a clean prior shift.
- **France** gives estimates that drift from 0.630 down to 0.581 as the threshold moves, and it has 1.28 to 1.74 times the mid-confidence score mass that any US/India mixture predicts. The records themselves look different to the model: a conditional shift.

The mid-confidence French pairs are dominated by word changes. In a read of 30 random French test pairs scored between 0.5 and 0.95, most looked like decoys. A word-swap pattern covers 20.6% of French test records in the 0.65 to 0.99 score band, against 3.2% for the US and 2.6% for India. Typical examples, after normalization:

| Pair (normalized text) | What changed | Looks like |
| --- | --- | --- |
| passion federation / passion maternelle sarl | Organisation word swapped at the same address | Decoy |
| sources amis sas / sources sante sas | Organisation word swapped at the same address | Decoy |
| 21 r louis mie / 16 r louis mie | House number moved | Decoy |
| carnto / carnot | Typo | Noise on a true copy |
| cf / comite de foot | Acronym | Noise on a true copy |

The learned US/India decoy-word list has no French entries, and French word statistics look different: in mid-band test pairs `developpement` was added 4,373 times and dropped 65 times, and `groupe` added 4,610 times and dropped 601 times. Without labels such asymmetries are only hints. After hand audits, trade suffixes such as Et Fils, Cie and Groupe used in place of the type word were read as the same business and treated as noise, while a swapped business-type word at the same address was treated as a decoy.

### How we handled it

Every French decision was made without labels, so the rules are conservative and each one had to pass a gate of hand-read samples and label-free fingerprints (for example, the lowercase and dot rates that separate decoys from true copies in train):

- no country feature in any model, so France is scored like the labelled countries;
- a guard: a French record is kept only when the final stage 2 model and an older one agree on its business, and a fixed 0.85 threshold instead of expected-F0.5 selection;
- audited French rules for type-word swaps and noise words, a French-adapted cross-encoder that removed 922 type-word-swap decoys, and the four French address fixes.

Details are in [methodology Sections 4.2, 4.6, 4.8 and 4.9](methodology.md#46-france-audit-rules).

## 1.8 Train versus test

| | Train | Test |
| --- | --- | --- |
| Source 1 businesses | 2,206,821 | 1,732,544 |
| Source 2 + 3 records | 10,320,219 | 9,969,589 |
| Countries | US, India | US, India, France |
| Records per Source 1 business | 4.6765 | 5.7543 |
| True matches per business | 3.4613 | 3.4613 assumed (label-shift estimate 3.47) |
| Unmatched records per business | 1.2153 | 2.2931 |
| Unmatched share of records | 26.0% | about 39.8% |
| Records with no address | 3.36% (S2), 3.33% (S3) | 2.65% (S2), 2.68% (S3) |
| Labels | Yes | No |

### The 1.887 factor

Test has more records per business than train. If a test business has the same number of true matches as a training one, all the extra records are unmatched:

```
train: 4.6765 records per business - 3.4613 true matches = 1.2153 unmatched
test:  5.7543 records per business - 3.4613 true matches = 2.2931 unmatched
ratio: 2.2931 / 1.2153 = 1.887
```

Three independent checks support the assumption:

- **Label-shift estimate.** BBSE and EM put the matched share of test records at 0.6045 (US) and 0.5968 (India), against 0.739 on validation. That is 3.47 true matches per test business, practically the training value of 3.46.
- **Denser decoy words.** Among stage 1 pairs scored between 0.65 and 0.99, learned decoy words appear in 1.74% of US test pairs against 0.68% on validation, and in 5.52% of Indian test pairs against 2.54%.
- **Fewer address-less records.** In train almost every address-less record is a true copy (97.8% and 97.7%). The address-less share falls from about 3.3% in train to about 2.7% in test, which is what extra decoys (which carry addresses) would do. We had also considered the opposite explanation, that test hides unmatched copies of true records, and rejected it on this evidence: the address-less rate of test records matches that of decoys, not of copies.

### Why it mattered

An unweighted local metric undercounts false merges from decoys, so it rewards changes that trade precision for recall. Our early uploads showed it: v3 scored 0.9817 on validation and 0.968 on the leaderboard. After we weighted false positives from unmatched records by 1.887, local and leaderboard gains moved together (v11: +0.0048 locally, +0.008 on the leaderboard over the 0.974 file), and the gap between local validation and the leaderboard narrowed to 0.0022 by v15 (0.98901 locally, 0.986784 on the leaderboard). The full upload history is in [10. Leaderboard journey](10-leaderboard-journey.md).

## 1.9 What this meant for the design

| Finding | Design decision | Where |
| --- | --- | --- |
| 2.2M x 10.3M possible pairs | IDF-weighted rare-key blocking, at most 10 candidates per record | [02-blocking.md](02-blocking.md) |
| Every record has at most one owner | Resolve from the Source 2/3 side: each record picks one business or none | [05-decision-rules.md](05-decision-rules.md) |
| 28% and 19% of Indian names in non-Latin scripts | Transliteration plus a word map learned from training pairs | [2.2](02-blocking.md#22-normalization-highlights-that-matter-for-blocking) |
| F0.5, singletons, per-business break-even | Per-business expected-F0.5 selection instead of one global threshold | [05-decision-rules.md](05-decision-rules.md) |
| Decoys are the odd one out among records of the same business | Stage 2 context and "odd one out" features, learned decoy-word veto | [03-lightgbm-stages.md](03-lightgbm-stages.md), [05-decision-rules.md](05-decision-rules.md) |
| 1.887x unmatched density in test | Test-weighted validation metric | [08-validation.md](08-validation.md) |
| France: no labels, conditional shift | No country feature, guard model, fixed threshold, audited rules, French address fixes | [methodology 4.6 to 4.9](methodology.md#46-france-audit-rules) |

Next: [2. Blocking](02-blocking.md), which turns 10 million noisy records into about 19 million candidate pairs while keeping 98.13% of the true training pairs.
