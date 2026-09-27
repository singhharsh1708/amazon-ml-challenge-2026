# 2. Blocking: candidate generation

Part 2 of the technical deep dive. Previous: [1. The problem and the data](01-problem-and-data.md). Next: [3. The two LightGBM stages](03-lightgbm-stages.md). The methodology section this expands is [methodology Section 3](methodology.md#3-candidate-generation-blocking); the code is [`src/block_candidates.py`](../src/block_candidates.py), with normalization in [`src/normalize.py`](../src/normalize.py).

## Summary

Blocking picks, for every Source 2/3 record, a short list of Source 1 businesses worth scoring. A true pair that blocking drops can never be predicted, so this step sets the recall ceiling for the whole pipeline.

| | Value |
| --- | --- |
| Direction | From the record side: each Source 2/3 record gets up to 10 Source 1 candidates |
| Keys | Name and address words, adjacent word pairs, the name with spaces removed, name-word x address-word cross keys; all scoped to the record's country |
| Scoring | Sum of the IDF of shared keys, with over-frequent keys dropped |
| Pruning | Top 10 per record, and only candidates scoring at least half of the record's best |
| Recall on train | 98.13% of the 7,638,365 true pairs kept; 96.16% at rank 1 |
| Size on train | 17,769,228 pairs for 10,287,648 records (1.73 per record) |
| Size on test | 19,225,118 pairs for 9,947,553 records (1.93 per record) |
| Final candidate file | 19,691,692 pairs, 11.37 per Source 1 business, median 10 |
| Hardware | One 16 GB laptop, DuckDB with a 4 GB memory cap, about 30 minutes per split |

## Contents

- [2.1 Why blocking decides the ceiling](#21-why-blocking-decides-the-ceiling)
- [2.2 Normalization highlights that matter for blocking](#22-normalization-highlights-that-matter-for-blocking)
- [2.3 Key families](#23-key-families)
- [2.4 IDF scoring](#24-idf-scoring)
- [2.5 Top-10 and score-ratio pruning](#25-top-10-and-score-ratio-pruning)
- [2.6 Recall by version](#26-recall-by-version)
- [2.7 What blocking still misses](#27-what-blocking-still-misses)
- [2.8 Candidate set statistics](#28-candidate-set-statistics)
- [2.9 Engineering on a 16 GB laptop](#29-engineering-on-a-16-gb-laptop)
- [2.10 Deterministic scores](#210-deterministic-scores)
- [2.11 Limitations and lessons](#211-limitations-and-lessons)

---

## 2.1 Why blocking decides the ceiling

Train has 10,320,219 Source 2/3 records and 2,206,821 Source 1 businesses: about 2.28 x 10^13 possible pairs. Test has about 1.73 x 10^13. Blocking reduces this to under two candidates per record while keeping 98.13% of the true training pairs.

The two failure modes have different costs:

- **A missed true pair** is lost for good. No later model sees it. At the v11 checkpoint, fixing only the blocking misses would have regained 0.0061 of validation F0.5, against 0.0073 for fixing only the true pairs scored below the decision line and 0.002 for removing only the false positives (the three overlap, so they do not add up).
- **An extra candidate** only costs compute and gives the matcher one more wrong option. The score-ratio rule in [2.5](#25-top-10-and-score-ratio-pruning) keeps this small.

![F0.5 regained at v11 if one kind of error were fixed on its own](../assets/remaining_loss.png)

We block from the record side because the rest of the pipeline is an assignment from that side: each Source 2/3 record picks one Source 1 business or none ([1.4](01-problem-and-data.md#14-the-data-at-a-glance)). The output, `data/candidates/{train,test}.parquet`, has one row per (record, business) pair with the columns `rid`, `s1`, `score`, `nkeys` and `rk`. The score, the number of shared keys and the rank are reused as stage 1 features ([3. The two LightGBM stages](03-lightgbm-stages.md)).

## 2.2 Normalization highlights that matter for blocking

Blocking keys are built from normalized text, so normalization decides which true pairs can share a key at all. [`src/build_normalized.py`](../src/build_normalized.py) streams each raw file with pyarrow in 200,000-row chunks through a process pool on every core and writes one parquet per split and source with `entity_id`, `country`, `name_full`, `name_core`, `address`, `name_translit` and `address_missing`. Name keys use `name_core`; address keys use `address`.

### What the normalizer does

Illustrative inputs, mostly built from the examples in [1.5](01-problem-and-data.md#15-noise-patterns), run through [`src/normalize.py`](../src/normalize.py):

| Input name | Full name | Core name (used for keys) |
| --- | --- | --- |
| Kelovantagehalo Co DBA: Heritage Midstream Inc. | `heritage midstream inc` | `heritage midstream` |
| M/s Ster1ing Traders Pvt. Ltd. | `m s sterling traders pvt ltd` | `sterling traders` |
| www.omedicine.com | `omedicine` | `omedicine` |
| \*\*\* Shri Balaji Enterprises Private Limited | `shri balaji enterprises private limited` | `balaji enterprises` |
| Wise Data Services LLC | `wise data services llc` | `wise data` |
| अल टेक टेक्नोलॉजीज प्राइवेट लिमिटेड | `al tech technologies private limited` | `al tech technologies` |

| Input address | Country | Normalized |
| --- | --- | --- |
| #61, Second Street, Austin, Texas | US | `61 2nd st austin tx` |
| H.NO 61 Gandhi Nagar, Pune, Maharashtra | India | `61 gandhi ngr pune mh` |
| Door No 00691 MG Road, Pune, MH | India | `691 mg rd pune mh` |
| 12 MG Road, Pune, महाराष्ट्र | India | `12 mg rd pune mh` |
| None | US | (empty, flagged as missing) |

The name steps, in order: transliterate non-ASCII text to ASCII with `anyascii` and lowercase it; turn `&` into "and"; reduce a domain to its label; drop dots; keep only the part after "DBA" or "doing business as"; undo leetspeak digits in mostly-alphabetic words of four or more characters; apply the learned transliteration map (only to names that were not ASCII); remove consecutive duplicate words. The core name then drops legal forms, filler words and stop words in English, French and their common transliterations (`praivet`, `limitet`, `piraivet`, ...) plus single letters. If nothing is left, the core falls back to the full name.

The address steps: drop placeholder and house-number marker tokens (`none`, `null`, `no`, `h`, `hn`, `door`, `nos`, `c`, `o`), strip leading zeros from numbers, map street types and ordinal words to one short form, and map US and Indian state names to their codes. The state table includes transliterated native-script spellings such as `mharastr` and `tmilnatu`: `anyascii` turns the Devanagari "महाराष्ट्र" above into `mharastr`, which then maps to `mh`.

### The learned transliteration map

Without the map, the normalizer turns the Devanagari name above into `al tek teknolojij praivet limited`, with core `al tek teknolojij`, which shares almost no keys with "Al Tech Technologies". [`src/learn_translit.py`](../src/learn_translit.py) learns the fix from the training pairs:

1. Take every matched pair whose Source 2/3 name is non-Latin and has as many words as the Source 1 name (954,525 pairs).
2. Align the words by position and count, for each transliterated word, which English word it lines up with.
3. Keep a mapping when the most frequent English counterpart differs from the word, is seen at least 20 times, and accounts for at least 60% of the word's aligned occurrences.

The result has 521 entries, for example `praivet` to private, `teknolojij` to technologies and `tredimg` to trading. For transliterated Indian names it raises exact agreement of the core name with the true Source 1 name from 15.3% to 89.5%, and it was the v3 blocking change (recall 97.05% to 98.2%, see [2.6](#26-recall-by-version)).

### French addresses

The US/India address path handled French formats badly: region names added tokens that the other side did not have, `N°56` became the single token `ndeg56`, `5B` never matched `5 bis`, and `19BIS` never matched `19 bis`. The final code sends French records through `normalize_address_france`, which applies four fixes:

| Raw address | Before the fix | After the fix |
| --- | --- | --- |
| record: N°56 AVENUE DE VILLENEUVE, SAINT-NAZAIRE, Pays de la Loire | `ndeg56 ave de villeneuve st nazaire pays de la loire` | `56 ave de villeneuve st nazaire` |
| Source 1: 56 Avenue de Villeneuve, Saint-Nazaire, Pays de la Loire | `56 ave de villeneuve st nazaire pays de la loire` | `56 ave de villeneuve st nazaire` |
| record: 5B R. Maurice Duval, Pays de la Loire, Nantes | `5b r maurice duval pays de la loire nantes` | `5 bis r maurice duval nantes` |
| record: 58 Q. LERAY, PORNIC, Loire-Atlantique | `58 q leray pornic loire atlantique` | `58 quai leray pornic` |

1. **Region names.** A comma-separated part that is only a region, a department or "France" is removed.
2. **Number sign.** `N°` and `Nº` before a house number are removed.
3. **Number suffixes.** `19BIS` becomes `19 bis`, `5B` becomes `5 bis`, `12T` becomes `12 ter`.
4. **Street types.** French abbreviations map to one form (`q` to quai, `crs` to cours, `bvd` to blvd, `appt` to apt, ...).

For blocking, the fix shrank the French candidate set from 3,929,813 pairs to 3,700,328 while finding more of the earlier French matches: 838,880 of the v15 file's 845,271 French matches are among the new candidates, against 835,911 among the old ones. It changes nothing for the 10,007,688 US and Indian test records.

## 2.3 Key families

Every Source 1 business and every Source 2/3 record is turned into a set of keys. A record and a business become a candidate pair only if they share at least one key that survives the frequency caps. The six families, with the keys produced for an illustrative US record "Wise Data Services LLC, #61, Second Street, Austin, Texas" (core name `wise data`, address `61 2nd st austin tx`):

| Prefix | Family | Built from | Keys for the example | Frequency cap |
| --- | --- | --- | --- | --- |
| `n` | Name word | Core-name tokens of 2+ characters | `data`, `wise` | 50 |
| `a` | Address word | Address tokens of 2+ characters | `2nd`, `61`, `austin`, `st`, `tx` | 50 |
| `N` | Name word pair | Adjacent core-name tokens, sorted so order does not matter | `data_wise` | 500 |
| `A` | Address word pair | Adjacent address tokens, sorted | `2nd_61`, `2nd_st`, `austin_st`, `austin_tx` | 500 |
| `C` | Spaceless name | Core name with spaces removed (4+ characters) | `wisedata` | 500 |
| `X` | Name x address cross | Every core-name token with every address token | `wise\|austin`, `data\|61`, ... (10 keys) | 500 |

That record produces 23 keys. Each key is prefixed with the record's country (`us|Xwise|austin`) and hashed to a 64-bit integer, so a French business can never become a candidate for an Indian record and all joins run on integers.

What each family is for:

- **Words** are the base. They catch pairs that share a rare name word or a rare street name.
- **Word pairs** catch reordered names and addresses, and give words that are too common on their own a second chance in combination.
- **The spaceless name** catches glued and split words: `wisedata` matches both "Wise Data" and "WiseData".
- **Cross keys** combine the two fields. A name word and an address word can each be too common to use alone, yet rare together (the methodology's example is `primary|chelsea`). They let a record with a generic name find its business through its location, and a record with a common street find it through its name.

The spaceless and cross keys together took blocking recall from 94.45% to 97.05% and rank-1 recall from 88.8% to 94.4% (v1 to v2).

## 2.4 IDF scoring

The index is built on the Source 1 file of the same split (train for train, test for test; the test Source 1 file is used without labels):

```
idf(k)       = ln(N1 / df(k))
               N1    = number of Source 1 businesses in the split
               df(k) = number of Source 1 businesses that carry key k
keep k if      df(k) <= 50   for word keys (n, a)
               df(k) <= 500  for all other keys (N, A, C, X)
score(r, s)  = sum of idf(k) over the kept keys k shared by record r and business s
```

A key above its cap is dropped entirely rather than down-weighted. That is what keeps the join small: every posting list in the index has at most 50 or 500 businesses, so no single key can pair a record with more than 500 businesses, however common its words are. Words that are too common on their own still contribute through the pair and cross keys, which have the higher cap.

Because of the caps, surviving keys have similar weights:

| Split | N1 | IDF range of kept word keys | IDF range of other kept keys |
| --- | --- | --- | --- |
| Train | 2,206,821 | 10.70 to 14.61 | 8.39 to 14.61 |
| Test | 1,732,544 | 10.45 to 14.37 | 8.15 to 14.37 |

The heaviest key weighs less than twice the lightest, so the score behaves like "how many rare keys the pair shares", with rarer keys counting a little more. The number of shared keys (`nkeys`) is stored next to the score.

## 2.5 Top-10 and score-ratio pruning

Each record's candidates are ranked by score, with ties broken by the smaller Source 1 id, and two rules decide which are kept:

```sql
row_number() OVER (PARTITION BY rid ORDER BY score DESC, s1) AS rk
...
QUALIFY rk <= 10 AND score >= 0.5 * max(score) OVER (PARTITION BY rid)
```

The top-10 cap (the `TOP_K` environment variable, default 10) bounds the work per record. The ratio rule (`SCORE_RATIO = 0.5`) drops candidates that share far less evidence with the record than its best one. It was the last change to the main blocking:

| Training candidates | Top 10 only (v3) | Top 10 and score at least 0.5 x best (final) |
| --- | --- | --- |
| Pairs | 101,866,356 | 17,769,228 |
| Pairs per record | 9.90 | 1.73 |
| True pairs kept | 98.17% | 98.13% |
| True pairs at rank 1 | 96.16% | 96.16% (the best candidate always passes the ratio test) |

It removed 83% of the training pairs for 0.04 points of recall. Every later step (features, both LightGBM stages, the [cross-encoder band](04-cross-encoders.md)) runs on the smaller set.

After pruning, most records have a single clear candidate:

| | Train | Test |
| --- | --- | --- |
| Records with at least one candidate | 10,287,648 | 9,947,553 |
| Records with no candidate | 32,571 | 22,036 |
| Candidate pairs | 17,769,228 | 19,225,118 |
| Candidates per record: mean / median / max | 1.73 / 1 / 10 | 1.93 / 1 / 10 |
| Records with exactly one candidate | 8,870,649 (86.2%) | 8,217,822 (82.6%) |
| Records at the cap of 10 | 5.6% | 7.2% |
| Candidates per Source 1 business | 8.05 | 11.10 |

Recall by rank on the final training candidates, as printed by [`src/evaluate_blocking.py`](../src/evaluate_blocking.py):

```
true pairs: 7,638,365
recall@1 0.9616  @3 0.9739  @5 0.9777  @10 0.9813  @15 0.9813  @20 0.9813  any 0.9813
source 2: recall@1 0.9637  any 0.9824
source 3: recall@1 0.9597  any 0.9802
```

Of the true pairs that survive blocking, 97.99% are the record's top candidate. The matcher's main job is therefore not to find the right business among ten, but to decide whether the top candidate is a true copy or a decoy.

## 2.6 Recall by version

Recall was measured against the full training ground truth (7,638,365 pairs) after every change, and new keys were designed from the misses.

| Version | Blocking change | True pairs kept | True pairs at rank 1 |
| --- | --- | --- | --- |
| v1 | Name and address words and ordered word pairs, top 10 | 94.45% | 88.8% |
| v2 | + spaceless name and name x address cross keys | 97.05% | 94.4% |
| v3 | + learned transliteration map | 98.2% | 96.2% |
| Final | + score-ratio pruning (at least 0.5 x best) | 98.13% | 96.16% |

In the final code the word-pair keys are order-insensitive (each pair is written in sorted order).

### Recall by source and country (final training candidates)

| Slice | True pairs | Rank 1 | Kept | Missed pairs |
| --- | --- | --- | --- | --- |
| All | 7,638,365 | 96.16% | 98.13% | 143,070 |
| Source 2 | 3,693,619 | 96.37% | 98.24% | 65,044 |
| Source 3 | 3,944,746 | 95.97% | 98.02% | 78,026 |
| India | 3,059,843 | 95.95% | 98.01% | 60,773 |
| US | 4,578,522 | 96.30% | 98.20% | 82,297 |
| Source 2, India | 1,480,545 | 96.75% | 98.43% | 23,285 |
| Source 2, US | 2,213,074 | 96.11% | 98.11% | 41,759 |
| Source 3, India | 1,579,298 | 95.19% | **97.63%** | 37,488 |
| Source 3, US | 2,365,448 | 96.48% | 98.29% | 40,538 |

Source 3 India is the weakest cell. On the held-out validation slice recall is 98.10% (375,278 of 382,531 true pairs), in line with the full set, and 93.60% of matched Source 1 businesses have every one of their true records among the candidates.

## 2.7 What blocking still misses

143,070 true training pairs (1.87%) are not in the candidate set. 30,409 of them belong to records that got no candidate at all; the other 112,661 belong to records that got candidates, just not the right business. The recurring kinds of miss:

- domain-style names such as "omedicine.com";
- names that dropped words and have no address;
- invented names with a partial address;
- words glued or split in a way the spaceless key does not cover;
- typos in rare name words.

Rather than loosening the main blocking (which would add candidates for every record), we added separate searches whose additions go only to records that are still unassigned after the decision step, each accepted by its own model:

| Pass | How it searches | Pairs scored on test | Records added on test | Held-out gain |
| --- | --- | --- | --- | --- |
| Rescue | Six extra key families: exact sorted address, spaceless core name plus house number, glued or split name words, one-letter typos in rare name words, house number plus address word, pairs of address words. Keys must be rare (Source 1 frequency at most 100, lower per family); at most 5 candidates per record | 381,243 | 12,451 | +0.00071 |
| Name-key rescue | Sorted name words without legal and stop words; the same with one word deleted; the same with one word cut to its initial. A key is used only when at most 3 businesses in the country carry it | 22,504 | 3,169 | +0.00017 |
| hc rescue | The same kind of search for records whose existing candidates all score below 0.3 | 24,871 | 2,252 | +0.000075 together with reverse blocking |
| Reverse blocking | From the Source 1 side: 20 records per business, with its own stage 1 and stage 2 models | 204,736 | 563 (324 also added by hc) | (included above) |

All four are covered in detail in [7. Rescue](07-rescue.md). They are US and India only, because their acceptance models need labels. France got its own label-free rescue instead: 2,248 unassigned French records whose name is a run-together, domain-style or hashtag form of a French Source 1 name at the same address (equal house numbers), kept after hand-read gates. The methodology describes all of them in [Sections 4.7 and 4.9](methodology.md#47-rescue-for-blocking-misses).

## 2.8 Candidate set statistics

### Test blocking by country

| Country | Records with candidates | Records without | Pairs | Pairs per record | Pairs per Source 1 business |
| --- | --- | --- | --- | --- | --- |
| US | 3,808,021 | 9,010 | 6,145,197 | 1.61 | 9.27 |
| India | 4,707,842 | 9,723 | 9,150,108 | 1.94 | 11.30 |
| France, original normalization | 1,431,690 | 3,303 | 3,929,813 | 2.74 | 15.15 |
| France, fixed normalization (v17 re-run) | 1,431,134 | 3,859 | 3,700,328 | 2.59 | 14.26 |

French records keep more candidates per record than US or Indian ones. Under the ratio rule that means fewer French records have one clearly dominant candidate, so more of the French decision is left to the matcher.

### The final candidate file

The packaged `candidate_pairs.tsv` is exactly the set of pairs the final models scored:

- the blocking candidates of US and Indian records (15,295,305 pairs, scored by both LightGBM stages);
- the French candidates from the re-run with the fixed French normalization (3,700,328 pairs);
- every pair scored by a rescue step: 445,514 rescue pairs including the French rescue pool, 22,504 name-key pairs, 24,871 hc pairs and 204,736 reverse-blocking pairs;
- every submitted match.

These sources overlap. Their union is **19,691,692 pairs** over all 1,732,544 Source 1 businesses. All 5,833,349 submitted matches are in it (0 missing), and the official validator passes.

![Candidates per Source 1 entity in the final candidate file, overall and by country](../assets/candidates_per_entity.png)

| Candidates per Source 1 business (final file) | Value |
| --- | --- |
| Mean | 11.37 (US 9.59, India 11.81, France 14.51) |
| Median | 10 |
| 90th / 99th percentile | 18 / 37 |
| Maximum | 4,434 |
| Businesses with no candidate | 29 |
| Businesses with 1 to 5 candidates | 190,926 |
| Businesses with 6 to 10 candidates | 795,559 |
| Businesses with 11 or more candidates | 746,030 |

The per-record list is capped at 10, but nothing caps how many records list the same business, so the per-business count has a long tail (maximum 4,434). The stage 1 features count how many records list a business as a candidate or as their top candidate, so the model sees this directly.

How the file grew over the submissions:

| File | Candidate pairs | What was added |
| --- | --- | --- |
| Blocking output (test) | 19,225,118 | |
| v15 | 19,246,911 | Pairs from the French exact-address rule, the France audit rules and the rescue pass |
| v17 | 19,250,779 | The 3,868 French v17 matches the v15 file did not contain |
| Final (v21) | 19,691,692 | Rebuilt as the exact set of scored pairs described above |

## 2.9 Engineering on a 16 GB laptop

The whole blocking step is SQL in DuckDB; Python only orchestrates. The design choices that make 10 million records fit on one laptop:

| Choice | Detail |
| --- | --- |
| Integer IDs | `S2-12345` becomes `20000012345` (source digit times 10^10 plus the number). All joins, group-bys and window functions run on 64-bit integers |
| Hashed keys | Each key is stored as `hash(country \|\| '\|' \|\| key)`, a 64-bit integer, instead of a string |
| Keys generated in SQL | `string_split`, `list_filter` and `unnest` build all six families inside the query; no Python loop touches a record |
| Frequency caps before the join | The index drops keys above 50 or 500 businesses before any record is joined, which bounds the fan-out of every key |
| On-disk database | The index lives in a temporary on-disk DuckDB file, so it does not have to fit in RAM |
| 40 slices | Records are processed in 40 slices (`hash(entity_id) % 40`); each slice is keyed, joined, aggregated, ranked and pruned on its own, so peak memory is set by the index plus one slice |
| Explicit limits | `memory_limit = '4GB'`, `threads = 4`, spill to a temp directory capped at 10 GB, `preserve_insertion_order = false` |
| Compact output | The result is written once as ZSTD-compressed parquet |
| Cleanup | The database file, its WAL and the temp directory are deleted in a `finally` block, even when a run fails |

The core of one slice, slightly trimmed from [`src/block_candidates.py`](../src/block_candidates.py):

```sql
INSERT INTO cand
SELECT rid, s1, score, nkeys,
    row_number() OVER (PARTITION BY rid ORDER BY score DESC, s1) AS rk
FROM (
    SELECT q.eid AS rid, s.eid AS s1,
        sum(round(s.idf * 1e9)::BIGINT) / 1e9 AS score,
        count(*) AS nkeys
    FROM (<keys of this slice's Source 2 and Source 3 records>) q
    JOIN s1_index s ON q.h = s.h
    GROUP BY q.eid, s.eid
)
QUALIFY rk <= 10 AND score >= 0.5 * max(score) OVER (PARTITION BY rid)
```

A full split takes about 30 minutes on the M2 with the 4 GB cap. The cap is not generous: building the index runs out of memory at 1,000 MB. Normalization runs before this in a process pool over all cores and writes compressed parquet, so blocking reads columnar files rather than raw TSVs.

## 2.10 Deterministic scores

Late in the challenge we re-ran the old blocking on unchanged French data to check it. It returned the same number of pairs (3,929,813), but 38,703 records got a different candidate set. 99,264 of the 99,276 pairs found only in the re-run tied with the record's lowest kept score to within 1e-9.

The cause was floating-point summation order. DuckDB adds the IDF doubles in a different order on each parallel run, so two candidates with mathematically equal scores differ in the last bits, and the tie-break by Source 1 id never applies. The fix is the `round(s.idf * 1e9)::BIGINT` in the query above: each IDF is scaled to an integer and the integers are summed, which gives the same result in any order.

| Check (71,532 French records, blocked three times) | Old query | New query |
| --- | --- | --- |
| Pairs that differ between runs | 3,140 and 1,983 | 0 |
| Pairs that differ between the old and new query | 7,102, of which 7,098 are ties at the record's lowest kept score | |

The v15 candidate files and the v17 French candidates were produced with the old query, so a rebuild with the released code can differ from them in tie pairs. The saved intermediate files are the reference for those versions.

## 2.11 Limitations and lessons

**Lessons**

- **Measure recall after every change, on the full ground truth.** Every key family after v1 was designed by reading the pairs the previous version missed.
- **Rare combinations of exact tokens go a long way.** The spaceless and cross keys lifted recall from 94.45% to 97.05% while every join stayed an exact join on integers.
- **Prune by relative score, not only by rank.** The 0.5 x best rule cut the candidate set by 83% for 0.04 points of recall and made every later step cheaper.
- **Fix the text before adding keys.** The French address fix made the French candidate set smaller (3,929,813 to 3,700,328 pairs) and still found more of the earlier French matches.
- **Make scores deterministic.** Summing floats in parallel made ties unstable between runs; integer sums fixed it.

**Limitations**

- Blocking runs only from the record side. The Source 1 side is searched only as a late rescue (reverse blocking, 20 records per business).
- 1.87% of true training pairs never reach the matcher. Rescue recovers part of them, but it only runs for records that are still unassigned, so a record whose true business was missed and that was assigned to a wrong one stays wrong.
- The transliteration map is learned from all training pairs, including the held-out validation slice. It only affects non-Latin names, and its effect on the validation numbers was not measured.
- Blocking uses DuckDB's `hash()`, and so does the validation split. Keep DuckDB at the pinned version (1.5.5) to reproduce both.

Previous: [1. The problem and the data](01-problem-and-data.md). Next: [3. The two LightGBM stages](03-lightgbm-stages.md). Full details: [methodology.md](methodology.md). Run order: [reproduction.md](reproduction.md).
