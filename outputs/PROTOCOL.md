# Evaluation protocol (shared by all methods)

Code: `src/join.py` (matching), `src/labels.py` (targets), `src/protocol.py` (splits, pool, fallback), `src/evaluate.py` (metric). Every number below was produced by these scripts (see `outputs/LOG.md`, `join_stats.json`, `step3_summary.json`).

## 1. Matching views to programmes
- A view is the interval [start_time, start_time + duration) of one user on one channel (`vsetv_id` = schedule `channel_id`). The view end is computed from `duration`, not from `stop_time`, because 117 views crossing the 2020-03-29 DST change have stop_time − start_time = duration + 3600 s. All timestamps are naive local wall-clock time in both sources.
- 17 exact duplicate view rows are removed. Overlapping or touching views of the same user on the same channel are merged into one block (5,821,970 → 5,441,780 rows), which removes 116,216,293 s (1.35%) of double-counted time. Views on different channels are never merged.
- Airings: schedules 11-20 and 21-30 concatenated; 1 airing with duration 0 removed. Airing interval = [start, start + duration).
- Watched seconds of a (user, airing) pair = sum over the user's merged blocks on that channel of max(0, min(view_end, airing_end) − max(view_start, airing_start)). Screen time `frac` = watched seconds / airing duration (max 1.0 after merging; without merging 24,005 pairs would exceed 1).
- Verified against a pure-Python brute force on 30 users (19,767 pairs, 0 mismatches) and three hand-computed examples from the raw CSV lines.

## 2. Threshold and tv_show_id = 0
- A (user, airing) pair *qualifies* when frac ≥ 0.8. The threshold comes from the competition's label recipe; we keep it so our validation labels mimic the hidden test labels. Sensitivity: thr 0.5 / 0.8 / 1.0 gives 938,121 / 611,748 / 421,600 qualifying pairs in weeks 11-30 (`threshold_sensitivity.csv`).
- Airings with tv_show_id = 0 are never items (never recommended, never in a truth list). Their watched seconds (41.2% of merged watch time) are kept only as context in the watch-time accounting.
- `tv_show_id` is series-level (median 6 airings per id over the three schedules), so a show can qualify several times per user.

## 3. Target recipe and tie-break
For a window [start, end): count each user's qualifying airings per tv_show_id; the target is the user's 5 most frequent tv_show_ids. Ties are broken by total watched seconds over the qualifying airings (desc), then tv_show_id (asc). **This tie-break is our convention: the competition's tie handling is unobservable.**

## 4. Splits
| | train interactions (airing_start) | target window | users | pool |
|---|---|---|---|---|
| val | ISO weeks 11-20: 2020-03-09 ≤ t < 2020-05-18 (1,824,077 pairs) | ISO weeks 21-30: 2020-05-18 ≤ t < 2020-07-27 | 4,030 users with ≥ 5 qualifying shows in the target window (ALL); 1,150 of them are submission users (SUB) | 6,376 shows |
| test | ISO weeks 11-30: 2020-03-09 ≤ t < 2020-07-27 (3,014,110 pairs) | ISO weeks 31-42 (hidden) | the 1,260 template users, template order | 6,642 shows |

- Each evaluated validation user has exactly 5 truth ids, as in the competition.
- **Pool** = labelled tv_show_ids (tv_show_id ≠ 0, duration > 0) with at least one airing starting inside the target window, taken from every schedule file available to the split. The files hand over per channel in the morning, so the 11-20 file holds 283 labelled airings on 2020-05-18 and the 21-30 file holds 267 on 2020-07-27. Using only "schedule 21-30 from 05-18" would give 6,363 shows (val), and "schedule 31-42 only" would give 6,636 (test). Every validation truth id is in the pool (asserted).
- The 31-42 schedule is used only for the test pool and for show metadata. It is never joined to viewing data, and no viewing from weeks 31-42 exists in our data.
- Training data for validation = training-window interactions + schedule metadata only.

## 5. Recommendation rules
- **Repeats allowed:** recommending a show the user already watched is permitted for every method (repeat viewing is the main signal: 65.3% of validation truth pairs were watched in training).
- **Cold-start fallback:** if a method yields fewer than 5 distinct candidates (or the user has no usable history), the list is padded from the chosen popularity ranking (`pop_recent`: distinct users with a qualifying airing in the last 4 training weeks) restricted to the pool, skipping duplicates (`pad_with_popularity`).

## 6. Metric
MAP@5, assumed to follow the standard Kaggle definition (the competition page only names "MAP@5"): AP@5 = Σ_{i≤5} P(i)·rel(i) / min(5, |truth|), where a repeated id counts once and the order of the truth is irrelevant. Unit-tested (`src/test_evaluate.py`) and hand-checked (`verify_map_hand.txt`). Uncertainty: 95% percentile bootstrap over users (1,000 resamples, seed 42); comparisons use a paired bootstrap on the same resamples.

## 7. Seeds and reproducibility
Seed 42 everywhere (bootstrap, user sampling). Two full rebuilds of the interactions table gave identical SHA-256 hashes; two full runs of Step 3 gave byte-identical result CSVs and submission files.
