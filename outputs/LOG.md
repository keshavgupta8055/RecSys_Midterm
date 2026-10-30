# Project log

## Step 1: setup and data audit (2026-10-30)

### What was run
1. `python src/env_info.py` -> `outputs/environment.txt` (Python 3.13.1, Windows 11 10.0.26200, Intel64 Family 6 Model 186, 6 physical / 8 logical cores, 7.69 GB RAM; pandas 2.3.1, numpy 2.3.2, pyarrow 21.0.0, matplotlib 3.10.5, psutil 7.0.0).
2. `python -m src.build_cache`: pyarrow CSV parse with explicit types -> parquet in `cache/`. Output `outputs/load_runtime_memory.csv`.
3. `python -m src.audit` -> all audit CSVs in `outputs/`, figures in `outputs/figures/`, full stdout in `outputs/audit_stdout.txt`, key facts in `outputs/key_facts_step1.csv`.

### Column mapping (raw -> canonical), from the actual CSV headers
| file | raw columns |
|---|---|
| dataset11-30.csv (view log) | user_id, vsetv_id, start_time, **stop_time**, **duraton** |
| export_arh_*-final.csv (schedules) | channel_id, channel_title, start_time, duration, tv_show_title, tv_show_id, tv_show_category, tv_show_genre_1, tv_show_genre_2, tv_show_genre_3, year_of_production, director, actors |
| submission.csv | user_id, tv_show_id (space-separated string "0 0 0 0 0") |

Mapping applied in `src/io_utils.py`: view log `stop_time -> end_time`, `duraton -> duration`. My notes (stop_time, duraton) match the raw file; the competition page names (end_time, duration) are the canonical names used from now on. `vsetv_id` (view log) = `channel_id` (schedule).

Dtypes: user_id is **uint64** (values up to 18,417,531,283,109,304,442 exceed int64 max); vsetv_id/channel_id int32; durations int32; tv_show_id int64; year_of_production kept as string (contains values like "1996-").

### Runtime / memory
- First CSV parse: view log 5.0 s, schedules 2.2 / 2.1 / 2.1 s (rebuilt after the bug fix below: 1.9 / 1.6 / 1.9 s).
- Cached parquet load: view log 0.25 s (178 MB in pandas), schedules 1.6-1.9 s (519 / 521 / 628 MB in pandas, deep).
- Cache size: view log 79 MB parquet (raw CSV 396 MB); schedules 14-17 MB each.
- Full audit (all files in memory): 26.1 s, peak process working set 1034 MB.

### Key numbers
- Row counts match expected exactly: view log 5,821,987; schedules 1,074,235 / 1,074,394 / 1,295,378; submission 1,260.
- View log: 4,914 distinct users, 202 distinct vsetv_id, 0 null cells, 17 exact duplicate rows (34 rows involved).
- View start_time 2020-03-09 00:00:02 to 2020-07-26 23:58:42 (ISO weeks 11-30). 139 days with data; **2020-07-03 has zero rows**; 2020-07-02 has 5,108 and 2020-07-04 has 7,814 (lowest days). Active users per ISO week range 2,406 (week 26) to 4,721 (week 12); weeks 21-26 are visibly lower (`viewlog_active_users_per_week.csv`).
- Timestamps: naive local-time strings, no timezone. View log `YYYY-MM-DD HH:MM:SS` with seconds (98.33% have seconds != 0); schedules `DD.MM.YYYY HH:MM:SS`, always on whole minutes (0% non-zero seconds). The formats differ but both parse to the same naive datetime type. Both appear to use the same local wall clock (Europe/Kyiv): the nonexistent DST hour 2020-03-29 03:00-03:59 is empty in both the view log and the 11-20 schedule (0 rows each).
- View duration: min 61 s, median 396 s, mean 1,480 s, p99 14,079 s, max 342,145 s; 52 views > 24 h. The log appears pre-filtered to views > 60 s.
- duration vs end_time - start_time: **117 rows violate it**, all with (end - start) - duration = +3600 s, and all 117 are exactly the rows crossing 2020-03-29 03:00 (DST spring-forward). Conclusion: `duration` is true elapsed seconds; timestamps are wall-clock. 0 rows with end < start.
- Schedule coverage: 11-20 = 2020-03-08 23:00 to 2020-05-18 12:00 (86 rows on 03-08, 2,431 on 05-18); 21-30 = 2020-05-18 00:00 to 2020-07-27 12:00; 31-42 = 2020-07-27 00:00 to 2020-10-19 12:00. Files do not share any (channel_id, start_time); per channel they are contiguous across files (gap 0 s for 398/399 and 402/403 shared channels).
- Channel overlap: 202 view channels; schedules have 400 / 403 / 404 channels. Intersection 193 (11-20), 196 (21-30), 196 (31-42). View rows on matched channels: 97.07% (vs 11-20 channel set), 97.99% (vs 21-30). 5 view channels appear in no schedule: 149 (1,302 rows), 1174 (86,883), 1508 (7,491), 1509 (8,546), 1510 (9,357).
- Schedule rows within a channel are contiguous: overlapping consecutive pairs 2 / 2 / 8, gaps 396 / 3 / 4, duplicate (channel, start) 1 / 0 / 0.
- Labelled share (tv_show_id != 0): 10.76% / 11.31% / 11.18%. Distinct labelled programmes 5,847 / 6,363 / 6,636. Channels with any labelled airing 177 / 188 / 191.
- Only 8 categories exist among labelled rows (Детям/Kids, Сериалы/Series, Фильмы/Films, Развлечения/Entertainment, Инфо/Info, Познавательное/Educational, Другое/Other, Спорт/Sport); Kids dominate (47,862 of 115,613 labelled rows in 11-20).
- Missingness among labelled rows (11-20 / 21-30 / 31-42): genre_1 4.7/4.5/3.9%, genre_2 40.7/40.1/39.3%, genre_3 82.6/80.5/80.1%, year 0.8/1.2/0.8%, director 13.3/13.7/13.5%, actors 13.2/13.1/13.1%. Category and title never missing.
- tv_show_id is series-level: median 4 airings per id per file (6 over all three files), max 6,201 / 6,132 / 6,691 per file; 44.99% of ids have >1 distinct title (episode titles vary); 0 ids change category, 80 ids have >1 genre_1. Only 13.59% of ids air exactly once.
- Labelled airing duration: median 2,700 s, p1 240 s, p99 13,500 s, max 32,700 s.
- Programme continuity: 6,636 labelled ids in the 31-42 schedule, 5,199 of them also aired (labelled) in the 11-30 schedules. (Schedule metadata only, no test viewing used.)
- submission.csv: columns user_id,tv_show_id; 1,260 rows; 0 duplicate users; all 1,260 appear in the view log and none has zero rows in weeks 11-30 (min 3 rows, median 1,377). 3,654 view-log users are not in the submission.
- View rows per user (all 4,914 users): min 1, median 874, p90 2,429.7, max 38,114.

### Figures (outputs/figures/, dpi 150)
01_view_duration_hist_logx, 02_views_per_user_loglog, 03_views_per_day, 04_schedule_labelled_share, 05_top20_categories_labelled (only 8 categories exist; stated in title), 06_airing_duration_labelled_hist, 07_airings_per_tv_show_id.

### Bugs / issues and fixes
1. **Empty strings not null.** pyarrow's CSV reader by default reads empty string fields as `""`, not null. The first check showed 0 nulls and 928,009 empty strings in genre_2 (11-20), which would have reported 0% missingness. Fix: `strings_can_be_null=True` in the schedule loader, cache rebuilt, recheck shows 928,009 nulls and 0 empty strings.
2. **Overflowing user_id.** user_id values exceed int64; loading as uint64 explicitly avoids overflow/float coercion.
3. **DST artefact.** 117 rows have duration != end - start (+3600 s), all spanning the 2020-03-29 spring-forward. Not a parsing bug; noted so the join in Step 2 uses `duration`/wall-clock consistently.
4. Plot fixes: duration histogram first started its x-axis at 1 s although min = 61 s (bins now start at the data min); views-per-user scatter of counts-of-counts was unreadable and was replaced with a log-binned histogram; "top-20 categories" title now states only 8 exist.

## Step 2: join, interactions, label function (2026-10-30)

### What was run
1. `python -m src.join`: dedup, merge, overlap join, aggregation -> `cache/interactions.parquet`, `cache/watchtime_per_user.parquet`, `outputs/join_stats.json` (stdout `outputs/join_stdout_run1.txt`).
2. `python -m src.verify_step2 hand | brute | repro | merge` (C2, C1, C3, C4) -> `verify_hand_examples.txt`, `verify_bruteforce.csv` (+ `_stdout.txt`), `verify_reproducibility.csv`, `merge_sensitivity.csv`.
3. `python -m src.diagnostics_step2` (A7, D1-D7) -> CSVs below, figures 08-13, `diagnostics_step2_stats.json`, `diagnostics_step2_stdout.txt`.
4. Ad-hoc check of schedule labelling per ISO week -> `schedule_labelling_per_week.csv`.
The 31-42 schedule was not loaded. No model was built.

### Rules used
- **View end** = start_time + duration (seconds), not stop_time/end_time (DST: 117 rows differ by +3600 s). All times are naive local wall-clock, converted to integer seconds.
- **Dedup**: exact duplicate view rows dropped (17).
- **Merge**: within (user_id, vsetv_id), views sorted by v_start; a new block starts when v_start > running max of earlier v_end (touching views, with v_start equal to the previous end, are merged). Views on different channels are never merged.
- **Airings**: schedules 11-20 + 21-30 concatenated; duration <= 0 dropped (1); duplicated (channel_id, start_time) dropped keeping the first (0 left after the previous rule: the only duplicate pair, channel 644 at 2020-03-29 04:00, consisted of the 0-second airing and a 1,800-second airing, so removing the 0-second one resolved it). All airings incl. tv_show_id = 0 enter the join; only labelled ones are saved.
- **Overlap join**: per channel, airings sorted by start; candidates for view [vs, ve) run from the first index where the running max of airing ends > vs up to the last airing with start < ve (np.searchsorted on both), expanded with np.repeat, then filtered on overlap > 0. Using the running max keeps it correct where airings overlap slightly.
- **frac** = watched_seconds / scheduled airing duration. Not filtered in the cache.
- **Targets** (`src/labels.build_targets`): airings with start <= airing_start < end and frac >= thr count. Per (user, show): n_qualifying_airings, watched_seconds_total (sum over the qualifying airings only). Rank by n_qualifying_airings desc, then watched_seconds_total desc, then tv_show_id asc. **The tie-break is our convention**; the competition's own tie handling is unobservable. Full ranked list returned. Week starts by ISO week (`iso_week_start(21)` = 2020-05-18, `iso_week_start(31)` = 2020-07-27).
- **tv_show_id = 0** is dropped as an item, but its watched seconds are counted as watching context in the accounting (A7).

### interactions.parquet columns (3,014,110 rows, 31.5 MB, 4,895 users, 4,363 shows)
| column | type | meaning |
|---|---|---|
| user_id | uint64 | viewer |
| tv_show_id | int64 | labelled series id (never 0) |
| channel_id | int32 | = vsetv_id |
| airing_start | datetime64[s] | scheduled start, naive local time |
| airing_duration | int32 | scheduled duration, s |
| watched_seconds | int64 | sum of overlaps of the user's merged views with the airing (> 0) |
| frac | float64 | watched_seconds / airing_duration, in (0, 1] |
| iso_week | int16 | ISO week of airing_start |

### Key numbers
- Views: 5,821,987 raw -> 5,821,970 after dedup -> 5,441,780 merged blocks (380,190 rows absorbed; 187,000 blocks built from >= 2 raw views). Watched seconds 8,619,206,214 -> 8,502,989,921 (merging removed 116,216,293 s = 1.348% double counting).
- Airings: 2,148,629 in 11-20 + 21-30 -> 2,148,628 kept; 237,118 labelled (11.04%).
- frac max = 1.0; pairs with frac > 1 + 1e-9: 0 (assert passes). Pairs with frac == 1: 421,600.
- Labelled pairs: 3,014,110; airing in ISO week 10: 0; airing in week 31 (>= 2020-07-27): 0 (no view reaches 07-27, so none dropped). 20.30% of pairs have frac >= 0.8.
- **Watch-time accounting** (`watchtime_accounting.csv`, fig 13), all users / submission users: labelled 52.89% / 51.44%; unlabelled (tv_show_id = 0) 41.15% / 42.58%; channels absent from schedules 1.36% / 1.44%; schedule gaps 4.60% / 4.53%; double counting where airings overlap: 4,746 s / 1,808 s (subtracted). Total 8,502,989,921 s / 2,887,792,159 s.
- **Threshold sensitivity** (weeks 11-30, `threshold_sensitivity.csv`): thr 0.8 gives 611,748 qualifying pairs, 4,841 users, 3,866 shows, 4,633 users with >= 5 shows; submission users 196,134 pairs, 1,256 users, 1,241 with >= 5 shows. At thr 0.5 / 1.0: 938,121 / 421,600 pairs.
- **Distinct qualifying shows per user** (thr 0.8, weeks 11-30): all users median 32, p90 105, max 545, 73 users with 0; submission users median 40, p90 116.1, max 329, 4 with 0, **1,241 of 1,260 with >= 5**.
- **Candidate windows** (`candidate_window_target_counts.csv`), users with >= 1 / >= 3 / >= 5 shows:
  - W1 weeks 21-30: all 4,643 / 4,333 / 4,030; submission 1,238 / 1,206 / 1,150.
  - W2 weeks 25-30: all 4,569 / 4,193 / 3,757; submission 1,233 / 1,188 / 1,103.
  - W3 weeks 11-20: all 4,737 / 4,553 / 4,349; submission 1,242 / 1,222 / 1,197.
- **Repeat consumption** (`repeat_consumption_W1_vs_W3.csv`): 4,030 users have >= 5 W1 shows (submission 1,150). Of their 20,150 (5,750) top-5 (user, show) pairs, 50.82% (56.64%) had a qualifying airing in W3 and 65.32% (71.03%) had any watched seconds in W3. 396 of 1,484 distinct W1 target shows (submission: 186 of 774) have no labelled airing in the 11-20 schedule; these make up 7.43% (6.26%) of the target pairs.
- **Popularity** (`top20_shows_by_distinct_users.csv`): #1 ТСН (2400480, Info) 3,995 users; #2 Факты (240081, Info) 3,808. Info is 25.23% of qualifying pairs but 3.78% of labelled airings; Kids 25.15% vs 39.83%; Series 10.27% vs 27.18% (`category_share_qualifying_vs_schedule.csv`, fig 12).
- **Weekly** (`weekly_qualifying.csv`, fig 10): qualifying pairs per week are 26k-49k except weeks 25 / 26 / 27 with 13,732 / 9,948 / 10,933. Schedule labelling in those weeks is normal (labelled share 11.3-11.7%, `schedule_labelling_per_week.csv`), so the dip comes from the viewing side (labelled watched hours 53,164 / 53,926 in weeks 23 / 24 vs 29,749 / 24,651 / 28,147 in weeks 25 / 26 / 27). Week 27 contains the missing day 2020-07-03.

### Verification evidence
- **C1 brute force**: 30 users (seed 42; 11 submission users; includes the owner of the 342,145 s view; the max number of airings spanned by one merged view is 209). Pure-Python dedup + merge + double loop over all airings of the channel. 19,767 labelled (user, airing) pairs compared, **0 mismatches**, identical pair sets. Runtime 124.4 s.
- **C2 hand examples** (`verify_hand_examples.txt`, submission user 10976640701901441368, csv module only): A frac >= 0.8: airing ch 7 2020-03-12 15:20 (6,600 s), 4,361 + 1,156 = 5,517 s, frac 0.835909. B frac < 0.8: ch 3 2020-03-09 09:10 (8,400 s), 8 views, 3,113 s, frac 0.370595. C merged: ch 3 2020-03-29 11:40 (10,500 s), a 77 s view lies inside a 10,838 s view; merged 7,046 s (frac 0.671048), unmerged would give 7,123 s. All three MATCH the pipeline.
- **C3 reproducibility**: two builds from an empty cache (incl. CSV parsing): 57.6 s and 69.0 s wall; 3,014,110 rows each; identical SHA-256 98a7a80a...be72bd; assert_frame_equal passes.
- **C4 merge sensitivity** (full data, `merge_sensitivity.csv`): pairs with frac >= 0.8: merged 611,748 vs unmerged 615,965 (+4,217, +0.69%); without merging, 24,005 pairs exceed frac 1 (max 31.13) and labelled watched seconds rise by 77,964,905.

### Runtime and memory
- join (warm parquet cache): 38.7 s, peak 1,078 MB; from empty cache (incl. CSV parse): 59.0 s inside build, peak 1,471 MB.
- Merge-sensitivity builds 33.3 s / 35.2 s; brute force 124.4 s; diagnostics 35.5 s, peak 1,187 MB. All well below the 4 GB budget.

### Bugs, failure cases and fixes
1. **Brute-force sampling bug (float64 user ids).** `np.array(sorted(set_of_python_ints))` turned the 1,260 submission ids into float64 because they exceed the int64 range; rng.choice then produced rounded ids such as 14399299985567565824 that exist in no file. The first C1 run therefore "verified" 0 views for those users (a vacuous pass). Fixed by sampling from the uint64 column directly and asserting every picked id exists in the view log; rerun: 30 real users, 0 mismatches. Lesson: never route these ids through a numpy array built from Python ints.
2. **Reproducibility script deleted its own evidence.** The "from scratch" step removed `cache/*.parquet`, including run 1's saved copy, so assert_frame_equal crashed (the hashes already matched). Copies now go to `cache/repro/`.
3. **Duplicate airing = zero-length airing.** Step 1's single duplicated (channel, start) pair contains the 0-second airing on channel 644, so the dedup rule drops 0 rows after the duration rule.
4. **Pandas index after concat** (ad-hoc weekly check): a lambda aggregation failed on duplicate index labels after concatenating the two schedules; fixed with `ignore_index=True`.
5. Plot fixes: the fig 10 legend overlapped the bars (moved below, lines drawn above bars); fig 11 was cramped (split into a 0-30 zoom and a full-range panel, with step outlines).
6. Encoding: printing Cyrillic to the Windows console failed under cp1252; scripts are run with `PYTHONIOENCODING=utf-8`, and CSVs with Cyrillic are written as UTF-8 with BOM.

## Step 3: protocol, evaluator, baselines, first submissions (2026-10-30)

### What was run
1. `python -m src.test_evaluate`: 9 AP@5 cases + 1 map_at_k case, all PASS (`test_evaluate_output.txt`).
2. `python -m src.protocol`: split sizes.
3. `python -m src.run_step3` (twice, for reproducibility): stdout in `run_step3_stdout.txt` / `run_step3_stdout_run1.txt`; summary in `step3_summary.json`.
No learned model was built; the 31-42 schedule was used only for the test pool and metadata.

### Decisions
- **Splits** (`src/protocol.py`, `outputs/PROTOCOL.md`): val train = weeks 11-20 (1,824,077 pairs), truth = top-5 of weeks 21-30 (thr 0.8, our tie-break), evaluated users = ≥ 5 truth shows: ALL 4,030, SUB 1,150. Test train = weeks 11-30 (3,014,110 pairs), 1,260 template users.
- **Pool definition (deviation from the literal wording, documented):** shows with a labelled airing starting inside the target window, from all available schedule files: val 6,376, test 6,642. The literal wording ("schedule 21-30 from 05-18", "schedule 31-42") gives 6,363 / 6,636. The reason is that the files hand over per channel mid-morning, so the 11-20 file contains 283 labelled airings on 05-18 (week 21) and the 21-30 file 267 on 07-27 (week 31). All truth ids are in the pool (asserted).
- **Selection rules:** popularity baseline = best pop_* by mean(MAP ALL, MAP SUB) → **pop_recent**; personal baseline = best PF_* by the same rule (padded with pop_recent) → **PF_softfrac** (sum of frac over all training airings of the show). This is model selection on the validation set, so the chosen scores are optimistically biased. Half-lives and the 4-week recency window were tuned on a 10-week training window and may not transfer to the 20-week test training window.
- Repeats allowed; fallback = `pad_with_popularity` with the chosen popularity list restricted to the pool.
- MAP@5 = standard Kaggle definition (assumption; the competition names only "MAP@5"). CIs: 1,000 user bootstrap resamples, seed 42, 95% percentile.

### Key numbers (validation, MAP@5 [95% CI]; `baseline_results_val.csv`)
| method | ALL (n=4,030) | SUB (n=1,150) | mean padded ALL |
|---|---|---|---|
| pop_users | 0.2138 [0.2075, 0.2197] | 0.2357 [0.2248, 0.2488] | 0 |
| pop_pairs | 0.2365 [0.2303, 0.2428] | 0.2637 [0.2521, 0.2776] | 0 |
| pop_hours | 0.2281 [0.2214, 0.2347] | 0.2528 [0.2412, 0.2670] | 0 |
| **pop_recent** | 0.2398 [0.2328, 0.2466] | 0.2656 [0.2528, 0.2799] | 0 |
| PF_count_0.8 | 0.3101 [0.3029, 0.3175] | 0.3496 [0.3358, 0.3639] | 0.176 |
| PF_count_0.5 | 0.3160 [0.3090, 0.3235] | 0.3539 [0.3398, 0.3675] | 0.091 |
| **PF_softfrac** | 0.3237 [0.3168, 0.3311] | 0.3600 [0.3457, 0.3738] | 0.044 |
| PF_seconds | 0.2293 [0.2231, 0.2364] | 0.2609 [0.2484, 0.2737] | 0.044 |
| PF_decay_hl2 | 0.2867 [0.2799, 0.2939] | 0.3222 [0.3075, 0.3360] | 0.176 |
| PF_decay_hl4 | 0.3044 [0.2976, 0.3122] | 0.3440 [0.3305, 0.3586] | 0.176 |
| PF_decay_hl8 | 0.3086 [0.3014, 0.3163] | 0.3502 [0.3361, 0.3649] | 0.176 |

- Paired difference PF_softfrac − pop_recent: ALL +0.0839 [0.0770, 0.0900], SUB +0.0944 [0.0821, 0.1068]. PF_softfrac − PF_count_0.8: ALL +0.0136 [0.0112, 0.0162], SUB +0.0104 [0.0051, 0.0152] (`baseline_paired_diffs_val.csv`).
- Runtime per variant 0.4-0.8 s (popularity), 3.5-4.6 s (PF) (`baseline_runtimes_val.csv`).
- **Ceiling** (`ceiling_analysis_val.csv`): truth pairs in the user's training history with any watch: ALL 65.32% / SUB 71.03%; with a qualifying airing: 50.82% / 56.64%. A perfect history-only ranker would therefore reach MAP@5 0.6532 / 0.7103 (any watch) and 0.5082 / 0.5664 (qualifying); with all hits ranked first, AP = hits/5, so ceiling = share. Truth shows absent from the 11-20 schedule: 7.43% / 6.26%.
- **Robustness** (`robustness_target_windows_val.csv`), pop_recent / PF_softfrac:
  - main 21-30: ALL 0.2398 / 0.3237, SUB 0.2656 / 0.3600.
  - clean 21-24 + 28-30 (n=3,925 / 1,128): ALL 0.2307 / 0.3148, SUB 0.2545 / 0.3486.
  - weeks 21-25 (n=2,558 / 775): ALL 0.2253 / 0.3234, SUB 0.2346 / 0.3507.
  - weeks 26-30 (n=3,663 / 1,078): ALL 0.1974 / 0.2676, SUB 0.2185 / 0.2958.
  - The ranking PF > pop holds in every window (CIs do not overlap).
- **Activity buckets** (PF_softfrac vs pop_recent, ALL): 0-4 shows (n=218) 0.1170 vs 0.1240; 5-9 (383) 0.1980 vs 0.1480; 10-19 (1,104) 0.2793 vs 0.1986; 20-49 (1,894) 0.3744 vs 0.2731; 50+ (431) 0.4305 vs 0.3391.
- **Hit ranks** (ALL): PF_softfrac hit rate by rank 0.655 / 0.489 / 0.324 / 0.247 / 0.181 (1.90 hits per user); pop_recent 0.498 / 0.420 / 0.171 / 0.222 / 0.187 (1.50).
- **Per-user AP**: share AP = 0: PF 14.86% (ALL) / 11.04% (SUB), pop 25.93% / 22.52%; share AP = 1: PF 1.19% / 1.13%.
- **Hand MAP check** (`verify_map_hand.txt`): SUB user 57357915868236403, PF_softfrac hits at ranks 1, 2, 4 → (1/1 + 2/2 + 3/4)/5 = 0.55 = ap_at_k. MATCH.
- **Example users** (`example_users.json`, `example_users_profiles.csv`): eligible SUB users n = 1,112, p25 = 17, p90 = 55 distinct qualifying training shows. Light = 385172162373391221 (17 shows; truth in history 4/5 any watch, 3/5 qualifying; AP pop 0.600, PF 0.353). Heavy = 5672174985970291620 (55 shows; 4/5, 4/5; AP pop 0.483, PF 0.550).
- **Test submissions** (`outputs/submissions/`): sub01_popularity.csv (pop_recent on weeks 27-30: 2400480 240081 2400508 2400467 20088 for everyone) and sub02_personal_baseline.csv (PF_softfrac on weeks 11-30). Both pass the validator (1,260 rows, template order, 5 distinct ints, no zeros, all in the test pool, exact header). PF padding: 3 users received padded ids, 2 of them fully padded; 4 users have no qualifying training show (a user can have soft history without a qualifying airing). Not uploaded.
- **Reproducibility**: two full runs, all 12 result files and both submissions byte-identical (`step3_reproducibility.csv`). Runtime 106.1 s / 108.1 s, peak memory 1,314 / 1,235 MB.

### Bugs, failure cases and fixes
1. **Pool definition per the literal spec would have missed airings**: the 11-20 file's 05-18 morning airings belong to week 21. Fixed by defining the pool on airing time across all available files (see Decisions).
2. **Runtime in a hashed result file** would make the reproducibility check fail spuriously; moved runtimes to `baseline_runtimes_val.csv` (not hashed).
3. **Big-int ids**: test users are converted with `np.array([int(s) ...], dtype=np.uint64)` plus a round-trip assert (str → uint64 → str) to avoid the float64 bug from Step 2.
4. Plot fixes: fig 14 legend covered a bar (wider x-limit); fig 18 legend overlapped the bars (y-limit 1); fig 19 x-labels collided (shorter labels).

## Step 4: Method A, Personalized PageRank on a show-show co-view graph (2026-10-30)

### What was run
1. `python -m src.verify_ppr_toy` -> `verify_ppr_toy.txt`.
2. `python -m src.run_step4` twice from scratch (stdout `run_step4_stdout_run1.txt`, `run_step4_stdout.txt`; summaries `step4_summary_run1.json`, `step4_summary.json`).
Shared protocol only (`get_split`, `pad_with_popularity`, `src.evaluate`, `write_submission`); padding list = pop_recent. The 31-42 schedule was used only for the test pool and metadata.

### Design decisions
- **Nodes** = every show in the split's training interactions (val 2,973, test 4,363), incl. non-pool shows; only pool shows are recommended.
- **Edges** from B01 = (B > 0) of the users x shows weight matrix (raw co-count / cosine / weighted cosine, diagonal removed). Note: with seed weight count08, B > 0 only for qualifying airings, so that graph is sparser (608,915 vs 3,624,267 undirected edges in val) and has 474 isolated nodes.
- **Dense vs sparse**: the softfrac graphs are 82.0% dense (val: 3,624,267 of 4,417,878 possible edges, median degree 2,661 of 2,972 possible). A dense float64 matrix is 71 MB (val) / 152 MB (test), far below 1 GB, and dense BLAS products are faster than sparse at this density, so T is dense. scipy 1.16.1 (sparse B and B^T B) was added to requirements.txt.
- **Isolated nodes**: self-loop T_ii = 1 so every row sums to 1 (0 isolated nodes in the selected graph; 474 in the count08 graphs).
- **PPR**: batched power iteration (1,000 users per batch), R_0 = P, R_{t+1} = (1-d) P + d R_t T, up to 50 iterations, stop when max_u L1 change < 1e-6. Users without seeds get the popularity fallback.
- **Score** s = R / max(deg, 1)^beta; candidates = pool shows with s > 0, sorted by s desc then id asc (stable sort over id-sorted nodes).
- **new_only mode**: shows with any training watch seconds masked, padded with pop_recent minus the user's history.
- **Precision choice**: the grid ran in float32 (one 1,000 x 2,973 product took 0.79 s in float64 vs 0.26 s in float32), everything else in float64. The selected config gives exactly the same MAP@5 in both (ALL 0.325645988420182, SUB 0.363956521739130).

### Grid (`ppr_grid_val.csv`, runtime columns included)
3 edges x 4 d x 2 seed weights x 2 beta = 48 configs, all converged (max 23 iterations; softfrac graphs 5-12 iterations, because the near-complete graph mixes almost immediately). MAP@5 ALL ranged 0.2873-0.3256. Best count08 config: raw, d 0.7, beta 0 (ALL 0.3194, SUB 0.3540). beta 0.5 hurts count08 configs by up to 0.029 ALL. Softfrac configs are all within 0.3232-0.3256.
**Selection rule**: best mean of ALL and SUB MAP@5; ties -> faster -> simpler. **Selected: edge raw, seed softfrac, d 0.85, beta 0.5.** Selection on validation is optimistic, and d/beta tuned on the 10-week graph (2,973 nodes) may transfer imperfectly to the 20-week test graph (4,363 nodes, 7,977,182 edges).

### Key numbers (validation, selected PPR)
- MAP@5: ALL 0.3256 [0.3188, 0.3331], SUB 0.3640 [0.3502, 0.3781]; 7 iterations per batch; mean padded 0.036.
- Paired vs PF_softfrac: ALL +0.0020 [0.0006, 0.0032], SUB +0.0040 [0.0011, 0.0067]. Vs pop_recent: ALL +0.0858 [0.0793, 0.0918], SUB +0.0984 [0.0868, 0.1101] (`ppr_vs_baselines_val.csv`).
- **Familiar vs new** (ALL, `ppr_familiar_vs_new_val.csv`): familiar share of recommended items: PPR 99.09%, PF 99.12%, pop 84.63%. Precision familiar / new: PPR 0.3823 / 0.2350; PF 0.3805 / 0.2416; pop 0.3430 / 0.0607. New items per list: PPR 0.045, PF 0.044, pop 0.769. Share of hits that are new items: PPR 0.56%, PF 0.56%, pop 3.11%.
- **new_only** (`ppr_new_only_val.csv`): truth_new = truth minus any-watch training history; evaluated users = those with >= 1 truth_new show (ALL 2,918, SUB 766); MAP@5 against truth_new uses min(5, |truth_new|) as the normaliser; recall@5 = |top-5 ∩ truth_new| / |truth_new|. PPR_new_only MAP 0.0284 [0.0244, 0.0324], recall 0.0456; pop_recent_new_only MAP 0.0447 [0.0397, 0.0501], recall 0.0720. Paired PPR - pop: ALL -0.0162 [-0.0210, -0.0122], SUB -0.0170 [-0.0260, -0.0082]. **PPR discovers new shows worse than popularity.**
- **Activity buckets** (ALL, PPR / PF / pop): 0-4 0.1214 / 0.1170 / 0.1240; 5-9 0.2001 / 0.1980 / 0.1480; 10-19 0.2816 / 0.2793 / 0.1986; 20-49 0.3758 / 0.3744 / 0.2731; 50+ 0.4330 / 0.4305 / 0.3391.
- **Robustness** (ALL, PPR / PF / pop): main 0.3256 / 0.3237 / 0.2398; clean 0.3165 / 0.3148 / 0.2307; weeks 21-25 0.3230 / 0.3234 / 0.2253; weeks 26-30 0.2692 / 0.2676 / 0.1974.
- **Hit rate by rank** (ALL): PPR 0.662 / 0.485 / 0.332 / 0.243 / 0.182. **Per-user AP**: share AP = 0 is 14.32% (PF 14.86%, pop 25.93%).
- **Coverage/diversity** (`ppr_coverage_diversity_val.csv`): distinct shows recommended PPR 513 (8.05% of the 6,376-show pool), PF 675 (10.59%), pop 5. Lists with >= 3 categories: PPR 62.95%, PF 63.47%, pop 0%. Category mix PPR vs truth: Info 44.74% vs 28.06%, Films 2.52% vs 19.51%, Kids 14.67% vs 18.36% (`ppr_category_mix_val.csv`).
- **Cost** (`ppr_cost.csv`, run 2): val graph build 5.1 s, PPR for 4,030 users 33.9 s (run 1: 3.5 s / 18.8 s); test graph build 7.1 s, PPR for 1,260 users 5.2 s. Peak memory for the whole step 1,228 MB (run 1) / 1,147 MB (run 2).

### Verification
- **B1 toy** (`verify_ppr_toy.txt`): 3 users, 4 shows; after 2 iterations R = [23/36, 7/36, 5/36, 1/36] by hand-style fractions; vectorised max abs diff 2.8e-17; power iteration converges (33 iterations) to the closed form (1-d) p (I - dT)^-1 = [0.6, 0.225, 0.15, 0.025] within 1.1e-16.
- **B2 naive** (`verify_ppr_naive.csv`): 10 random val users (seed 42, uint64 sampling, ids asserted to exist); pure-Python list power iteration over all edges: max abs score difference 2.1e-16 at equal iteration count (vs the batched main run, which stops on the batch-wide criterion: <= 2.2e-10). All 10 top-5 lists identical.
- **B3 sanity**: T row sums in [1 - 4.3e-14, 1 + 4.0e-14]; R row sums in [1 - 4e-16, 1 + 4e-16]. At d = 0.01 (raw, softfrac, beta 0) the top-5 equals PF_softfrac's for 97.05% of users (97.24% among users with >= 5 pool shows in history), above the 95% target.
- **B4 leakage**: val graph airing_start 2020-03-09 01:16 to 2020-05-17 23:55 (< 05-18 asserted); test graph up to 2020-07-26 23:55 (< 07-27 asserted).
- **B5 determinism**: selected config run twice, recommendation hashes identical (03b4a49c...193ab6).
- **Step reproducibility** (`step4_reproducibility.csv`): two full from-scratch runs; all 13 result files (timing columns excluded from the hash) and sub03_ppr.csv identical. Runtime 864.1 s / 1,412.5 s (the machine was slower during run 2: grid 523.6 s vs 857.5 s).
- Baselines recomputed inside the step reproduce Step 3's MAP@5 exactly (asserted).

### Example users (`example_users_ppr.csv`)
- Light 385172162373391221: PPR top-5 = PF top-5 as a set (overlap 5); order differs (ТСН first instead of the serial 10003159). Each recommended show's score comes 75-93% from the same show as a seed (linear decomposition; parts sum to the total within 3.5e-18). PPR new_only top-5: entertainment shows, 0 hits.
- Heavy 5672174985970291620: overlap 4 with PF; PPR swaps PF's 'Один за всех' for Факты (not a hit). Self-seed shares 57-85%; "all other seeds" contribute 9-24%.

### Test file
`outputs/submissions/sub03_ppr.csv` (selected config retrained on weeks 11-30): validator passed; 2 users padded (both without any seed); 7 / 8 iterations. Not uploaded.

### Bugs, surprises and fixes
1. **The graph is nearly complete** (82% density), so the random walk mixes in about 7 steps and its stationary part is almost the same for every user. PPR therefore behaves like "own history + a global co-view popularity term". That explains why it beats PF only by 0.002-0.004 and why its new_only lists are worse than plain popularity. This is a substantive finding for the report, not a bug.
2. **Console encoding**: printing Cyrillic titles from the stdout log failed under cp1252 again; read with UTF-8.
3. **Reproducibility hashing**: the grid CSV must contain runtime (as requested), so the content hash drops timing columns before hashing.
4. Plot fixes: fig 22 legend overlapped bars (moved, y-limit raised); fig 20 label colour threshold changed to the midpoint of the colour range.

## Step 5: Method B, content-based filtering; schedule-exposure diagnostic (2026-10-30)

### What was run
1. `python -m src.verify_cbf_toy` -> `verify_cbf_toy.txt`.
2. `python -m src.run_step5`: a failed first attempt (tie-break bug, log kept as `run_step5_stdout_failed_tiebreak.txt`), then two full runs after the fix (`run_step5_stdout_run1.txt`, `run_step5_stdout.txt`; summaries `step5_summary_run1.json`, `step5_summary.json`).
Shared protocol only; padding = pop_recent; the 31-42 schedule was used only for the test pool, metadata and test A_target.

### Design decisions (src/cbf.py)
- Item vectors for all 8,234 shows in the val metadata (9,671 in test). Blocks, each L2-normalised, times a block weight, concatenated and L2-normalised:
  - category: one-hot.
  - genres: multi-hot with **position weights 1.0 / 0.7 / 0.5** (genre_1 is the primary genre).
  - year: decade one-hot from the first 4 digits (e.g. '2006 - ???' -> 2000s).
  - director, actors: comma tokens, lowercased, binary tf × idf, idf = ln((1+N)/(1+df)) + 1 over all shows.
  - title: lowercase words of ≥ 3 letters, digits dropped, episode-noise words (сезон, серия, выпуск, часть, эпизод, ...) removed; 1-2 letter fragments such as 'с.', 'эп', 'т/с' fall out by length.
- 0 shows have an all-zero vector. Max |norm − 1| = 3.3e-16.
- Profile = Σ w_us · x_s over training shows (w = softfrac or count08, optional 4-week half-life decay from the end of training), L2-normalised.
- Score = cosine over the pool. **Tie-break: scores rounded to 12 decimals, then pop_recent rank asc, then tv_show_id asc** (the pop_recent ranking is itself score desc, id asc, so one stable sort over pop-ordered columns implements both).
- Ranking uses a partial sort (`top_k_stable`), asserted identical to a full stable argsort (about 5× faster).
- cbf_pop = (1 − lam)·cosine + lam·pop_norm is **CBF with a popularity prior (a blend, not pure CBF)**.

### Grid (`cbf_grid_val.csv`, 96 configs + 6 block-weight configs)
- F1-F4 × {softfrac, count08} × decay {none, 4 wk} × {pure, pop lam 0.05 / 0.1 / 0.2 / 0.35 / 0.5}.
- Pure CBF ALL ranges 0.0487 (F2 softfrac, 4wk decay) to 0.1779 (F4 count08); adding the year block hurts (F1 0.1376 -> F2 0.0925 for count08), and title words help most.
- Block-weight grid on the best configs: category 0.5 helps cbf_pure (0.1779 -> 0.1815); it does not beat the default for cbf_pop (avg 0.29465 vs 0.29472).
- **Selection rule** (applied separately per variant): best mean of ALL and SUB MAP@5; ties -> faster -> simpler.
  - Selected **cbf_pure**: F4, count08, no decay, category weight 0.5 (genres 1.0).
  - Selected **cbf_pop**: F4, count08, 4-week decay, lam 0.35, default weights.
- The selection is optimistic: best-of-102 on the validation set.

### Key numbers (validation)
- cbf_pure: ALL 0.1815 [0.1761, 0.1873], SUB 0.1992 [0.1896, 0.2091]; tie at the 5th/6th boundary for 11.36% of users; mean padded 0.073.
- cbf_pop: ALL 0.2783 [0.2717, 0.2849], SUB 0.3111 [0.2997, 0.3242]; ties 0.20%.
- Paired differences (ALL / SUB):
  - cbf_pure − pop_recent: −0.0583 / −0.0663; − PF_softfrac −0.1421 / −0.1608; − PPR −0.1441 / −0.1647.
  - cbf_pop − pop_recent: +0.0385 [0.0340, 0.0431] / +0.0456; − PF −0.0453 / −0.0489; − PPR −0.0473 / −0.0528.
  - cbf_pop − cbf_pure: +0.0968 / +0.1119 (`cbf_vs_baselines_val.csv`).
- **Familiar vs new** (ALL; `cbf_familiar_vs_new_val.csv`):

  | method | familiar share | precision familiar | precision new | new items per list | share of hits that are new |
  |---|---|---|---|---|---|
  | cbf_pure | 61.91% | 0.3396 | 0.0122 | 1.90 | 2.17% |
  | cbf_pop | 86.42% | 0.3781 | 0.0442 | 0.68 | 1.80% |

  Compare PF (99.12%, 0.3805 / 0.2416) and pop (84.63%, 0.3430 / 0.0607).
- **new_only** (truth_new = truth minus any-watch history; 2,918 ALL / 766 SUB users), MAP@5 vs truth_new (ALL):

  | method | MAP@5 | recall@5 |
  |---|---|---|
  | cbf_pure_new_only | 0.0089 | 0.0146 |
  | cbf_pop_new_only | 0.0183 | 0.0298 |
  | PPR_new_only | 0.0284 (reproduces Step 4, asserted) | |
  | pop_recent_new_only | 0.0447 (reproduces Step 4, asserted) | |

  Paired cbf_pure_new_only − pop_new_only −0.0358 [−0.0409, −0.0308]. **CBF discovers unseen truth shows worse than popularity.**
- **Cold items** (`cbf_cold_items_val.csv`): 2,129 truth pairs (ALL; 530 SUB) involve a show nobody watched in training; 1,497 of them are not even in the 11-20 schedule. Recall of these pairs: cbf_pure 0.14%, cbf_pop 0.09%; every history-based method 0% (by construction).
- **Activity buckets** (ALL; PF / cbf_pure / cbf_pop / PF_exposure): 0-4 0.1170 / 0.0955 / 0.1135 / 0.1266; 5-9 0.1980 / 0.1350 / 0.1687 / 0.2094; 10-19 0.2793 / 0.1776 / 0.2441 / 0.2941; 20-49 0.3744 / 0.2058 / 0.3195 / 0.3944; 50+ 0.4305 / 0.1696 / 0.3657 / 0.4483.
- **Robustness** (ALL; cbf_pure / cbf_pop / PF_exposure): main 0.1815 / 0.2783 / 0.3406; clean 0.1765 / 0.2691 / 0.3292; weeks 21-25 0.1790 / 0.2689 / 0.3342; weeks 26-30 0.1572 / 0.2323 / 0.2791. Method order is the same in every window.
- **Hit rate by rank** (ALL): cbf_pure 0.529 / 0.243 / 0.148 / 0.095 / 0.059; cbf_pop 0.620 / 0.434 / 0.264 / 0.211 / 0.135. **AP = 0**: cbf_pure 32.41%, cbf_pop 17.37%.
- **Coverage/diversity** (`cbf_coverage_diversity_val.csv`):
  - Distinct shows recommended: cbf_pure 850 (13.33% of the pool), cbf_pop 566, PF 675, PPR 513, PF_exposure 421.
  - Lists with ≥ 3 categories: cbf_pure 13.47%, cbf_pop 10.62% vs PF 63.47%. CBF lists are narrow in category by design.
  - Category mix (`cbf_category_mix_val.csv`): cbf_pop is 66.9% Info vs 28.1% in the truth; Films get 1.7-7.9% from every method vs 19.5% in the truth.
- **Cost** (`cbf_cost.csv`): item matrix 8,234 × 23,841 (82,089 non-zeros) val, 9,671 × 26,758 (96,300) test. Item build 0.8-2.9 s; profiles + scoring + ranking for 4,030 users 6.4-7.6 s. Peak memory 1,053 MB for the whole step.

### Schedule-exposure diagnostic (non-learned; the target-window schedule is used ONLY to count which show airs on which channel when; it is never joined to viewing, and no schedule later than the target window is used)
- Val table: 692,119 (user, history show in pool) pairs.
- A_target_us (airings of the show in weeks 21-30 on the channels where the user watched it in training), `exposure_diagnostics_val.csv`:

  | pairs | A_target_us median | A_target_us p10-p90 | ratio A_target/A_train median |
  |---|---|---|---|
  | truth shows (n = 13,161) | 344 | 40-904 | 1.016 |
  | non-truth history shows (n = 614,540) | 3 | 0-97 | 0.600 |

- Hit rate by A_target_us bin (`exposure_hitrate_by_atarget_val.csv`): 0 airings 0.05%, 8-15 0.72%, 32-63 2.93%, 64+ 11.74%.
- PF-exposure baselines (`exposure_baselines_val.csv`; **baselines, not Method B**). If A_target_us = 0 but the show airs elsewhere, exposure = 0.1 × A_target(s).

  | variant | ALL | SUB | paired vs PF (ALL) |
  |---|---|---|---|
  | gamma 0 | 0.3237 | 0.3600 | reproduces PF exactly (asserted) |
  | gamma 0.25 | **0.3406** | **0.3803** | +0.0170 [0.0148, 0.0191] (SUB +0.0203 [0.0160, 0.0248]) |
  | gamma 0.5 | 0.3397 | 0.3759 | |
  | gamma 1.0 | 0.3289 | 0.3593 | |
  | rate (q-based) | 0.2893-0.3083 | | all below PF |
  | rate (softfrac-based, best: a 0, b 0.001) | 0.3240 | | |

- Gap closed: the best variant (gamma 0.25) closes **5.15% (ALL) / 5.79% (SUB)** of the gap between PF_softfrac and the any-watch history ceiling (0.6532 / 0.7103).

### Verification
- **B1 toy** (`verify_cbf_toy.txt`): 4 shows, 1 user (A×2, B×1); exact sympy vectors and cosines (A 0.956479837892768, B 0.812043546357891, C 0.359583344287329, D 0); src.cbf differs by 1.1e-16.
- **B2 naive** (`verify_cbf_naive.csv`): 10 users × {cbf_pure, cbf_pop}. Pure-Python dict item vectors (own idf and normalisation; tokenisers shared), profiles and scores over all 6,376 pool shows: max abs diff 3.3e-16; all 20 top-5 lists identical after the fix below.
- **B3 sanity**: 23 users have a single training show; for the 20 whose show (or an identical-vector show) is in the pool, a cosine-1 show is in the top-5 in all 20 cases. Cosines in [0, 1 + 2e-16].
- **B4 leakage**: val training airings 2020-03-09 01:16 .. 2020-05-17 23:55; test up to 2020-07-26 23:55 (both asserted). Metadata comes from schedules only.
- **B5 determinism**: cbf_pure and cbf_pop run twice, recommendation hashes identical.
- **Step reproducibility** (`step5_reproducibility.csv`): two from-scratch runs; all 18 result files (timing columns excluded) and both submissions identical. Runtime 819.1 s / 1,120.0 s (grid 586.1 / 873.3 s), peak 1,053 MB both.

### Example users (`example_users_cbf.csv`)
- **Light** 385172162373391221:
  - Profile led by Info and Series categories and the genres "общество и политика" and "мелодрама"; the serial 10003159 contributes 60.7% of the profile.
  - cbf_pure top-5 (serials and multi-part films, 2 new shows) has 0 hits; cbf_pop and PF_exposure have 3 hits each.
  - Overlaps: cbf_pop vs PF 5, cbf_pure vs PF 1.
- **Heavy** 5672174985970291620:
  - cbf_pure recommends talk/news shows ('Абзац!', 'Деньги', '60 минут', ...), 1 hit (Сегодня).
  - cbf_pop and PF_exposure have 3 hits each. Overlaps: cbf_pop vs PF 3, PF vs PPR 4.
- For each recommended show, the two most similar training shows are listed (item-item cosine 0.53-0.69).

### Test files
- `sub04_cbf.csv` = cbf_pop (better than cbf_pure on val): validator passed; 4 users padded, all 4 without a profile.
- `sub05_pf_exposure.csv` = PF_soft_exposure_g0.25 (A_target from the 31-42 schedule in [2020-07-27, 2020-10-19), A_train weeks 11-30): validator passed; 3 users padded, 2 without any history show in the pool.
- Not uploaded.

### Bugs, failure cases and fixes
1. **Tie-break decided by floating-point noise (caught by the naive check).** For user 1697310425691724323, shows 12002450 and 12002526 have identical item vectors but their cosines differed by about 1e-16 (0.52497512637251020 vs ...31), so the vectorised ranking ordered them by rounding noise instead of pop rank, and the naive implementation (different summation order) ordered them the other way. The run stopped on the assertion. Fix: round scores to 12 decimals before ranking in both implementations (`cbf.ROUND`). cbf_pure ALL moved 0.1814 -> 0.1815, and the measured tie share rose from 9.28% to 11.36%, which shows how often ties had been silently broken by noise.
2. **Stale log read**: my wait loop read the previous run's log (which contained the old Traceback) before the new run truncated it, which briefly looked like the fix had failed. A direct debug showed the fix works; I then waited on the fresh log.
3. NumPy deprecation (array-to-scalar in the profile contribution) fixed with explicit indexing; plot 27 value labels moved above the CIs.

## Step 6: hybrid (candidates + learned re-ranker), rank fusion, temporal backtest (2026-10-30)

### What was run
`python -m src.run_step6`. Two crashed attempts (logs kept: `run_step6_stdout_failed_float_ids.txt`, `run_step6_stdout_failed_cache_id.txt`) were followed by a complete run 1 (`run_step6_stdout_run1.txt`, `step6_summary_run1.json`) and a from-scratch run 2 (`step6_summary.json`). New code: `src/tasks.py`, `src/hybrid_features.py`, `src/run_step6.py`. `src/protocol.py`, `baselines.py`, `ppr.py` and `cbf.py` were **not modified**. The step re-derives the Step 3-5 val numbers from its own sources and asserts them to 1e-12: pop_recent 0.2398/0.2656, PF_softfrac 0.3237/0.3600, PPR 0.3256/0.3640, cbf_pure 0.1815/0.1992, cbf_pop 0.2783/0.3111, M0 = PF_soft_exposure_g0.25 0.3406/0.3803.
LightGBM 4.7.0 was installed (no fallback needed); scikit-learn 1.6.1 and sympy 1.14.0 were added to requirements.txt.

### Tasks (`tasks_summary.csv`; leakage asserted per task)

| task | history | target | users | pool | truth pairs in history |
|---|---|---|---|---|---|
| inner | weeks 11-15 [03-09, 04-13), 925,145 rows | weeks 16-20 | 3,785 (1,086 SUB) | 4,402 | 63.31% |
| val | = get_split('val') | | 4,030 / 1,150 | 6,376 | 65.32% |
| test | weeks 11-30 | weeks 31-42 | 1,260 | 6,642 | |

- Assertions: max history airing_start < target start; schedule rows < target end; inner target end (05-18) <= val history end.
- The inner task uses only the 11-20 schedule.
- **The target-window schedule is used only to count which show airs on which channel and when. It is never joined to viewing.**

### Candidates (`hybrid_candidates_summary.csv`, `hybrid_candidates_by_source.csv`)
- Union of PF_softfrac top-30, PF_exposure top-30, PPR top-20, cbf_pop top-30, cbf_pure top-30 and pop_recent top-20, restricted to the pool. It is built as a top-50 superset with every source rank kept, so other cut-offs are row filters.
- Candidates per user (mean / median / max): inner 67.1 / 66 / 107; val 69.5 / 69 / 102; test 73.1 / 71 / 105; minimum 20 everywhere, because pop_recent always contributes.
- Candidate recall of truth pairs: inner 65.87% (history ceiling 63.31%), val 65.59% (ceiling 65.32%).
- Val truth pairs found only by one source: pop 2.52%, pfx 0.97%, cbf_pop 0.49%, ppr 0.22%, cbf 0.21%, pf 0.05%.

### Features (`src/hybrid_features.py`; 59 model features in 8 groups)
- Groups: history strength 10, recency/habit 5, exposure 12, PPR 2, CBF 5, popularity 5, user context 5, show attributes 10 (8 category one-hots, median airing duration, channels airing it in the target window).
- **Every count is divided by the window length in weeks** (history 5 / 10 / 20, target 5 / 10 / 12), and ratios use per-week rates, so the inner-trained model applies to val and test. Raw counts are kept only as `raw_*` verification columns, never as model inputs.
- PPR score is relative to the user's maximum. Within-user ranks are 1/rank, with 0 when the show is not ranked.
- Labels: binary (show in truth top-5) and graded (6 - truth rank).
- Build time: inner 57.5 s, val 89.4 s, test 79.8 s; peak memory 1,432 MB at that point.

### Verification
- **Naive features** (`verify_features_naive.csv`): 6 val users × 5 candidates (each user has an exposure-0 history show and a new show). 31 history/exposure features recomputed with pure-Python loops over raw interaction rows and raw schedule rows: max abs diff 3.55e-15.
- **Hand-worked example** (`verify_features_hand.txt`): user 8152220956056607861, show 400357 'Холостяк'; 17 interaction rows; softfrac 10.348701, 7 qualifying airings, qualifying weeks [11, 14, 16, 18, 19] -> share 0.5, C_us = {6}, A_train_us 33, A_target_us 8, ratio 0.242424. All match the vectorised values.
- **Leakage tests** (val MAP@5):

  | test | val MAP@5 |
  |---|---|
  | M3 trained on randomly permuted inner labels | 0.0535 |
  | random re-ranking inside the candidates | 0.0223 |
  | the real M3 scored against truth shuffled across val users | 0.1617 |
  | real M3 | 0.3482 |

  - No feature name contains 'truth' or 'target_watch' (asserted).
  - Moving the val history end back one week (to 2020-05-11) changes softfrac for 12.48% of the 857,090 (user, show) history pairs, and 68,156 pairs disappear.
- **Determinism**: the inner features rebuilt within the run are identical (pandas hash).
- **Step reproducibility** (`step6_reproducibility.csv`): two from-scratch runs; all 21 result files (timing columns excluded) incl. sub06/sub07 identical, and the superset feature hashes of all three tasks identical. Runtime 2,120.2 s / 2,226.7 s; peak memory 1,908 / 1,609 MB.
- **Fast evaluator**: the padded users × candidates evaluator is asserted equal to `map_at_k` for three score vectors (real, random, all-zero ties).

### Models and protocols
- **M0** = PF_soft_exposure_g0.25 (reference).
- **M1 fusion**: RRF score = Σ w_c / (k0 + rank_c) over {PF_softfrac, PF_exposure, PPR, cbf_pop, pop_recent}, with w in {0, 0.5, 1, 2}^5 (1,023 non-zero combinations) and k0 in {10, 30, 60}, plus a z-score fusion with the same weights. 4,092 configurations, **tuned on inner only** (`hybrid_m1_fusion_grid_inner.csv`).
  - Best RRF: k0 10, w = (PF 0.5, PFX 2, PPR 0, cbf_pop 0.5, pop 0), inner 0.3580.
  - Best z-score: w = (0, 2, 0, 0.5, 0), inner 0.3598. This one is chosen as M1.
- **M2**: logistic regression (L2, standardised), C in {0.1, 1}.
- **M3**: LightGBM LambdaRank with the graded label.
- **M4**: LightGBM binary objective.
- Grid for M3/M4: leaves {7, 15, 31} × lr {0.03, 0.1} × min_child_samples {20, 100}. Each fit uses early stopping (50 rounds) on a user-grouped 20% hold-out of its training data, then a refit on all of it with the best iteration. seed 42, deterministic, force_row_wise, 6 threads.
- **Hyper-parameter selection: 3-fold user-grouped CV on the inner task, never val** (`hybrid_model_selection_inner_cv.csv`).
  - Selected: M2 C = 1.0 (CV 0.3648); M3 leaves 31, lr 0.1, mcs 20 (CV 0.3709; 59 trees on full inner); M4 leaves 31, lr 0.03, mcs 20 (CV 0.3734; 365 trees).
- **P1 (primary, temporal)**: train on inner, evaluate on val.
- **P2**: 5-fold user-grouped CV within val, fixed hyper-parameters and tree counts. P2 mixes training and test users from the same time window, so **P1 is the honest temporal estimate**.
- **P3**: train on inner + val (groups = (task, user)), same hyper-parameters and tree counts, then predict test.
- **Selection rule**: the final hybrid is the best of M1-M4 by mean(ALL, SUB) under P1. It counts as better than M0 only if its paired P1 difference vs M0 has a 95% CI above 0 on ALL; otherwise M0 stays final. Choosing among models on val is selection; P1 never fits on val labels.

### Key numbers (P1, val; `hybrid_p1_results_val.csv`, `hybrid_p1_paired_diffs_val.csv`)

| model | ALL [95% CI] | SUB [95% CI] | paired vs M0, ALL | paired vs M0, SUB |
|---|---|---|---|---|
| M0 | 0.3406 | 0.3803 | | |
| M1 fusion (z-score) | 0.3416 | 0.3803 | +0.0010 [-0.0007, 0.0026] | +0.0000 [-0.0031, 0.0032] |
| M1 RRF | 0.3411 | 0.3803 | | |
| M2 logistic regression | 0.3421 | 0.3780 | +0.0015 [-0.0014, 0.0046] | -0.0022 [-0.0079, 0.0035] |
| **M3 LambdaRank** | **0.3482** [0.3412, 0.3556] | **0.3880** [0.3743, 0.4026] | **+0.0075 [0.0050, 0.0099]** | +0.0078 [0.0029, 0.0128] |
| M4 binary | 0.3479 | 0.3881 | +0.0073 [0.0045, 0.0100] | +0.0078 [0.0026, 0.0132] |

- M3 against the other methods (ALL / SUB): vs PF_softfrac +0.0245 [0.0213, 0.0274] / +0.0280; vs PPR +0.0225 / +0.0241; vs cbf_pop +0.0699 / +0.0769; vs cbf_pure +0.1667 / +0.1888; vs pop_recent +0.1084 / +0.1225.
- **Decision**: the best hybrid is M3; its CI vs M0 on ALL is above 0, so **M3 is the final strategy**.
- **P2** (`hybrid_p2_cv_val.csv`): M2 0.3482 / 0.3874, M3 0.3534 / 0.3910, M4 0.3567 / 0.3954. These are higher than P1, which is expected for within-window CV.
- **Gap to the history ceiling**: M3 closes 7.44% (ALL) / 8.00% (SUB) of the PF_softfrac -> ceiling gap; M0 closed 5.15% / 5.79%.

### Analysis
- **Ablation, drop one group** (`hybrid_ablation_val.csv`; full M3 0.3482):

  | dropped group | Δ ALL |
  |---|---|
  | exposure | -0.0081 |
  | recency/habit | -0.0019 |
  | CBF | -0.0011 |
  | show attributes | -0.0004 |
  | popularity | -0.0000 |
  | history strength | +0.0001 (other groups carry the same information) |
  | user context | +0.0008 |
  | PPR | +0.0009 |

- **Ablation, add one group to history strength only** (history-only model 0.3291):

  | added group | Δ ALL |
  |---|---|
  | exposure | +0.0177 |
  | show attributes | +0.0060 |
  | CBF | +0.0054 |
  | recency | +0.0035 |
  | user context | +0.0014 |
  | PPR | +0.0003 |
  | popularity | -0.0063 |

- **Importance** (`hybrid_feature_importance_M3.csv`):
  - Mean |SHAP| top 5: ex_a_target_glob_pw 0.358, hs_share_softfrac 0.348, pop_norm 0.201, ex_a_target_us_pw 0.199, ex_rate_ratio_us 0.168.
  - Gain top 3: hs_share_softfrac, ppr_rank_inv, ppr_score_rel.
- **Candidate size** (`hybrid_candidate_size_sensitivity.csv`), same k for every source:

  | k per source | val candidate recall | candidates per user | MAP ALL |
  |---|---|---|---|
  | 10 | 56.67% | 25.0 | 0.3471 |
  | 20 | 63.93% | 51.1 | 0.3473 |
  | 30 | 67.47% | 79.0 | 0.3475 |
  | 50 | 70.95% | 130.2 | 0.3478 |

  MAP is almost flat: extra candidates raise recall but barely reach the top-5.
- **Activity buckets** (ALL, M3 vs M0): 0-4 0.1373 vs 0.1266; 5-9 0.2175 vs 0.2094; 10-19 0.3001 vs 0.2941; 20-49 0.4025 vs 0.3944; 50+ 0.4567 vs 0.4483 (`hybrid_breakdowns_val.csv`).
- **Robustness** (ALL, M3 vs M0; `hybrid_robustness_windows_val.csv`): main 0.3482 vs 0.3406; clean 0.3365 vs 0.3292; weeks 21-25 0.3408 vs 0.3342; weeks 26-30 0.2855 vs 0.2791. M3 is ahead in every window.
- **By category of the user's top truth show** (ALL, M3 / M0): info (n = 1,956) 0.4268 / 0.4177; kids (885) 0.3705 / 0.3610; series (498) 0.2162 / 0.2110; entertainment (279) 0.2897 / 0.2827; films (254) 0.0455 / 0.0436; educational (148) 0.2475 / 0.2564 (the only drop).
- **Lists** (`hybrid_lists_analysis_val.csv`), M3 vs M0:

  | measure | M3 | M0 |
  |---|---|---|
  | familiar share | 96.27% | 99.12% |
  | precision familiar | 0.4166 | 0.3971 |
  | precision new | 0.1330 | 0.2416 |
  | new items per list | 0.19 | 0.04 |
  | share of hits that are new | 1.22% | 0.54% |
  | recall@5 of truth_new | 0.0074 | 0.0029 |
  | cold-pair recall | 0 | 0 |
  | AP = 0 share | 12.23% | 13.55% |
  | AP = 1 share | 1.64% | 1.36% |
  | hit rate rank 1-5 | 0.677 / 0.521 / 0.363 / 0.260 / 0.209 | |
  | distinct shows recommended | **163 (2.56% of the pool)** | 421 |
  | lists with ≥ 3 categories | 45.04% | 54.59% |
  | Info share | 52.77% | |
  | Films share | 0.24% | |

  Truth mix: Info 28.06%, Films 19.51%.
- **Failure analysis** (`hybrid_failure_groups_val.csv`, `hybrid_failure_cases_val.csv`): 55.56% of val users have exactly the same AP as under M0.

  | group | mean AP change | q-shows | truth new to user | news share in truth | median exposure ratio of truth history shows |
  |---|---|---|---|---|---|
  | 200 largest losses | -0.1794 | 25.0 | 23.2% | 29.3% | 1.016 |
  | 200 largest gains | +0.2121 | 26.7 | 24.0% | 35.8% | 1.021 |

  The groups differ mainly in the news share of their truth; activity and new-show share are similar.

  Three concrete loss cases (SHAP of M3):
  1. User 18133386382515269463 (no history at all, popularity-only candidates): M3 put SpongeBob (12001682) at rank 1, driven by ex_a_target_glob_pw +1.71 (613.7 airings per week globally) and pop_norm +0.56. It dropped 'Викна-Новости' (2400467), a truth show at M0 rank 3, scored -1.31 overall.
  2. User 5320532063609856651: the truth series 'CSI' (1000381; M0 rank 2) fell out, while 'Викна-Новости' (non-truth) entered at rank 5. Its contributions were ex_a_target_us_pw +0.43 (18.6 airings per week on the user's channels) and pop_norm +0.39, against CSI's 8.7 per week; CSI's hs_share_softfrac (0.083 vs 0.028) was not enough.
  3. User 4684516662594146356: truth 'Служба розыска детей' (2400340, M0 rank 3) was dropped although its exposure is high (118 per week on the user's channels). The non-truth 'Реальная мистика' (5800509) won on hs_share_softfrac +0.94 and ppr_rank_inv +0.28.
- **Cost** (`hybrid_cost.csv`, run 1):

  | stage | seconds |
  |---|---|
  | feature builds (inner / val / test) | 57.5 / 89.4 / 79.8 |
  | M1 grid | 342 |
  | M2 CV | 43 |
  | M3 CV | 364 |
  | M4 CV | 480 |
  | final fits (M2 / M3 / M4) | 10.5 / 9.0 / 26.5 |
  | val inference (M2 / M3 / M4) | 0.7 / 0.85 / 3.4 |
  | P2 | 138 |
  | ablation | 96 |
  | sensitivity | 69 |
  | P3 fits | 58 |
  | test inference | 2.4 |
  | whole step | 2,120 (about 35 min, slightly above the 30 min target) |

  Peak memory 1,908 MB. Most of the time goes to the model-selection grids; candidates were not reduced. For comparison, M0 costs only the history + exposure tables (18.6 s on val).

### Example users (`example_users_hybrid.csv`)
- **Light** 385172162373391221: M3 top-5 ТСН, Факты, Сегодня, 'Викна-Новости', serial 10003159; 3 hits; AP 0.55 vs M0 0.52. hs_share_softfrac is the largest SHAP term for every item; the serial gets ex_rate_ratio_us -0.71.
- **Heavy** 5672174985970291620: 3 hits; AP 0.60 = M0.

### Test files (P3)
- `sub06_hybrid.csv` (M3, selected by the rule) and `sub07_fusion.csv` (M1 z-score fusion); both pass the validator.
- 0 users without candidates; 2 users with popularity-only candidates; 0 padded.
- Mean overlap with sub05_pf_exposure: sub06 4.06 of 5 ids (9.21% identical lists); sub07 4.48.
- Not uploaded.

### Recommended finalists for Kaggle
1. **sub06_hybrid.csv** (M3 LambdaRank): best honest temporal estimate, val P1 0.3482 / 0.3880, +0.0075 [0.0050, 0.0099] over M0. Risk: the test history window is 20 weeks against the 5 and 10 weeks it was trained on (per-week features mitigate this but cannot test it), and it recommends only 163 distinct shows on val.
2. **sub05_pf_exposure.csv** (M0): simplest, no learned parameters beyond gamma, val 0.3406 / 0.3803. It is the safe fallback if the re-ranker does not transfer.

sub07 (fusion) is not recommended: it is not significantly better than M0 on val, and its lists are almost identical to sub05.

### Bugs, failure cases and fixes
1. **Big-int ids turned into floats, again.** `rng.permutation(list_of_python_ints)` in the shuffled-truth leakage test produced float64 ids and a KeyError. This is the same class of bug as Step 2; fixed by permuting positions. Lesson repeated: never put these ids through numpy without an explicit uint64 dtype.
2. **Cache keyed on `id()` of temporary DataFrames.** The faster evaluator cached per-frame layouts by `id(frame)`. Fold frames created inside the CV loop were garbage-collected, and their ids were reused by new frames of a different size, giving a shape-mismatch crash. Fixed by storing a reference to the frame in the cache (so the id cannot be reused) and building the fold frames once.
3. **Fusion-grid speed**: a global lexsort per configuration took 0.17 s; a padded users × candidates per-row lexsort takes 0.027 s. The grid went from more than 15 min to 342 s; the evaluator is asserted equal to `map_at_k`.
4. **Noisy LightGBM 4.7 warnings** (eval_at in params, eval_set deprecation) are filtered by message.
5. **Run 2 launch incident**: run 2 was first started as a detached shell job; the attempt to restart it as a tracked background task left two processes running at once. The tracked one was stopped and the detached run completed. Its console log was truncated by the stopped task, but the result files are unaffected.

Correction (Step 7 review): the feature count is 54, not 59; 59 is the tree count of M3.

## Step 7: final comparison, report-ready examples, repository preparation (2026-10-30)

No model was trained or re-run; `run_step3..6` were not executed. Everything was assembled from existing outputs by `src/assemble_step7.py` (seconds). The only computation was `ap_at_k` on the stored example lists.

### What was assembled, and where
- **A. Final comparison**: `final_method_comparison.csv` and fig 38 (`38_accuracy_vs_cost.png`): 10 methods, role, MAP@5 ALL/SUB with CI, protocol, paired differences vs PF_softfrac and M0, learned parameters/config, measured cost on val and test, peak memory, evaluating step.
- **B. Example users**:
  - `report_example_tables.csv` / `.md`, figs 39-40: top-5 of 7 methods, truth, training top-10.
  - `report_example_facts.md`: hits, AP@5, history counts, overlaps, PF-vs-hybrid differences with stored SHAP, selection rule.
- **C. Limitations**: `limitations_evidence.csv` / `.md`, 16 limitations with numbers and sources, plus the failure cases.
- **D. Figures and plan**: `figure_index.csv` (40 figures), `figures_contact_sheet.png`, `REPORT_PLAN.md` (5-page budget; no report text).
- **E. Kaggle and submission tooling**:
  - `kaggle_results.md` (empty cells except the known sub02 scores).
  - `src/designate_final.py`: tested on a temporary copy with sub06, output deleted; `submission/` does not exist.
  - `src/check_submissions.py` -> `submissions_check.csv`, `check_submissions_stdout.txt`: all 7 files valid; sub06 vs sub05 identical for 116 of 1,260 users; mean overlap 4.06.
- **F. Repository**:
  - README rewritten; `run_all.sh` / `run_all.ps1` (not run); requirements verified; `.gitignore` extended.
  - `src/smoke_check.py` -> `smoke_check_output.txt`: 23 headline checks and 64 file hashes pass in about 3 s.
  - One local git commit (no remote).
- **G. AI usage**: `AI_USAGE_FACTS.md`.

### Consistency checks
- 18 cross-file MAP@5 comparisons in `assemble_step7.py` agree to 1e-12 (hybrid_p1_results_val, baseline_results_val, ppr_vs_baselines_val, cbf_vs_baselines_val, exposure_baselines_val, step4/5 summaries).
- 23 headline checks in `smoke_check.py` agree.
- The example-user lists are identical across example_users_profiles / _ppr / _cbf / _hybrid (asserted). The recomputed APs match the values logged in Steps 3 and 6.
- Paired differences vs PF for pop_recent and vs M0 for PF_softfrac were obtained by flipping the sign of stored paired differences. This is exact: the same bootstrap resamples apply, so the mean and the interval simply negate and swap.

### Inconsistencies found (listed, not resolved; all are timings, none are accuracy numbers)
The cost CSVs were overwritten by the second (reproducibility) run of each step, while LOG.md quotes the first run's timings for some steps. They are two different measurements of the same work. `final_method_comparison.csv` uses the CSV values and labels them "LAST of the two reproducibility runs".
- **Step 3**: LOG says runtime per variant 0.4-0.8 s (popularity) and 3.5-4.6 s (PF). `baseline_runtimes_val.csv` now holds 0.41-0.68 s and 5.55-7.71 s.
- **Step 5**: LOG says item build 0.8-2.9 s and profile + scoring + ranking 6.4-7.6 s. `cbf_cost.csv` now holds 1.02-2.91 s and 5.67-11.18 s.
- **Step 6**: LOG cost table (run 1) vs `hybrid_cost.csv` (run 2):

  | stage | run 1 (LOG) | run 2 (CSV) |
  |---|---|---|
  | feature builds (inner / val / test) | 57.5 / 89.4 / 79.8 s | 50.4 / 57.2 / 61.4 s |
  | M1 grid | 342 s | 380.6 s |
  | M3 CV | 364 s | 343.7 s |
  | M4 CV | 480 s | 493.1 s |
  | M4 val inference | 3.4 s | 10.38 s |
  | ablation | 96 s | 172.6 s |

- **Step 4**: LOG already quotes both runs (consistent).

### Missing (not available without re-running a model)
- Paired differences vs M0 for pop_recent, PPR, cbf_pure and cbf_pop. Per-user AP vectors were never stored, so these cells read "not computed in Steps 3-6".
- The bootstrap of a resolvable MAP difference at n = 630 (Kaggle half size), for the same reason. It is stated as not quantified in `limitations_evidence.md`.
- Test-split timings per learned model (P3 fit and inference were measured for all models together).

### Repository check
- No file over 5 MB outside `data/` and `cache/`; committable tree about 5.4 MB.
- "kaggle.json" and "password" are absent. "token" appears only as the TF-IDF term in the CBF code and toy output.
- The absolute path `C:\Users\DELL` appears only inside 4 console logs (tracebacks/warnings): `run_step5_stdout_failed_tiebreak.txt`, `run_step5_stdout_run1.txt`, `run_step6_stdout_failed_cache_id.txt`, `run_step6_stdout_failed_float_ids.txt`. Not rewritten, as instructed.

### Bugs and fixes in Step 7
- Fig 38 labels of the three learned hybrids overlapped; fixed with per-label offsets.
- `int(NaN)` on the overlap rows of example_users_hybrid.csv; those rows are now filtered out.
- The run scripts first stated a total runtime that was not computed; corrected to the measured sum (67-84 min).

### Step 7 review: feature-count correction
- `len(src.hybrid_features.FEATURES)` = 54 (groups 10 + 5 + 12 + 2 + 5 + 5 + 5 + 10). "59" had been written as the feature count; 59 is the number of trees of M3.
- Changed 59 -> 54 where it meant the feature count:
  - README.md (method overview);
  - outputs/final_method_comparison.csv (M2 "54 coefficients + intercept", M3 and M4 "54 features"; "59 trees" kept);
  - src/assemble_step7.py, which now writes `len(FEATURES)` instead of a literal.
- LOG.md Step 6 text is left unchanged; a correction line was added at the end of the Step 6 section.
- Untouched: "59 trees", "59-tree re-ranker" and all other uses of 59.
- The ablation n_features (44 = 54 − 10 when dropping history strength, and every other row) is consistent with 54.
- A syntax check accidentally imported `src.assemble_step7`, which re-ran the Step 7 assembly (no model). git shows no output changed except the intended CSV edit.
- smoke_check: 23 headline checks and 64 hashes pass. check_submissions: all 7 files valid, hashes unchanged. None of the changed files is in a reproducibility hash list.
