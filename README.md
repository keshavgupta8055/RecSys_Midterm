# sweet.tv TV programme recommender (Kaggle, MAP@5)

University exam project. For each of 1,260 sweet.tv users, recommend the 5 TV programmes (tv_show_id) they will watch most in ISO weeks 31-42 of 2020. A programme airing counts when the user watched at least 80% of it, and the target is the user's 5 most frequently watched programmes. The score is MAP@5.

**The hidden test labels are not available.** All accuracy numbers in this repository are *validation* numbers (weeks 21-30). Test numbers come only from Kaggle (`outputs/kaggle_results.md`).

## Data (not included)
Download the competition files from Kaggle (you must accept the competition rules) and place them in `data/`:
`dataset11-30.csv` (view log, weeks 11-30), `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` (TV schedules), `submission.csv` (template).
The data audit is in `outputs/LOG.md` (Step 1).

## Method overview
- **Matching**: views are merged per (user, channel) and matched to schedule airings by channel and time overlap (view end = start + duration, because of a DST jump). frac = watched seconds / airing duration; an airing qualifies at frac ≥ 0.8. Brute-force verified with 0 mismatches on 19,767 pairs.
- **Baselines**: popularity (`pop_recent`), personal frequency (`PF_softfrac` = sum of frac per show the user watched), and an exposure-weighted PF (`M0`: PF × (1 + airings of the show in the target window on the user's channels)^0.25; schedule counts only).
- **Method A: Personalized PageRank** on a show-show co-view graph (`src/ppr.py`).
- **Method B: content-based filtering** on schedule metadata: category, genres, decade, director, actors and title TF-IDF (`src/cbf.py`). Also tested with a popularity prior (`cbf_pop`).
- **Hybrid**: candidates from six sources, 54 per-week-normalised features, LightGBM LambdaRank re-ranker trained on an earlier time window (`src/tasks.py`, `src/hybrid_features.py`, `src/run_step6.py`).

## Validation protocol
1. Train on weeks 11-20 (2020-03-09 to 2020-05-17); the target is the top-5 of weeks 21-30.
2. Evaluated users have ≥ 5 qualifying target shows: ALL = 4,030, SUB = 1,150 (users in the Kaggle template).
3. Candidate pool = labelled shows airing in the target window (6,376). Repeats are allowed; short lists are padded with pop_recent.
4. The learned hybrid is trained on an **inner** task (weeks 11-15 -> 16-20) and evaluated on val (P1), so val labels are never fitted.
5. Metric: MAP@5 (standard Kaggle definition, assumed), with 95% bootstrap CIs over users and paired bootstrap differences.
6. Hyper-parameters were chosen on validation (or inner-task CV for the hybrid), so the scores are optimistic.

Full definitions: `outputs/PROTOCOL.md`.

## Results (validation MAP@5; `outputs/final_method_comparison.csv`)
| method | role | ALL [95% CI] | SUB | vs M0 (ALL, paired) |
|---|---|---|---|---|
| pop_recent | baseline | 0.2398 [0.2328, 0.2466] | 0.2656 | |
| PF_softfrac | baseline | 0.3237 [0.3168, 0.3311] | 0.3600 | -0.0170 [-0.0191, -0.0148] |
| PPR | Method A | 0.3256 [0.3188, 0.3331] | 0.3640 | |
| cbf_pure | Method B | 0.1815 [0.1761, 0.1873] | 0.1992 | |
| cbf_pop | Method B + popularity prior | 0.2783 [0.2717, 0.2849] | 0.3111 | |
| M0 PF_soft_exposure_g0.25 | baseline | 0.3406 [0.3336, 0.3482] | 0.3803 | reference |
| M1 score fusion | non-learned hybrid | 0.3416 [0.3346, 0.3490] | 0.3803 | +0.0010 [-0.0007, 0.0026] |
| M2 logistic regression | hybrid | 0.3421 [0.3349, 0.3497] | 0.3780 | +0.0015 [-0.0014, 0.0046] |
| **M3 LambdaRank** | **final hybrid** | **0.3482 [0.3412, 0.3556]** | **0.3880** | **+0.0075 [0.0050, 0.0099]** |
| M4 LightGBM binary | hybrid | 0.3479 [0.3406, 0.3552] | 0.3881 | +0.0073 [0.0045, 0.0100] |

History-only ceiling (perfect ranking of shows the user already watched): 0.6532 / 0.7103.

**Final submission file:** `outputs/submissions/sub06_hybrid.csv` (M3, val 0.3482 / 0.3880). The fallback candidate is `sub05_pf_exposure.csv` (M0). Designate the final file with `python -m src.designate_final <name>`. Kaggle scores so far: `outputs/kaggle_results.md` (sub02 PF_softfrac: public 0.41112, private 0.39369).

## Repository layout
```
src/            all code (io_utils, join, labels, protocol, evaluate, baselines, ppr, cbf, tasks,
                hybrid_features, make_submission, run_step3..6, verify_*, assemble_step7,
                designate_final, check_submissions, smoke_check)
outputs/        LOG.md (every step, number and bug), PROTOCOL.md, HANDOFF.md, REPORT_PLAN.md,
                AI_USAGE_FACTS.md, limitations_evidence.md, report_example_*.md, all result CSVs,
                figures/ (01-40), submissions/ (sub01-sub07), step*_summary.json + *_reproducibility.csv
data/, cache/   not tracked (raw data, parquet caches)
```

## Reproduce (in this order; measured runtimes on the hardware below)
| command | what it does | measured runtime |
|---|---|---|
| `pip install -r requirements.txt` | Python 3.13.1 libraries | - |
| `python src/env_info.py` | records hardware/versions | seconds |
| `python -m src.build_cache` | typed parquet cache of the CSVs | ~11 s |
| `python -m src.audit` | Step 1 data audit | 26-31 s |
| `python -m src.join` | Step 2 view-airing join -> `cache/interactions.parquet` | 57.6-69.0 s from an empty cache |
| `python -m src.diagnostics_step2` | Step 2 diagnostics | ~35 s |
| `python -m src.run_step3` | protocol, baselines, sub01/sub02 | 106-108 s |
| `python -m src.run_step4` | Method A (PPR), sub03 | 864-1,413 s |
| `python -m src.run_step5` | Method B (CBF) + exposure, sub04/sub05 | 819-1,120 s |
| `python -m src.run_step6` | hybrid, sub06/sub07 | 2,120-2,227 s |
| `python -m src.assemble_step7` | final comparison, example tables, figure index | seconds |
| `python -m src.check_submissions` | re-validates all submission files | seconds |
| `python -m src.smoke_check` | offline check of headline numbers and file hashes (no data needed) | ~3 s |

`run_all.ps1` / `run_all.sh` list the same commands. Verification scripts (`src.verify_step2`, `src.verify_ppr_toy`, `src.verify_cbf_toy`, `src.test_evaluate`) are described in `outputs/LOG.md`.
Hardware (`outputs/environment.txt`): Windows 11, Intel 6 cores / 8 threads, 7.69 GB RAM, Python 3.13.1. Peak memory of every step was below 2 GB.

## Known limitations
See `outputs/limitations_evidence.md` (each limitation with its supporting number and source file). In short: only about 11% of schedule rows carry a show id, and 41.2% of watch time falls on unlabelled airings; 10.6% of validation truth pairs are shows nobody watched in training; the final re-ranker recommends few distinct shows and almost no films; the exposure feature uses the public test-period schedule (airing counts only, never viewing).
