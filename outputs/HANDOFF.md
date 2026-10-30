# Handoff: sweet.tv TV recommender (MAP@5), state after Step 7

**Task.** For each of 1,260 users, predict the 5 tv_show_ids they will watch most in ISO weeks 31-42 (≥ 80% screen time per airing, counted per show). The report needs real numbers; every number lives in `outputs/LOG.md` and the CSVs. Session rules: no invented numbers, no weeks 31-42 viewing, seed 42, PNG dpi 150, CSVs under `outputs/`, append to LOG.md, and user ids stay uint64 or strings (never float; this caused bugs in Steps 2 and 6).

**Protocol** (`outputs/PROTOCOL.md`, `src/protocol.py`):
- Views are merged per (user, channel) and matched to airings by channel + time overlap; frac ≥ 0.8 qualifies; tv_show_id 0 is never an item.
- Validation: history weeks 11-20, truth = top-5 of weeks 21-30. Users: ALL 4,030, SUB 1,150. Pool 6,376 (test 6,642).
- Repeats allowed; pad with pop_recent; Kaggle MAP@5 with a 1,000× user bootstrap.
- Step 6 added an **inner** task (weeks 11-15 -> 16-20, 3,785 users) so the re-ranker can be trained before val (`src/tasks.py`).

**Validation MAP@5 (ALL / SUB).**
| method | role | ALL | SUB |
|---|---|---|---|
| pop_recent | baseline | 0.2398 | 0.2656 |
| PF_softfrac | baseline | 0.3237 | 0.3600 |
| PPR (raw graph, softfrac, d 0.85, beta 0.5) | **Method A** | 0.3256 | 0.3640 |
| cbf_pure (F4, count08) / cbf_pop (lam 0.35) | **Method B** | 0.1815 / 0.2783 | 0.1992 / 0.3111 |
| M0 = PF_soft_exposure_g0.25 (target-window airings on the user's channels) | baseline | 0.3406 | 0.3803 |
| M1 score fusion (tuned on inner) | non-learned hybrid | 0.3416 | 0.3803 |
| **M3 LightGBM LambdaRank re-ranker (trained on inner, P1)** | **final hybrid** | **0.3482** | **0.3880** |

- M3 vs M0: +0.0075 [0.0050, 0.0099] ALL, so the selection rule picks M3.
- History ceiling: 0.6532 / 0.7103. M3 closes 7.4% of the PF -> ceiling gap.

**Kaggle results:** sub02_personal_baseline private 0.39369 / public 0.41112. Nothing else has been uploaded yet; uploads are manual.

**Recommended finalists:**
1. `sub06_hybrid.csv` (M3, best val).
2. `sub05_pf_exposure.csv` (M0, simplest safe fallback).
Their lists share 4.06 of 5 ids on average. sub07_fusion is not significantly better than M0.

**Example users:** light 385172162373391221, heavy 5672174985970291620 (`outputs/example_users.json`).

**Files.**
- `src/`: io_utils, join, labels, protocol, evaluate, baselines, ppr, cbf, tasks, hybrid_features, make_submission, run_step3-6, verify_*, test_evaluate.
- `outputs/`: LOG.md (Steps 1-6 incl. every bug), PROTOCOL.md, CSVs (baseline_*, ppr_*, cbf_*, exposure_*, hybrid_*), figures 01-37, step3-6_summary.json and step*_reproducibility.csv (hashes).
- `outputs/submissions/`: sub01-sub07, all validated.
- Running `python -m src.run_step6` takes about 36 min.

**Key findings.**
- Repeat viewing dominates (96-99% of good recommendations are shows the user already watched); the co-view graph is 82% dense, so PPR ≈ PF; CBF is weak, also for unseen shows.
- Target-window exposure is the strongest added signal: dropping the exposure features costs the re-ranker 0.0081.
- The re-ranker concentrates on few shows (163 distinct on val), over-recommends Info (53% vs 28% of the truth) and almost never recommends Films (0.24% vs 19.5%). Truth shows nobody watched in training (2,129 pairs) are never recovered.

**Step 7 (done).** Assembled with no re-running:
- final_method_comparison.csv + fig 38;
- report example tables/facts (figs 39-40);
- limitations_evidence.md, figure_index.csv + contact sheet, REPORT_PLAN.md, kaggle_results.md (to fill), AI_USAGE_FACTS.md;
- tools: designate_final.py, check_submissions.py, smoke_check.py;
- README rewritten; one local git commit.

Timing values in LOG.md (run 1) differ from the cost CSVs (run 2) for Steps 3, 5 and 6; they are listed in LOG Step 7. No accuracy number differs between files.

**What remains (student).**
1. Upload sub06_hybrid and sub05_pf_exposure; fill outputs/kaggle_results.md with scores and screenshots.
2. `python -m src.designate_final <name>` for the chosen file.
3. Write the 5-page report following outputs/REPORT_PLAN.md (and the AI-usage disclosure from AI_USAGE_FACTS.md).
4. Add the remote and push; export the report.
