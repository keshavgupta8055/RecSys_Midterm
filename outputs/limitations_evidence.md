# Limitations: evidence

| # | limitation | supporting number | source |
|---|---|---|---|
| 1 | Only a small share of schedule rows carries a tv_show_id | labelled share per schedule file: 10.76%, 11.31%, 11.18% | schedule_summary.csv |
| 2 | A large part of watch time falls on unlabelled airings (tv_show_id = 0) and is invisible to every method | 41.15% of merged watched seconds (all users) | watchtime_accounting.csv |
| 3 | Label recipe: the competition's tie handling is unobservable; our tie-break is count, watched seconds, tv_show_id | assumption (no number) | outputs/PROTOCOL.md, src/labels.py |
| 4 | Metric: the competition page only names MAP@5; the standard Kaggle AP@K definition is assumed | assumption; unit-tested on 9 cases | src/evaluate.py, outputs/test_evaluate_output.txt |
| 5 | Validation target window contains weeks with about half the usual viewing | qualifying pairs weeks 25/26/27: 13,732 / 9,948 / 10,933 vs week 24: 27,489 and week 28: 48,511 | weekly_qualifying.csv |
| 6 | A day is missing from the view log | days without any view row: 2020-07-03 | viewlog_rows_per_day.csv |
| 7 | Truth pairs on shows nobody watched in training cannot be recovered by history methods | 2,129 of 20,150 truth pairs (10.57%); recall of these by the final hybrid 0.0000 | cbf_cold_items_val.csv, hybrid_lists_analysis_val.csv |
| 8 | Some truth shows did not even air in the training schedule | 7.43% (ALL) / 6.26% (SUB) of truth pairs | ceiling_analysis_val.csv |
| 9 | Ceiling of any history-only ranker | MAP@5 0.6532 (ALL) / 0.7103 (SUB) | ceiling_analysis_val.csv |
| 10 | The co-view graph is nearly complete, so PPR adds little beyond the user's own history | density 0.8204; PPR - PF_softfrac +0.0020 ALL | step4_summary.json, ppr_vs_baselines_val.csv |
| 11 | Content-based filtering is weaker than popularity on unseen shows | MAP@5 vs truth_new: cbf_pure 0.0089, cbf_pop 0.0183, pop_recent 0.0447 | cbf_new_only_val.csv |
| 12 | The final re-ranker concentrates on few shows | 163 distinct shows recommended on val (M0: 421) | hybrid_lists_analysis_val.csv |
| 13 | The final re-ranker over-recommends Info and almost never recommends Films | Info 52.77% vs truth 28.06%; Films 0.24% vs truth 19.51% | hybrid_lists_analysis_val.csv |
| 14 | Selection on validation is optimistic (best of many configurations) | grid sizes: PPR 48 configs, CBF 102 configs, M1 4,092 configs (tuned on inner), LightGBM 24 configs x 3 folds (on inner); P1 never fits on val labels | ppr_grid_val.csv, cbf_grid_val.csv, hybrid_m1_fusion_grid_inner.csv, hybrid_model_selection_inner_cv.csv |
| 15 | Exposure features use the target-period schedule (public 31-42 schedule on Kaggle), counting airings only | design statement; dropping the exposure group costs the re-ranker -0.0081 MAP@5 ALL | hybrid_ablation_val.csv, outputs/LOG.md Step 5/6 |
| 16 | Kaggle public/private halves of about 630 users limit what a leaderboard difference can show | NOT quantified: per-user AP vectors were not stored in Steps 3-6, so the requested bootstrap at n = 630 is skipped (it would need re-running a model, which Step 7 forbids) | - |

## Failure cases supported by results (hybrid_failure_cases_val.csv; SHAP of M3)

- user 18133386382515269463 (AP change vs M0 -0.4467):
  - truth show dropped by hybrid: 2400467 'Викна-Новости'. (Info); M0 rank 3.0, hybrid rank nan; top SHAP: pop_norm=+0.6877 (value 0.575); ex_a_target_glob_pw=+0.5614 (value 18.6)
  - non-truth show promoted by hybrid: 12001682 М/с 'Губка Боб Квадратные Штаны'. (Kids); M0 rank nan, hybrid rank 1.0; top SHAP: ex_a_target_glob_pw=+1.7125 (value 613.7); pop_norm=+0.5591 (value 0.3603)
- user 5320532063609856651 (AP change vs M0 -0.3667):
  - truth show dropped by hybrid: 1000381 Т/с 'CSI: Место преступления', 10 сезон, 1 и 2 с. (Series); M0 rank 2.0, hybrid rank nan; top SHAP: hs_share_softfrac=+1.8975 (value 0.08341); ppr_rank_inv=+0.4798 (value 1)
  - non-truth show promoted by hybrid: 2400467 'Викна-Новости'. (Info); M0 rank nan, hybrid rank 5.0; top SHAP: hs_share_softfrac=+0.8275 (value 0.02802); ex_a_target_us_pw=+0.4331 (value 18.6)
- user 4684516662594146356 (AP change vs M0 -0.3633):
  - truth show dropped by hybrid: 2400340 Служба розыска детей. (Info); M0 rank 3.0, hybrid rank nan; top SHAP: ex_a_target_glob_pw=+0.6589 (value 126.4); ex_a_target_us_pw=+0.5584 (value 118)
  - non-truth show promoted by hybrid: 5800509 Реальная мистика. (Entertainment); M0 rank nan, hybrid rank 3.0; top SHAP: hs_share_softfrac=+0.9420 (value 0.02912); ex_a_target_us_pw=+0.3889 (value 16.8)
- The first case (user 18133386382515269463) has no training history at all: all candidates come from popularity, and the re-ranker promotes a frequently aired cartoon (ex_a_target_glob_pw) over a truth news show.