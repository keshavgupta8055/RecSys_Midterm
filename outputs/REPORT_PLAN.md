# Report plan (5 pages excluding references): structure and placement only, no report text

Figures and tables refer to `outputs/figure_index.csv` and the files named below. Everything not placed here is linked from the GitHub repository (LOG.md, PROTOCOL.md, all CSVs, all 40 figures).

| section | budget | content to place | figures / tables |
|---|---|---|---|
| Title block | 2 lines | title, author, course, date, repository link | - |
| Abstract (4 sentences, one number) | 0.15 p | task; approach (baselines, Method A PPR, Method B CBF, hybrid re-ranker); main validation number (M3 0.3482 MAP@5 vs M0 0.3406); Kaggle result once known | - |
| 1 Introduction (with a concrete example) | 0.4 p | the TV-viewing recommendation task, MAP@5, why repeat viewing matters; concrete example = the light user (17 training shows, truth = 3 news shows + 2 films) | inline numbers from `report_example_facts.md` |
| 2 Related work (3-7 numbered references, 1-3 alternatives considered) | 0.35 p | PPR / random walks on co-view graphs; content-based TF-IDF profiles; learning-to-rank (LambdaRank / LightGBM); rank fusion (RRF). Alternatives considered: matrix factorisation (not used: repeat viewing dominates), association rules (not used), sequence models (out of scope) | - |
| 3 Data and matching rule | 0.45 p | view log + schedules; channel + interval overlap with merged views; view end = start + duration (DST); frac ≥ 0.8; tv_show_id 0 handling; 41.2% of watch time on unlabelled airings | Fig 13 (small) or the watchtime table |
| 4 Method (tools, versions, key design decisions) | 0.8 p | protocol (train 11-20 / val 21-30 / inner 11-15 -> 16-20; ALL / SUB users; pool; repeats; padding); baselines; Method A PPR (graph, d, beta); Method B CBF (blocks, profile, tie rule); exposure feature (schedule-only); hybrid candidates (6 sources) + LambdaRank; Python 3.13.1, pandas 2.3.1, numpy 2.3.2, scipy 1.16.1, scikit-learn 1.6.1, LightGBM 4.7.0; hardware from `environment.txt` | one compact table of design decisions (from PROTOCOL.md) |
| 5 Experiments: setup, results, baseline comparison, verification | 1.3 p | **main results table** (`final_method_comparison.csv`: MAP@5 ALL/SUB + CI, paired diff vs PF and M0, role); **Fig 32** (all methods with ceiling) or **Fig 38** (accuracy vs cost); verification paragraph (toy examples, naive cross-checks, brute-force join, leakage tests, reproducibility hashes); example-user table | Table 1 = final_method_comparison (trimmed); Fig 32 *or* 38; Fig 39 (light user) |
| 6 Business / algorithmic analysis | 0.6 p | familiar vs new (96-99% familiar); category mix vs truth (Info 52.8% vs 28.1%, Films 0.24% vs 19.5%); coverage (163 shows); cost vs accuracy; exposure as main signal (ablation -0.0081) | Fig 31 or 28; Fig 33 (ablation) |
| 7 Discussion (failure case, root cause, fix, what worked, what surprised, next improvement) | 0.5 p | failure case = no-history user and cartoon promotion (SHAP); root cause = global airing-rate features dominate when history is empty; fix = cap exposure for users without history / films component; surprised = dense co-view graph (82%) so PPR ≈ PF, CBF below popularity; next improvement = new-show and films component, Kaggle transfer check | Fig 37 (small) or a 3-row SHAP table from `limitations_evidence.md` |
| 8 Limitations | 0.25 p | 4-6 bullets from `limitations_evidence.md` (labelled share, unlabelled watch time, weeks 25-27 dip and the missing day, cold truth pairs 10.6%, selection optimism with grid sizes, public test schedule) | - |
| 9 AI usage disclosure | 0.15 p | written by the student from `AI_USAGE_FACTS.md` | - |
| References | not counted | 3-7 numbered references + Kaggle competition page + library citations | - |

Total ≈ 5.0 pages. If space is short: drop Fig 31/28 first, then shrink the example table to the final hybrid vs PF and M0 columns only.

**Candidate figures if space allows (in priority order):** 30 (exposure diagnostics), 36 (activity buckets), 24/29 (discovery of unseen shows), 20 (PPR grid), 26 (CBF grid), 10 (weekly dip), 18 (ceiling).
