# AI-usage fact sheet (facts for the student's own disclosure text; not the disclosure itself)

## Tools
- Claude Code (Anthropic) running inside the VS Code extension, working in this repository through file, shell and Python tools.
- Model as reported by the session environment: Claude Opus 5.5 (model id `claude-opus-5-5`).
- The student wrote the step-by-step instructions (Steps 1-7, each with rules, required checks and outputs); the assistant wrote and ran the code, produced the figures/CSVs and wrote LOG.md, PROTOCOL.md, HANDOFF.md and the Step 7 documents.
- No Kaggle API calls were made by the assistant; uploads are manual.

## Pipeline steps
7 steps: (1) data audit, (2) view-airing join and labels, (3) protocol, evaluator and baselines, (4) Method A PPR, (5) Method B CBF + exposure diagnostic, (6) hybrid re-ranker, (7) assembly and repository preparation.

## Bugs and wrong first attempts found by verification (from the "Bugs" sections of outputs/LOG.md)
| step | problem | how it was caught | fix |
|---|---|---|---|
| 1 | empty CSV fields loaded as "" instead of null (0% missingness reported) | explicit null vs empty-string check | `strings_can_be_null=True`, cache rebuilt |
| 1 | user ids exceed int64 | value inspection | load as uint64 |
| 1 | 117 views with duration ≠ stop − start (+3600 s) | audit check; traced to the 2020-03-29 DST jump | view end = start + duration |
| 2 | brute-force check sampled float64-rounded user ids (vacuous pass) | a picked user had 0 views | sample from the uint64 column; assert ids exist |
| 2 | reproducibility script deleted its own run-1 copy | crash in assert_frame_equal | copies moved to cache/repro/ |
| 2 | duplicate schedule key turned out to be the zero-length airing | inspection | documented, no extra rows dropped |
| 2 | pandas duplicate index after concat | ValueError | `ignore_index=True` |
| 3 | literal pool definition would miss airings at file boundaries | boundary inspection | pool by airing time across files |
| 3 | runtime column inside a hashed results file | reproducibility design review | runtimes moved to a separate CSV |
| 4 | (finding) co-view graph 82% dense, so PPR ≈ PF | graph statistics | reported as a result |
| 4 | hashing would fail on timing columns | design review | content hash excludes timing columns |
| 5 | tie-break decided by floating-point noise (~1e-16) | naive pure-Python cross-check (top-5 mismatch) | scores rounded to 12 decimals before ranking |
| 5 | stale log read made the fix look failed | direct debug of the user | waited on the fresh log |
| 6 | `rng.permutation(list of big ints)` -> float64 ids (KeyError) | crash in the shuffled-truth leakage test | permute positions instead |
| 6 | cache keyed on `id()` of temporary DataFrames (id reuse) | shape-mismatch crash | cache keeps a reference; fold frames built once |
| 6 | fusion grid too slow (> 15 min) | runtime | per-row padded lexsort (6× faster), asserted equal to map_at_k |
| 6 | run 2 launched twice (detached + tracked) | process listing | tracked copy stopped; results unaffected |
| all | plot readability problems (overlapping legends/labels, cramped bins) | every PNG opened and checked | layout fixes, logged per step |

## Verification artefacts that exist
- Toy hand examples: `outputs/verify_ppr_toy.txt` (exact fractions), `outputs/verify_cbf_toy.txt` (exact sympy arithmetic).
- Hand-worked examples: `outputs/verify_hand_examples.txt` (join), `outputs/verify_map_hand.txt` (AP@5), `outputs/verify_features_hand.txt` (hybrid features).
- Naive cross-checks: `outputs/verify_bruteforce.csv` (join, 19,767 pairs), `outputs/verify_ppr_naive.csv`, `outputs/verify_cbf_naive.csv`, `outputs/verify_features_naive.csv`.
- Unit tests: `src/test_evaluate.py` -> `outputs/test_evaluate_output.txt`.
- Reproducibility hashes: `outputs/verify_reproducibility.csv` (interactions), `outputs/step3_reproducibility.csv` ... `step6_reproducibility.csv`; `src/smoke_check.py` -> `outputs/smoke_check_output.txt`.
- Leakage tests: no-leakage asserts in `src/tasks.py` and the step scripts; permutation / random / shuffled-truth tests and the history-shift test (`outputs/step6_summary.json`, LOG Step 6).
- Submission validation: `src/make_submission.py`, `src/check_submissions.py` -> `outputs/submissions_check.csv`.

## Not independently verified
- The MAP@5 definition used by Kaggle (the standard Kaggle AP@K is assumed).
- The competition's tie-break between programmes with equal counts (unobservable; our convention is documented).
- Details of how Kaggle computes and splits the public/private leaderboard.
- LightGBM, scikit-learn, scipy, numpy and pandas internals (used as libraries; results are deterministic in this environment but not re-derived).
- Whether the 59-tree re-ranker transfers to the 20-week test history (only the Kaggle score can show this).

## Things only the student can do
- Hand-recompute one AP from `outputs/verify_map_hand.txt`.
- Check one Kaggle submission row against `outputs/submissions/sub06_hybrid.csv` (same user id, same 5 ids, same order).
- Write the Discussion in his/her own words.
- Upload the finalists, record the scores and screenshots in `outputs/kaggle_results.md`, designate the final file, and push the repository.
