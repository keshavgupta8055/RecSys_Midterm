# Example-user facts (validation, from existing files)

All numbers below come from outputs/example_users_*.csv (Steps 3-6) and are recomputed only with ap_at_k on those stored lists. **Two users are an illustration, not evidence: none of the per-user differences below is statistically significant; the significance statements in the report rest on the 4,030 / 1,150 validation users (final_method_comparison.csv).** No statement is made about user enjoyment or discovery value.

## light user 385172162373391221

Selected by the Step 3 rule: eligible = SUB val users with >= 5 truth shows and >= 5 distinct qualifying training shows (n=1112); light = closest to p25 = 17.0; heavy = closest to p90 = 55.0; ties smallest id. This user has 17 distinct qualifying training shows.

| method | hits | AP@5 (val) | recommended shows already in training history |
|---|---|---|---|
| pop_recent | 3 | 0.6000 | 5 |
| PF_softfrac | 3 | 0.3533 | 5 |
| PPR | 3 | 0.4533 | 5 |
| cbf_pure | 0 | 0.0000 | 3 |
| cbf_pop | 3 | 0.5200 | 5 |
| M0 exposure | 3 | 0.5200 | 5 |
| final hybrid M3 | 3 | 0.5500 | 5 |

Shows that differ between PF_softfrac and the final hybrid: 0 (in PF only: []; in hybrid only: []).

Overlaps (number of common ids) between method pairs (from example_users_hybrid.csv):

| pair | common ids |
|---|---|
| hybrid (M3_lambdarank) vs pop_recent | 4 |
| hybrid (M3_lambdarank) vs PF_softfrac | 5 |
| hybrid (M3_lambdarank) vs PPR | 5 |
| hybrid (M3_lambdarank) vs cbf_pure | 1 |
| hybrid (M3_lambdarank) vs cbf_pop | 5 |
| hybrid (M3_lambdarank) vs PF_soft_exposure_g0.25 (M0) | 5 |
| pop_recent vs PF_softfrac | 4 |
| pop_recent vs PPR | 4 |
| pop_recent vs cbf_pure | 0 |
| pop_recent vs cbf_pop | 4 |
| pop_recent vs PF_soft_exposure_g0.25 (M0) | 4 |
| PF_softfrac vs PPR | 5 |
| PF_softfrac vs cbf_pure | 1 |
| PF_softfrac vs cbf_pop | 5 |
| PF_softfrac vs PF_soft_exposure_g0.25 (M0) | 5 |
| PPR vs cbf_pure | 1 |
| PPR vs cbf_pop | 5 |
| PPR vs PF_soft_exposure_g0.25 (M0) | 5 |
| cbf_pure vs cbf_pop | 1 |
| cbf_pure vs PF_soft_exposure_g0.25 (M0) | 1 |
| cbf_pop vs PF_soft_exposure_g0.25 (M0) | 5 |

## heavy user 5672174985970291620

Selected by the Step 3 rule: eligible = SUB val users with >= 5 truth shows and >= 5 distinct qualifying training shows (n=1112); light = closest to p25 = 17.0; heavy = closest to p90 = 55.0; ties smallest id. This user has 55 distinct qualifying training shows.

| method | hits | AP@5 (val) | recommended shows already in training history |
|---|---|---|---|
| pop_recent | 3 | 0.4833 | 5 |
| PF_softfrac | 3 | 0.5500 | 5 |
| PPR | 3 | 0.5500 | 5 |
| cbf_pure | 1 | 0.0400 | 4 |
| cbf_pop | 3 | 0.6000 | 5 |
| M0 exposure | 3 | 0.6000 | 5 |
| final hybrid M3 | 3 | 0.6000 | 5 |

Shows that differ between PF_softfrac and the final hybrid: 1 (in PF only: [500331]; in hybrid only: [240081]).
- hybrid-only show 240081 (Факты., Info), hit = False: top-3 SHAP (M3) hs_share_softfrac=+0.7496; ex_a_target_glob_pw=+0.4562; ex_a_target_us_pw=+0.4078
- PF-only show 500331 ('Один за всех'.), hit = False: no SHAP stored (the show is not in the hybrid top-5, and Step 6 stored contributions only for hybrid top-5 items)

Overlaps (number of common ids) between method pairs (from example_users_hybrid.csv):

| pair | common ids |
|---|---|
| hybrid (M3_lambdarank) vs pop_recent | 4 |
| hybrid (M3_lambdarank) vs PF_softfrac | 4 |
| hybrid (M3_lambdarank) vs PPR | 5 |
| hybrid (M3_lambdarank) vs cbf_pure | 1 |
| hybrid (M3_lambdarank) vs cbf_pop | 4 |
| hybrid (M3_lambdarank) vs PF_soft_exposure_g0.25 (M0) | 4 |
| pop_recent vs PF_softfrac | 3 |
| pop_recent vs PPR | 4 |
| pop_recent vs cbf_pure | 1 |
| pop_recent vs cbf_pop | 4 |
| pop_recent vs PF_soft_exposure_g0.25 (M0) | 3 |
| PF_softfrac vs PPR | 4 |
| PF_softfrac vs cbf_pure | 1 |
| PF_softfrac vs cbf_pop | 3 |
| PF_softfrac vs PF_soft_exposure_g0.25 (M0) | 4 |
| PPR vs cbf_pure | 1 |
| PPR vs cbf_pop | 4 |
| PPR vs PF_soft_exposure_g0.25 (M0) | 4 |
| cbf_pure vs cbf_pop | 2 |
| cbf_pure vs PF_soft_exposure_g0.25 (M0) | 1 |
| cbf_pop vs PF_soft_exposure_g0.25 (M0) | 3 |
