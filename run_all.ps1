# PowerShell version of run_all.sh
# Full pipeline in order (about 67-84 minutes in total = sum of the measured step runtimes). Requires the Kaggle files in data/.
$ErrorActionPreference = "Stop"
pip install -r requirements.txt
python src/env_info.py
python -m src.build_cache
python -m src.audit
python -m src.join
python -m src.diagnostics_step2
python -m src.test_evaluate
python -m src.run_step3
python -m src.verify_ppr_toy
python -m src.run_step4
python -m src.verify_cbf_toy
python -m src.run_step5
python -m src.run_step6
python -m src.assemble_step7
python -m src.check_submissions
python -m src.smoke_check
