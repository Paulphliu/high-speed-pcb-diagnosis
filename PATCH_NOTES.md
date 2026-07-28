# GitHub-ready patch notes

The public filenames were aligned across all scripts.

Functional fixes:
1. `generate_validation_lot.py` now imports reusable simulation functions from `generate_training_dataset.py`.
2. `run_lot_diagnosis.py` now reads the actual lot-summary columns:
   - `pred_any_abnormal_rate`
   - `pred_stage1_out_of_spec_or_worse_rate`
3. User-facing error messages and source headers now use the public filenames.
4. The training generator author line is set to Po-Hung Liu with AI-assisted coding support disclosed.

Validation performed:
- All six Python files compile successfully.
- A reduced end-to-end smoke test completed:
  training-data generation -> training SQLite -> model training ->
  PN06 validation-lot generation -> validation SQLite -> lot diagnosis.
