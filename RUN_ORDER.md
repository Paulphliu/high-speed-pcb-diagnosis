# Execution Order

Run the scripts from the repository root in this order:

```bash
python src/generate_training_dataset.py
python src/build_training_database.py
python src/train_diagnostic_models.py
python src/generate_validation_lot.py
python src/build_validation_database.py
python src/run_lot_diagnosis.py
```

Generated datasets and model outputs are written to:

- `si_simulated_dataset_v3/`
- `si_simulated_dataset_v4_pn06_lot_aware/`

The default settings generate a relatively large simulation dataset. For an initial test, edit the `RUN_*` values near the top of the two generator scripts.
