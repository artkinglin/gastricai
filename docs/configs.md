# Experiment Configs

Training scripts accept `--config` with JSON or YAML files. Config values are
merged over CLI defaults, so short commands can reproduce longer experiments.

Examples:

```powershell
python efficientnetv2b0_120.py --config configs/efficientnetv2b0_120.example.json
python ct_tumor_detection.py --config configs/ct_tumor_detection.example.json
```

Use `run_name` to place outputs under a named run directory, and use
`resume_checkpoint` to initialize model weights from a previous checkpoint.
