# EfficientNetV2B0 120px

This runbook describes the improved `efficientnetv2b0_120.py` workflow for
120px gastric histology image classification.

## What changed

- Dataset discovery supports `Normal`/`Abnormal` and `benign`/`malignant`
  folders.
- Validation uses a stratified split to preserve class balance.
- Training applies augmentation designed for 120px histology images.
- EfficientNetV2-B0 uses ImageNet weights by default and replaces only the
  classifier head.
- Cross entropy is weighted from the training labels.
- Validation reports accuracy, F1, precision, recall, and ROC AUC.
- The best checkpoint is saved by validation F1.

## Run

```powershell
pip install -r requirements.txt
python efficientnetv2b0_120.py --data-dir data/GasHisSDB/120 --epochs 25
```

Use `--no-pretrained` if ImageNet weights are not available.
