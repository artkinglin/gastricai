# EfficientNetV2B0 + CatBoost 120px

This runbook describes the `efficientnetv2b0_120.py` workflow for gastric
histology image classification.

## Workflow

- Dataset discovery supports `Normal`/`Abnormal` and `benign`/`malignant`
  folders.
- Validation uses a stratified split to preserve class balance.
- EfficientNetV2-B0 uses ImageNet weights by default and extracts pooled deep
  feature vectors.
- CatBoost trains on those EfficientNetV2-B0 features for benign/malignant
  classification.
- Validation reports accuracy, F1, precision, recall, ROC AUC, and a confusion
  matrix.
- t-SNE visualizes the extracted feature space.
- Grad-CAM writes heatmap overlays from the EfficientNet branch.

## Run the hybrid pipeline

```powershell
pip install -r requirements.txt
python efficientnetv2b0_120.py --data-dir data/GasHisSDB/120
```

Key outputs are written to `runs/efficientnetv2b0_120`:

- `catboost_model.cbm`
- `hybrid_metrics.json`
- `train_features.npy` and `val_features.npy`
- `tsne_features.png`
- `grad_cam/*.png`

## Optional fine-tuning

For stronger class-specific Grad-CAM heatmaps, fine-tune EfficientNetV2-B0 first:

```powershell
python efficientnetv2b0_120.py --mode finetune --data-dir data/GasHisSDB/120 --epochs 25
python efficientnetv2b0_120.py --data-dir data/GasHisSDB/120 --checkpoint runs/efficientnetv2b0_120/best_model.pt
```

Use `--no-pretrained` if ImageNet weights are not available.
