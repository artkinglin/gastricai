# gastricai

Python tools for gastric image classification and CT/DICOM preprocessing.

## Environment

Use Python 3.11 or 3.12. The ML stack in this project depends on PyTorch, TorchVision, OpenCV, scikit-learn, and pydicom; these packages are more reliable on currently supported Python versions than on very new interpreter releases.

From PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If Python 3.12 is not installed, use Python 3.11 instead:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Verify Setup

```powershell
python -m pytest
python -m py_compile efficientnetv2b0_120.py ct_tumor_detection.py dicom_to_png.py
```

The current repository does not include training data. To train the EfficientNet workflow, place the 120px GasHisSDB images under:

```text
data/GasHisSDB/120/Normal
data/GasHisSDB/120/Abnormal
```

Then run:

```powershell
python efficientnetv2b0_120.py --data-dir data/GasHisSDB/120 --epochs 25
```

The best model checkpoint is written to `runs/efficientnetv2b0_120/best_model.pt`.

## CT Tumor Detection

The CT workflow expects PNG images grouped by class folder:

```text
data/train/normal
data/train/tumor
data/test/normal
data/test/tumor
```

Compatible folder aliases include `negative`, `benign`, `no_tumor`,
`positive`, `malignant`, and `abnormal`.

Run a short CPU smoke test:

```powershell
python ct_tumor_detection.py --device cpu --epochs 1 --max-images-per-class 8
```

Run a full training pass:

```powershell
python ct_tumor_detection.py --train-dir data/train --test-dir data/test --epochs 5 --plot
```

Outputs are written under `runs/ct_tumor_detection/`:

- `best_ct_tumor_model.pt`
- `history.json`
- `predictions.csv`
- `evaluation_histograms.png` when `--plot` is used

## DICOM Conversion

Convert DICOM files to PNG while preserving nested study/series folders:

```powershell
python dicom_to_png.py --input-dir data/raw_dicom --output-dir data/processed_png
```

The converter applies CT `RescaleSlope`/`RescaleIntercept`, uses DICOM window
center/width when available, skips invalid files with a warning, and writes
`conversion_metadata.csv` to the output directory.
