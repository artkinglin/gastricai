# Data Layout

## Histology Classification

Folder layout:

```text
data/GasHisSDB/120/Normal
data/GasHisSDB/120/Abnormal
```

Supported aliases are `benign`, `normal`, `negative`, `malignant`, `abnormal`,
`tumor`, and `positive`.

Manifest layout:

```csv
image_path,label
Normal/case_001.png,benign
Abnormal/case_002.png,malignant
```

## CT Tumor Detection

Folder layout:

```text
data/train/normal
data/train/tumor
data/test/normal
data/test/tumor
```

To split a single labeled folder automatically, set `test_dir` to `null` in a
config file and provide `validation_size`.

## DICOM Conversion

Input DICOMs can be nested by study/series:

```text
data/raw_dicom/study_001/series_001/image_001.dcm
```

Converted PNGs preserve the nested relative path under the output directory.
