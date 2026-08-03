# Model Card

## Scope

This repository contains prototype workflows for gastric histology image
classification and CT tumor screening experiments.

## Intended Use

The code is intended for research, coursework, and engineering experimentation.
It is not intended for clinical diagnosis, treatment planning, triage, or patient
care decisions.

## Data Assumptions

- Histology images are expected to be grouped into benign/malignant folders or
  provided through a manifest with `image_path,label` columns.
- CT PNG images are expected to be grouped into normal/tumor folders or provided
  through a manifest with `image_path,label` columns.
- DICOM conversion preserves non-patient study/series metadata for traceability,
  but downstream users must still manage privacy and compliance requirements.

## Metrics

Training scripts report accuracy, precision, recall, F1, ROC AUC, threshold
sweeps, calibration tables, and confusion matrices where labels are available.

## Limitations

- Performance has not been validated on external clinical cohorts.
- SimpleCNN is a baseline model and should not be treated as a robust CT model.
- CT tumor-area estimates are threshold-based approximations, not medical
  segmentations.
- Dataset shift, scanner differences, stain variation, and class imbalance can
  substantially change results.

## Clinical Safety Notice

This project is research/prototype software only. Do not use outputs as medical
advice or as a substitute for review by qualified clinicians.
