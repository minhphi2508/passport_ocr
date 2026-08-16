# Final Benchmark

This document records the frozen evaluation results for the final passport OCR
pipeline.

The pipeline was frozen before evaluation on the final holdout split. No
pipeline tuning was performed after inspecting holdout results.

## Dataset

Full inference set:

| Metric | Value |
|---|---:|
| Passport images | 1,447 |

Ground-truth evaluation set:

| Metric | Value |
|---|---:|
| Annotated image samples | 117 |
| Passport identities | 69 |
| Annotated fields | 933 / 936 |

The ground-truth files are not included in the public repository because they
contain passport-derived PII.

## End-to-End Coverage

| Status | Count | Rate |
|---|---:|---:|
| Complete | 1,324 / 1,447 | 91.5% |
| Partial | 122 / 1,447 | 8.4% |
| Failed | 1 / 1,447 | 0.1% |

## Sample-Level Accuracy

| Field | Correct | Accuracy |
|---|---:|---:|
| Passport number | 115 / 117 | 98.3% |
| Surname | 106 / 114 | 93.0% |
| Given names | 105 / 117 | 89.7% |
| Nationality | 102 / 117 | 87.2% |
| Date of birth | 112 / 117 | 95.7% |
| Sex | 110 / 117 | 94.0% |
| Date of expiry | 102 / 117 | 87.2% |
| Date of issue | 99 / 117 | 84.6% |
| All comparable fields correct | 90 / 117 | 76.9% |

`All comparable fields correct` is a strict passport-level metric: one wrong
field causes the entire image sample to be counted as incorrect.

## Identity-Level Accuracy

Each passport identity receives equal weight regardless of the number of image
variants available for that identity.

| Field | Correct | Accuracy |
|---|---:|---:|
| Passport number | 67 / 69 | 97.1% |
| Surname | 60 / 66 | 90.9% |
| Given names | 63 / 69 | 91.3% |
| Nationality | 59 / 69 | 85.5% |
| Date of birth | 68 / 69 | 98.6% |
| Sex | 66 / 69 | 95.7% |
| Date of expiry | 62 / 69 | 89.9% |
| Date of issue | 59 / 69 | 85.5% |
| All comparable fields correct | 52 / 69 | 75.4% |

## Split Results

| Split | Unit | All-fields correct |
|---|---|---:|
| Validation | Image | 33 / 35 (94.3%) |
| Validation | Identity | 18 / 20 (90.0%) |
| Test | Image | 19 / 30 (63.3%) |
| Test | Identity | 12 / 20 (60.0%) |
| DEV2 | Image | 23 / 33 (69.7%) |
| DEV2 | Identity | 12 / 19 (63.2%) |
| HOLDOUT2 | Image | 15 / 19 (78.9%) |
| HOLDOUT2 | Identity | 8 / 10 (80.0%) |

HOLDOUT2 was kept separate during development and was evaluated only after the
pipeline had been frozen.

## Quality Calibration

| Predicted quality | Samples | All-fields correct |
|---|---:|---:|
| High confidence | 37 | 89.2% |
| Medium confidence | 18 | 83.3% |
| Review | 15 | 86.7% |
| Low confidence | 47 | 61.7% |

## Failure Audit

Across the 117 annotated image samples:

- 82 wrong field values
- 22 affected passport identities

Probable failure categories:

| Category | Count | Share |
|---|---:|---:|
| Date-of-issue extraction | 18 | 22.0% |
| VIZ field extraction | 16 | 19.5% |
| MRZ OCR or parsing | 15 | 18.3% |
| MRZ parsing | 12 | 14.6% |
| Missing fields | 12 | 14.6% |
| MRZ/VIZ fusion | 9 | 11.0% |

The remaining errors are heterogeneous difficult cases rather than one single
dominant failure mode. Further sample-specific heuristic tuning was therefore
stopped to reduce the risk of overfitting.

## Final Frozen Pipeline

The final evaluated configuration includes:

- YOLO11 passport-page and MRZ detection
- MRZ-geometry-based semantic orientation
- safe perspective normalization with fallback
- adaptive PaddleOCR MRZ recognition
- geometry-aware MRZ reconstruction
- ICAO TD3 parsing and checksum validation
- conservative checksum-aware MRZ repair
- VIZ OCR and field extraction
- Date-of-Issue extraction with temporal consistency checks
- MRZ/VIZ quality-aware fusion
- consistency checks and review flags
- image-level and identity-level evaluation

The reported benchmark corresponds to the frozen pipeline state after the
orientation, MRZ, and Date-of-Issue improvements were finalized.
