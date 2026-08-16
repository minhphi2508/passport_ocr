# Passport OCR Pipeline

End-to-end OCR pipeline for extracting structured information from passport
images, with a primary focus on ICAO TD3 passports.

The system combines passport/MRZ detection, semantic orientation correction,
perspective normalization, MRZ and VIZ OCR, ICAO checksum validation,
Date-of-Issue extraction, source-aware field fusion, and end-to-end quality
evaluation.

## Final Results

The frozen pipeline was evaluated on 1,447 passport images.

### Coverage

| Status | Result |
|---|---:|
| Complete | 1,324 / 1,447 (91.5%) |
| Partial | 122 / 1,447 (8.4%) |
| Failed | 1 / 1,447 (0.1%) |

A private manually verified ground-truth set contains 117 image samples
representing 69 passport identities.

### Ground-Truth Accuracy

| Metric | Image level | Identity level |
|---|---:|---:|
| Passport number | 98.3% | 97.1% |
| Surname | 93.0% | 90.9% |
| Given names | 89.7% | 91.3% |
| Nationality | 87.2% | 85.5% |
| Date of birth | 95.7% | 98.6% |
| Sex | 94.0% | 95.7% |
| Date of expiry | 87.2% | 89.9% |
| Date of issue | 84.6% | 85.5% |
| **All fields correct** | **76.9%** | **75.4%** |

The final held-out split, which was not used for pipeline tuning, achieved:

- **78.9% image-level all-fields accuracy**
- **80.0% identity-level all-fields accuracy**

See [FINAL_BENCHMARK.md](FINAL_BENCHMARK.md) for the complete evaluation.

---

## Extracted Fields

The final structured output contains:

```text
passport_number
surname
given_names
nationality
date_of_birth
sex
date_of_expiry
date_of_issue
```

Additional metadata includes source selection, MRZ checksum state, confidence
and quality information, consistency flags, and review indicators.

---

## Pipeline

The main processing flow is:

```text
Passport image
    |
    v
YOLO passport-page + MRZ detection
    |
    v
Semantic orientation correction
    |
    v
Safe perspective normalization
    |
    +---------------------+
    |                     |
    v                     v
MRZ crop              VIZ crop
    |                     |
    v                     v
Adaptive MRZ OCR       VIZ preprocessing
    |                     |
    v                     v
MRZ reconstruction     VIZ OCR
    |                     |
    v                     v
ICAO TD3 parsing        Field extraction
    |                     |
    v                     v
Checksum validation     Date-of-Issue extraction
    |                     |
    +----------+----------+
               |
               v
       Quality-aware fusion
               |
               v
      Consistency validation
               |
               v
        Final CSV / JSON
```

### Key Design Features

- YOLO11 passport-page and MRZ detection
- geometry-based semantic orientation for 0°, 90°, 180°, and 270° inputs
- perspective normalization with safe fallback behavior
- adaptive PaddleOCR recognition
- geometry-aware reconstruction of the two TD3 MRZ lines
- ICAO 9303 checksum validation
- conservative checksum-aware MRZ correction
- separate VIZ OCR and structured field extraction
- temporal Date-of-Issue extraction
- MRZ/VIZ field fusion using validation and source quality
- consistency checks for impossible or conflicting field combinations
- deterministic sample IDs and manifests
- resumable checkpoints
- image-level and identity-level benchmark tooling
- manual annotation and failure-audit utilities

---

## Repository Structure

```text
passport_ocr/
├── models/
│   └── passport_detector_ver3_best.pt
│
├── input_images/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── ground_truth/
│   └── README.md
│
├── scripts/
│   ├── setup_cpu.ps1
│   ├── setup_rtx5060ti.ps1
│   └── ...
│
├── src/
│   ├── pipeline.py
│   ├── pipeline_rtx5060ti.py
│   ├── process_passport_pages.py
│   ├── crop_mrz_batch.py
│   ├── ocr_mrz_batch.py
│   ├── parse_mrz_results.py
│   ├── validate_mrz_results.py
│   ├── crop_viz_batch.py
│   ├── preprocess_viz_batch.py
│   ├── ocr_viz_batch.py
│   ├── extract_viz_fields.py
│   ├── extract_date_of_issue.py
│   ├── build_final_results.py
│   └── evaluation / annotation utilities
│
├── tests/
├── FINAL_BENCHMARK.md
├── requirements.txt
├── requirements-cpu.txt
├── requirements-rtx5060ti.txt
├── requirements-dev.txt
└── README.md
```

---

## Requirements

The maintained development environment uses:

```text
Python 3.12
```

The project supports:

1. Windows CPU
2. Windows with NVIDIA GPU
3. A tested dual-environment RTX 5060 Ti configuration

The detector weights are included at:

```text
models/passport_detector_ver3_best.pt
```

---

## CPU Setup

Commands below are written for Git Bash on Windows.

Clone the repository:

```bash
git clone https://github.com/minhphi2508/passport_ocr.git
cd passport_ocr
```

Create the CPU environment:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass   -File scripts/setup_cpu.ps1
```

Activate it:

```bash
source .venv/Scripts/activate
```

Place passport images in:

```text
input_images/
```

Run a clean pipeline:

```bash
python src/pipeline.py --fresh
```

---

## RTX 5060 Ti Setup

The tested RTX 5060 Ti configuration intentionally separates PyTorch and
PaddlePaddle into two Python environments.

```text
.venv-torch
    YOLO / PyTorch GPU

.venv-paddle
    PaddleOCR / PaddlePaddle GPU
```

This avoids Windows CUDA DLL conflicts caused by loading the two GPU frameworks
inside the same environment.

Create the environments:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass   -File scripts/setup_rtx5060ti.ps1
```

Run the full GPU pipeline from Git Bash:

```bash
py -3.12 src/pipeline_rtx5060ti.py --fresh
```

The launcher automatically routes YOLO and PaddleOCR stages to their
corresponding environments.

> The RTX configuration above was tested specifically on an NVIDIA GeForce
> RTX 5060 Ti. Other GPU configurations may require dependency adjustments.

---

## Outputs

Primary output files:

```text
outputs/final_results/passport_extraction_results.csv
outputs/final_results/passport_extraction_results.json
```

Each record contains the extracted passport fields together with quality,
source, validation, and review metadata.

Generated intermediate files and passport images are intentionally excluded
from Git.

---

## Evaluation

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run tests:

```bash
python -m pytest -q
```

Current test suite:

```text
16 passed
```

If the private ground-truth annotations are available, run:

```bash
python src/benchmark_suite.py
```

The benchmark includes:

- field-level accuracy
- strict all-fields accuracy
- identity-level accuracy
- split-level metrics
- confidence calibration
- failure taxonomy
- review bundle generation

---

## Evaluation Methodology

The evaluation set uses both image-level and identity-level metrics.

Image-level evaluation treats every passport image independently.

Identity-level evaluation gives each passport identity equal weight even when
multiple image variants exist for the same passport.

`All fields correct` is intentionally strict: if one comparable field is
incorrect, the entire passport sample is counted as incorrect.

A separate held-out identity split was annotated and evaluated only after the
pipeline had been frozen. No pipeline tuning was performed after inspecting
those results.

---

## Privacy

Passport images, manually verified annotations, annotation queues, and generated
OCR outputs are not included in this repository.

The `ground_truth/` directory contains only documentation explaining the
private evaluation dataset.

This prevents passport-derived personally identifiable information from being
distributed with the source code.

---

## Known Limitations

The remaining failures are heterogeneous rather than being dominated by one
single systematic issue.

Common difficult cases include:

- poor or incomplete MRZ recognition
- non-standard or difficult VIZ layouts
- visually similar OCR characters
- incomplete MRZ line reconstruction
- ambiguous Date-of-Issue extraction
- MRZ/VIZ disagreement
- low-quality or internally inconsistent source documents

The pipeline deliberately avoids aggressive sample-specific correction rules.
Development was stopped once the major systematic issues had been addressed in
order to reduce overfitting to the evaluation set.

---

## Final Development State

The pipeline is considered frozen for the reported benchmark.

The final development cycle focused on:

- semantic orientation handling
- checksum-aware MRZ reliability
- robust Date-of-Issue extraction
- source-quality-aware fusion
- identity-safe ground-truth evaluation
- held-out evaluation
- explicit failure auditing

Further improvements should be evaluated on additional independent data rather
than by adding heuristics for individual failures in the current benchmark.
