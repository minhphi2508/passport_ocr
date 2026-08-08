# Passport OCR Pipeline

End-to-end pipeline for extracting structured information from foreign passport images, with a primary focus on ICAO TD3 passports.

The system combines a YOLO11 passport detector, safe passport-page perspective processing, PaddleOCR-based MRZ/VIZ recognition, TD3 parsing and checksum validation, and hybrid Date of Issue extraction.

The development dataset also contains specimen, synthetic, AI-generated, and non-standard passport images. For this reason, TD3 structure and checksum validation are treated as strong quality signals rather than strict requirements for rejecting an entire record.

## 1. Extracted Fields

The final output contains:

- Passport number
- Surname
- Given names
- Nationality
- Date of birth
- Sex
- Date of expiry
- Date of issue

Most identity fields are extracted from the Machine Readable Zone (MRZ).

`date_of_issue` is extracted from the Visual Inspection Zone (VIZ), because this field is not contained in the standard ICAO TD3 MRZ.

## 2. Pipeline Architecture

```text
Input passport image
        |
        v
YOLO11 Ver3 detector
(passport_page + mrz)
        |
        v
Passport page crop
        |
        v
Safe perspective transformation
+ landscape normalization
        |
        +-----------------------------+
        |                             |
        v                             v
     MRZ branch                    VIZ branch
        |                             |
        v                             v
MRZ detection/crop               VIZ crop
        |                             |
        v                             v
MRZ preprocessing               VIZ preprocessing
        |                             |
        v                             v
PaddleOCR                      PaddleOCR
        |                             |
        v                             |
TD3 parsing + validation             |
        |                             |
        +-------------+---------------+
                      |
                      v
             Date of Issue extraction
             (VIZ + MRZ-assisted logic)
                      |
                      v
             Final structured output
                  (CSV + JSON)
```

## 3. Project Structure

```text
passport_ocr/
|-- input_images/
|-- models/
|   `-- passport_detector_ver3_best.pt
|-- outputs/
|-- src/
|   |-- device_config.py
|   |-- pipeline.py
|   |-- process_passport_pages.py
|   |-- crop_mrz_batch.py
|   |-- ocr_mrz_batch.py
|   |-- parse_mrz_results.py
|   |-- validate_mrz_results.py
|   |-- td3_parser.py
|   |-- td3_validator.py
|   |-- crop_viz_batch.py
|   |-- preprocess_viz_batch.py
|   |-- ocr_viz_batch.py
|   |-- extract_date_of_issue.py
|   |-- build_final_results.py
|   `-- evaluate_final_results.py
|-- requirements.txt
|-- requirements-cpu.txt
|-- requirements-gpu.txt
|-- MENTOR1_CHECKPOINT.md
`-- README.md
```

Experimental analysis, audit, recovery, and single-image scripts are not part of the production end-to-end pipeline.

Development findings and the state of the project at the current checkpoint are documented separately in `MENTOR1_CHECKPOINT.md`.

## 4. Environment

Tested Python baseline:

```text
Python 3.12.10
```

Common dependencies are stored in `requirements.txt`:

```text
numpy==2.3.5
opencv-python==5.0.0.93
paddleocr==3.7.0
paddlex==3.7.2
ultralytics==8.4.114
```

The project supports both CPU and GPU execution.

Device selection is automatic: YOLO/PyTorch and PaddleOCR independently use GPU 0 when their installed backend supports CUDA; otherwise they fall back to CPU.

Tested GPU baseline:

```text
GPU              : NVIDIA GeForce GTX 1660 Ti
PyTorch          : 2.5.1+cu118
Torchvision      : 0.20.1+cu118
PaddlePaddle GPU : 3.2.0
PaddleOCR        : 3.7.0
PaddleX          : 3.7.2
Ultralytics      : 8.4.114
```

## 5. Setup

### 5.1 Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

The `.venv` directory is local and should not be committed to Git.

### 5.2 CPU installation

For a CPU-only environment:

```powershell
pip install -r requirements-cpu.txt
```

`requirements-cpu.txt` installs the common project dependencies together with:

```text
paddlepaddle==3.2.0
torch==2.5.1
torchvision==0.20.1
```

No source-code changes are required.

The pipeline automatically selects CPU when CUDA is unavailable.

### 5.3 GPU installation

The tested GPU environment uses PyTorch 2.5.1 with CUDA 11.8.

Install the matching PyTorch and Torchvision CUDA wheels first:

```powershell
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining GPU dependencies:

```powershell
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` installs the common dependencies together with:

```text
paddlepaddle-gpu==3.2.0
```

A different GPU/CUDA environment may require compatible PyTorch and PaddlePaddle builds.

### 5.4 Verify device selection

Run:

```powershell
python src/device_config.py
```

Example GPU output:

```text
========================================================================
DEVICE CONFIGURATION
========================================================================
YOLO / PyTorch : GPU 0 (NVIDIA GeForce GTX 1660 Ti)
PaddleOCR      : gpu:0
========================================================================
```

On a CPU-only installation, both backends should report CPU.

The first PaddleOCR run may automatically download the required official OCR models.

## 6. Detection Model

Place the trained YOLO11 Ver3 weights at:

```text
models/passport_detector_ver3_best.pt
```

Detector classes:

```text
mrz
passport_page
```

### Ver3 validation result

Dataset split:

```text
Train : 585 images
Valid : 42 images
Test  : 42 images
```

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.989 | 1.000 | 0.995 | 0.900 |
| MRZ | 0.984 | 1.000 | 0.995 | 0.845 |
| Passport page | 0.994 | 1.000 | 0.995 | 0.955 |

On the 251-image unseen detection set:

```text
Exactly 1 MRZ + 1 passport_page : 250/251
Exact detection rate            : 99.60%
```

The remaining image produced two MRZ detections and one passport-page detection.

## 7. Input

Place passport images in:

```text
input_images/
```

Supported extensions include:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

For a small end-to-end test, keep only the desired test images in this directory before starting a fresh run.

## 8. Running the Pipeline

View all stages:

```powershell
python src/pipeline.py --list-stages
```

Stages:

```text
1. Detect + Process Passport Pages
2. Crop MRZ
3. OCR MRZ
4. Parse TD3
5. Validate MRZ
6. Crop VIZ
7. Preprocess VIZ
8. OCR VIZ
9. Extract Date of Issue
10. Build Final Results
```

Run a clean end-to-end batch:

```powershell
python src/pipeline.py --fresh
```

`--fresh` removes generated outputs/checkpoints from the previous run, preventing old OCR records from being mixed with a new dataset.

Run selected stages when debugging:

```powershell
python src/pipeline.py --start-stage 8 --end-stage 10
```

## 9. Passport Page Processing

`process_passport_pages.py`:

1. Detects `passport_page` and `mrz` with YOLO11 Ver3.
2. Selects the highest-confidence passport-page detection.
3. Crops the passport page with padding.
4. Searches for a valid page quadrilateral.
5. Applies perspective transformation only when safety checks pass.
6. Falls back to the ordinary crop when perspective correction is unsafe.
7. Rotates portrait outputs to landscape.

The perspective logic is intentionally conservative because an incorrect transformation can remove or distort the MRZ.

Outputs are stored under:

```text
outputs/passport_pages_safe/
```

## 10. MRZ Branch

### 10.1 MRZ crop and preprocessing

The transformed passport page is passed through YOLO11 Ver3 again to locate the MRZ.

The selected region is cropped with padding.

The OCR stage evaluates:

```text
original
grayscale
threshold
```

### 10.2 MRZ OCR

PaddleOCR processes the available variants.

The pipeline selects the best candidate using signals including:

- number of recovered MRZ lines;
- expected TD3 line lengths;
- OCR confidence.

### 10.3 TD3 parsing

The parser extracts fields including:

```text
document_type
issuing_country
surname
given_names
passport_number
nationality
birth_date
sex
expiry_date
personal_number
check digits
```

### 10.4 Checksum validation

TD3 check digits are calculated for:

- passport number;
- date of birth;
- date of expiry;
- personal number;
- final composite field.

Checksum validation is retained as a quality signal rather than being used to automatically reject the entire record.

This behavior is intentional because the development data includes specimen, synthetic, AI-generated, and non-standard documents that may contain useful readable information despite failing strict ICAO TD3 validation.

## 11. VIZ Branch

The VIZ branch is used primarily to extract `date_of_issue`.

The VIZ region is derived from the transformed passport page while excluding the lower MRZ area.

Preprocessing produces:

```text
color
grayscale
enhanced
```

PaddleOCR is then applied to the available variants.

VIZ OCR is inherently less standardized than MRZ OCR because passport layouts, languages, fonts, security backgrounds, specimen watermarks, and date formats differ across countries and document designs.

## 12. Date of Issue Extraction

Date of Issue is not encoded in the standard TD3 MRZ, so it must be recovered from VIZ OCR.

The production extractor uses a hybrid strategy.

### 12.1 Baseline extraction

The baseline Date of Issue extractor uses:

- multilingual Date of Issue label patterns;
- recognized date candidates;
- OCR geometry;
- MRZ Date of Birth;
- MRZ Date of Expiry;
- temporal plausibility rules.

MRZ information helps reject date candidates that are actually Date of Birth or Date of Expiry.

### 12.2 Spatial rescue

A fallback spatial rescue layer is used only when the baseline extractor does not find a Date of Issue.

The rescue stage:

1. searches the `enhanced`, `color`, and `grayscale` VIZ OCR variants;
2. identifies strong Date-of-Issue label candidates;
3. searches for nearby date candidates using OCR bounding-box geometry;
4. scores vertical distance, horizontal alignment, OCR confidence, and cross-variant support;
5. rejects obvious Date of Birth, Date of Expiry, and validity labels.

The rescue layer is deliberately fallback-only.

If the baseline extractor already finds a Date of Issue, that result is preserved rather than overwritten by the spatial rescue logic.

### 12.3 Extraction philosophy

The extractor does not force a Date of Issue for every passport.

When the available evidence is insufficient, `date_of_issue` can remain empty.

This is preferred to aggressively selecting another date from the VIZ and silently introducing an incorrect field.

## 13. Final Output

Primary fields:

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

Final files:

```text
outputs/final_results/passport_extraction_results.csv
outputs/final_results/passport_extraction_results.json
```

Example:

```json
{
  "passport_number": "L898902C3",
  "surname": "ERIKSSON",
  "given_names": "ANNA MARIA",
  "nationality": "UTO",
  "date_of_birth": "1974-08-12",
  "sex": "F",
  "date_of_expiry": "2012-04-15",
  "date_of_issue": "2007-04-16"
}
```

Records may be classified as:

```text
complete
partial
failed
```

A partial record can still contain useful successfully extracted fields.

## 14. Evaluation

Run:

```powershell
python src/evaluate_final_results.py
```

Outputs:

```text
outputs/evaluation/end_to_end_summary.csv
outputs/evaluation/end_to_end_details.csv
```

### Reference 15-image smoke test

A previous small end-to-end test produced:

```text
Total images              : 15
Complete records          : 11/15 (73.3%)
Partial records           : 4/15 (26.7%)
Failed records            : 0/15 (0.0%)

MRZ parsed                : 14/15 (93.3%)
MRZ validation ran        : 14/15 (93.3%)
MRZ all main checks valid : 14/15 (93.3%)

Date of Issue available   : 12/15 (80.0%)
DOI high confidence       : 9
DOI medium confidence     : 3
DOI low confidence        : 0
```

Field availability:

| Field | Available | Rate |
|---|---:|---:|
| Passport number | 14/15 | 93.3% |
| Surname | 14/15 | 93.3% |
| Given names | 12/15 | 80.0% |
| Nationality | 14/15 | 93.3% |
| Date of birth | 14/15 | 93.3% |
| Sex | 14/15 | 93.3% |
| Date of expiry | 14/15 | 93.3% |
| Date of issue | 12/15 | 80.0% |

These figures measure **field availability / pipeline success**, not field-level accuracy against manually annotated ground truth.

They should therefore be treated as a smoke-test reference rather than a definitive benchmark.

### Development-batch interpretation

Larger development runs contain multiple transformed or degraded variants of the same passport identity.

Consequently, image-level failure counts are not equivalent to independent passport-level failure counts.

For example, a single difficult source passport can produce multiple failures after blur, lighting, perspective, compression, or other transformations are applied.

Some development images are also extremely degraded or non-standard even for human inspection.

For these reasons:

- raw image-level failure rate should not be presented as passport-level accuracy;
- failures should also be analyzed at the identity level;
- visually unusable samples should be separated from meaningful OCR failures;
- field-level accuracy claims require manually verified ground truth.

## 15. Current Development Checkpoint

The current production pipeline is considered stable enough to serve as the Mentor #1 project checkpoint.

### Date of Issue

On the current 1061-record VIZ development batch, rerunning the baseline production code produced:

```text
Baseline DOI available : 734
Patched DOI available  : 749
Spatial rescues        : 15
Lost baseline DOI      : 0
```

These values describe **Date of Issue availability on the development batch**, not Date of Issue accuracy.

The development batch contains repeated passport identities and non-standard/specimen/synthetic images, so these numbers should not be interpreted as an independent 1061-passport benchmark.

### MRZ OCR failure analysis

The analyzed MRZ OCR batch contained:

```text
Total MRZ OCR records       : 1012
Usable 2-line records       : 922
Fewer than 2 selected lines : 90
```

The 90 image-level failures were categorized as:

```text
very_little_text       : 52
ocr_empty_all_variants : 38
```

No failed record had another OCR variant that already recovered two usable MRZ lines.

The failures were concentrated in roughly ten passport identities, with multiple transformed variants of the same difficult source images.

This indicates that the image-level failure count substantially overstates the number of independent problematic passports.

### MRZ recovery experiments

Checksum-guided MRZ reconstruction and repair were investigated during development.

Aggressive recovery could create false repairs, while conservative repair recovered only a very small number of additional strong cases.

MRZ recovery was therefore **not integrated into the production pipeline** at this checkpoint.

The project currently prefers a smaller reproducible pipeline over adding complex heuristics with limited demonstrated benefit.

More detailed development notes are stored in:

```text
MENTOR1_CHECKPOINT.md
```

## 16. Current Strengths

The current pipeline demonstrates:

- strong passport-page and MRZ detection performance;
- safe passport-page perspective handling with fallback behavior;
- multi-variant MRZ OCR;
- structured TD3 parsing;
- checksum-based MRZ quality validation;
- permissive handling of specimen and non-standard documents;
- MRZ-assisted Date of Issue extraction;
- fallback spatial DOI rescue;
- CSV and JSON structured outputs;
- CPU/GPU execution support;
- resumable multi-stage pipeline execution.

## 17. Known Limitations

The system can return partial results for:

- extremely blurred or low-resolution images;
- MRZ text that is unreadable even after preprocessing;
- very small VIZ text;
- severe perspective distortion;
- security patterns or specimen watermarks covering important fields;
- unusual multilingual passport layouts;
- uncommon Date of Issue formats;
- MRZ OCR that cannot recover two usable TD3 lines;
- missing or incorrect detector predictions.

The VIZ branch is particularly sensitive to image quality and document-layout diversity.

The current development dataset also contains repeated variants of the same passport identities, so evaluation must account for identity duplication before making general performance claims.

## 18. Future Improvements

The next improvements should prioritize evaluation quality before adding more heuristics.

Recommended directions:

1. Build a manually verified field-level ground-truth benchmark.
2. Evaluate using identity-aware train/validation/test organization.
3. Separate visually unusable images from readable but difficult OCR cases.
4. Measure failure rates both per image and per passport identity.
5. Investigate identities that consistently fail despite readable MRZ text.
6. Determine whether those failures originate from:
   - detector/crop quality;
   - preprocessing/upscaling;
   - PaddleOCR recognition;
   - TD3 parsing;
   - or document non-compliance.
7. Improve multilingual VIZ OCR and uncommon Date of Issue layouts where justified by verified failure cases.
8. Add automated regression tests for known representative failures.
9. Evaluate future changes against a fixed benchmark before integrating them into production.
10. Consider API/application deployment after the extraction pipeline and evaluation protocol are sufficiently stable.

Further parser or OCR heuristics should be introduced only when they produce measurable improvements without degrading previously correct cases.

## 19. Repository and Output Policy

Generated runtime data is intentionally excluded from Git.

The repository ignores:

```text
.venv/
input_images/*
outputs/*
```

Large/intermediate OCR outputs should be backed up separately when they are needed for later analysis.

The GitHub repository is intended to contain the reproducible source code, configuration, model weights, dependency files, and project documentation rather than generated runtime artifacts.

## 20. Baseline Status

This version is treated as the current Mentor #1 project checkpoint.

The production pipeline is intentionally kept relatively conservative:

- checksum failures are recorded rather than automatically rejecting non-standard documents;
- existing Date of Issue results are preserved before fallback rescue;
- experimental MRZ repair is not included without sufficient demonstrated benefit;
- development failures are interpreted at both image and identity level.

Future changes should be evaluated against fixed, manually verified data so improvements can be measured objectively rather than inferred only from OCR coverage or visual inspection.