# Passport OCR Pipeline

End-to-end pipeline for extracting structured information from foreign
TD3 passport images.

The system combines a YOLO11 passport detector, safe passport-page
perspective processing, PaddleOCR-based MRZ/VIZ recognition, TD3 parsing
and checksum validation, and hybrid Date of Issue extraction.

## 1. Extracted Fields

The final output contains:

-   Passport number
-   Surname
-   Given names
-   Nationality
-   Date of birth
-   Sex
-   Date of expiry
-   Date of issue

Most identity fields are extracted from the Machine Readable Zone (MRZ).
`date_of_issue` is extracted from the Visual Inspection Zone (VIZ),
because this field is not contained in the standard TD3 MRZ.

## 2. Pipeline Architecture

``` text
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

``` text
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
`-- README.md
```

Experimental single-image and `analyze_*` scripts are not part of the
production end-to-end pipeline.

## 4. Environment

Tested Python baseline:

``` text
Python 3.12.10
```

Common dependencies are stored in `requirements.txt`:

``` text
numpy==2.3.5
opencv-python==5.0.0.93
paddleocr==3.7.0
paddlex==3.7.2
ultralytics==8.4.114
```

The project supports both CPU and GPU execution. Device selection is
automatic: YOLO/PyTorch and PaddleOCR independently use GPU 0 when their
installed backend supports CUDA; otherwise they fall back to CPU.

Tested GPU baseline:

``` text
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

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

The `.venv` directory is local and should not be committed to Git.

### 5.2 CPU installation

For a CPU-only environment:

``` powershell
pip install -r requirements-cpu.txt
```

`requirements-cpu.txt` installs the common project dependencies together
with:

``` text
paddlepaddle==3.2.0
torch==2.5.1
torchvision==0.20.1
```

No source-code changes are required. The pipeline automatically selects
CPU when CUDA is unavailable.

### 5.3 GPU installation

The tested GPU environment uses PyTorch 2.5.1 with CUDA 11.8. Install
the matching PyTorch and Torchvision CUDA wheels first:

``` powershell
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining GPU dependencies:

``` powershell
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` installs the common dependencies together with:

``` text
paddlepaddle-gpu==3.2.0
```

The tested GPU baseline uses CUDA-enabled PyTorch and PaddlePaddle. A
different GPU/CUDA environment may require a compatible PyTorch or
PaddlePaddle build.

### 5.4 Verify device selection

Run:

``` powershell
python src/device_config.py
```

Example GPU output:

``` text
========================================================================
DEVICE CONFIGURATION
========================================================================
YOLO / PyTorch : GPU 0 (NVIDIA GeForce GTX 1660 Ti)
PaddleOCR      : gpu:0
========================================================================
```

On a CPU-only installation, both backends should report CPU.

The first PaddleOCR run may automatically download the required official
OCR models.

## 6. Detection Model

Place the trained YOLO11 Ver3 weights at:

``` text
models/passport_detector_ver3_best.pt
```

Detector classes:

``` text
mrz
passport_page
```

### Ver3 validation result

Dataset split:

``` text
Train : 585 images
Valid : 42 images
Test  : 42 images
```

  Class             Precision   Recall   mAP50   mAP50-95
  --------------- ----------- -------- ------- ----------
  All                   0.989    1.000   0.995      0.900
  MRZ                   0.984    1.000   0.995      0.845
  Passport page         0.994    1.000   0.995      0.955

On the 251-image unseen detection set:

``` text
Exactly 1 MRZ + 1 passport_page : 250/251
Exact detection rate            : 99.60%
```

The remaining image produced two MRZ detections and one passport-page
detection.

## 7. Input

Place passport images in:

``` text
input_images/
```

Supported extensions include `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`,
`.tif`, and `.tiff`.

For a small end-to-end test, keep only the desired test images in this
directory before starting a fresh run.

## 8. Running the Pipeline

View stages:

``` powershell
python src/pipeline.py --list-stages
```

Stages:

``` text
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

``` powershell
python src/pipeline.py --fresh
```

`--fresh` removes generated outputs/checkpoints from the previous run,
preventing old OCR records from being mixed with a new dataset.

Run selected stages when debugging:

``` powershell
python src/pipeline.py --start-stage 8 --end-stage 10
```

## 9. Passport Page Processing

`process_passport_pages.py`:

1.  Detects `passport_page` and `mrz` with YOLO11 Ver3.
2.  Selects the highest-confidence passport-page detection.
3.  Crops the passport page with padding.
4.  Searches for a valid page quadrilateral.
5.  Applies perspective transformation only when safety checks pass.
6.  Falls back to the ordinary crop when perspective correction is
    unsafe.
7.  Rotates portrait outputs to landscape.

The perspective logic is intentionally conservative because an incorrect
transformation can remove or distort the MRZ.

Outputs are stored under:

``` text
outputs/passport_pages_safe/
```

## 10. MRZ Branch

### MRZ crop and preprocessing

The transformed passport page is passed through YOLO11 Ver3 again to
locate the MRZ. The selected region is cropped with padding.

The OCR stage evaluates:

``` text
original
grayscale
threshold
```

### MRZ OCR

PaddleOCR processes the available variants. The pipeline selects the
best candidate using the number of recovered lines, expected TD3 line
lengths, and OCR confidence.

### TD3 parsing

The parser extracts fields including:

``` text
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

### Checksum validation

TD3 check digits are calculated for passport number, date of birth, date
of expiry, personal number, and the final composite field.

Checksum validation is retained as a quality signal rather than being
used to automatically reject the entire record. The development data
includes specimen/synthetic documents, so validation failure is recorded
separately from extraction success.

## 11. VIZ Branch

The VIZ branch is used primarily to extract `date_of_issue`.

The VIZ region is derived from the transformed passport page while
excluding the lower MRZ area.

Preprocessing produces:

``` text
color
grayscale
enhanced
```

PaddleOCR is then applied to the variants.

VIZ OCR is less standardized than MRZ OCR because passport layouts,
languages, fonts, security backgrounds, specimen watermarks, and date
formats differ across countries.

## 12. Date of Issue Extraction

Date of Issue extraction uses a hybrid strategy rather than a single
fixed layout.

The extractor uses:

-   multilingual Date of Issue label patterns;
-   OCR geometry;
-   recognized date candidates;
-   MRZ Date of Birth;
-   MRZ Date of Expiry;
-   temporal plausibility rules.

MRZ information helps reject candidates that are actually Date of Birth
or Date of Expiry.

The current approach prioritizes avoiding clearly incorrect DOI values
over forcing a result for every image. When evidence is insufficient,
`date_of_issue` can remain empty.

## 13. Final Output

Primary fields:

``` text
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

``` text
outputs/final_results/passport_extraction_results.csv
outputs/final_results/passport_extraction_results.json
```

Example:

``` json
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

Records may be classified as `complete`, `partial`, or `failed`. A
partial record can still contain useful successfully extracted fields.

## 14. End-to-End Evaluation

Run:

``` powershell
python src/evaluate_final_results.py
```

Outputs:

``` text
outputs/evaluation/end_to_end_summary.csv
outputs/evaluation/end_to_end_details.csv
```

### Current 15-image test

``` text
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

  Field               Available    Rate
  ----------------- ----------- -------
  Passport number         14/15   93.3%
  Surname                 14/15   93.3%
  Given names             12/15   80.0%
  Nationality             14/15   93.3%
  Date of birth           14/15   93.3%
  Sex                     14/15   93.3%
  Date of expiry          14/15   93.3%
  Date of issue           12/15   80.0%

These figures measure **field availability / pipeline success**, not
field-level accuracy against manually annotated ground truth.

An accuracy claim requires a separate ground-truth dataset and direct
comparison between predicted and true values.

## 15. Current Baseline Summary

The current baseline demonstrates:

-   strong passport-page and MRZ detection performance;
-   robust TD3 parsing for successfully recognized MRZs;
-   checksum validation as an MRZ quality signal;
-   no completely failed records in the current 15-image end-to-end
    test;
-   lower robustness for VIZ/Date of Issue extraction than for
    standardized MRZ fields.

## 16. Known Limitations

The system can return partial results for:

-   extremely blurred or low-resolution images;
-   very small VIZ text;
-   severe perspective distortion;
-   security patterns or specimen watermarks covering important fields;
-   unusual multilingual passport layouts;
-   uncommon Date of Issue formats;
-   MRZ OCR that cannot recover two usable TD3 lines;
-   missing or incorrect detector predictions.

The VIZ branch is particularly sensitive to image quality and
document-layout diversity.

## 17. Future Improvements

Potential improvements include:

-   manually annotated field-level ground truth;
-   exact field accuracy evaluation;
-   larger and harder detection datasets;
-   improved multilingual VIZ OCR;
-   stronger Date of Issue extraction for uncommon layouts;
-   targeted fallback OCR for difficult VIZ regions;
-   API/application deployment;
-   automated regression tests for known failure cases.

## 18. Baseline Status

This version is treated as the current project baseline.

Future changes should be evaluated against a fixed test set or a larger
manually annotated benchmark so improvements can be measured objectively
rather than only through visual inspection.
