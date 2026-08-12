# Passport OCR Pipeline

End-to-end pipeline for extracting structured information from foreign passport images, with a primary focus on ICAO TD3 passports.

The system combines a YOLO11 passport detector, safe passport-page perspective processing, PaddleOCR-based MRZ/VIZ recognition, TD3 parsing and checksum validation, and hybrid Date of Issue extraction.

The repository is currently maintained for **two execution environments only**:

1. Windows laptop, CPU only.
2. Windows machine with **NVIDIA GeForce RTX 5060 Ti**.

Other configurations may work, but they are not part of the maintained/tested setup.

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
|-- scripts/
|   |-- setup_cpu.ps1
|   `-- setup_rtx5060ti.ps1
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
|-- requirements-rtx5060ti.txt
|-- requirements-gpu.txt
`-- README.md
```

`requirements-gpu.txt` is kept only as a compatibility/deprecation file. New RTX 5060 Ti installations should use `scripts/setup_rtx5060ti.ps1`.

## 4. Supported Environments

### CPU laptop

```text
OS             : Windows
Python         : 3.12
PyTorch        : 2.8.0 CPU
Torchvision    : 0.23.0 CPU
PaddlePaddle   : 3.3.0 CPU
PaddleOCR      : 3.7.0
PaddleX        : 3.7.2
Ultralytics    : 8.4.114
```

### RTX 5060 Ti

```text
OS             : Windows
Python         : 3.12
GPU            : NVIDIA GeForce RTX 5060 Ti
GPU arch       : Blackwell / compute capability 12.0
PyTorch        : 2.8.0
Torchvision    : 0.23.0
PyTorch CUDA   : 12.8
PaddlePaddle   : 3.3.0 GPU
Paddle CUDA    : 12.9 wheel
PaddleOCR      : 3.7.0
PaddleX        : 3.7.2
Ultralytics    : 8.4.114
```

The old PaddlePaddle GPU 3.2.0 CUDA 12.6 setup must not be used on the RTX 5060 Ti. That wheel does not contain support for compute capability 12.0 and can fail with:

```text
Mismatched GPU Architecture
Unsupported GPU architecture
```

## 5. Clone

```powershell
git clone https://github.com/minhphi2508/passport_ocr.git
cd passport_ocr
```

## 6. CPU Setup

Recommended one-command setup:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_cpu.ps1
```

The script creates `.venv` with Python 3.12 if necessary and installs `requirements-cpu.txt`.

Manual equivalent:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-cpu.txt
```

Verify:

```powershell
python -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available())"
python -c "import paddle; print('Paddle:',paddle.__version__); print('CUDA:',paddle.device.is_compiled_with_cuda())"
python src/device_config.py
```

CPU is the expected result for both frameworks.

## 7. RTX 5060 Ti Setup

Use the setup script instead of manually installing a generic `requirements-gpu.txt`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_rtx5060ti.ps1
```

The script performs the installation in this order:

1. Create/activate Python 3.12 virtual environment.
2. Install common project dependencies.
3. Install PyTorch 2.8.0 + Torchvision 0.23.0 from the CUDA 12.8 index.
4. Install PaddlePaddle GPU 3.3.0 from the CUDA 12.9 package index.
5. Run dependency checks.
6. Run real GPU tensor operations with both frameworks.

Manual equivalent:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip setuptools wheel

python -m pip install -r requirements.txt

python -m pip install `
    torch==2.8.0 `
    torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install `
    paddlepaddle-gpu==3.3.0 `
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

### PyTorch GPU verification

```powershell
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x; print('Torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0)); print('Capability:',torch.cuda.get_device_capability(0)); print('Result:',float(y.mean()))"
```

### Paddle GPU verification

```powershell
python -c "import paddle; paddle.set_device('gpu:0'); x=paddle.randn([1024,1024]); y=paddle.matmul(x,x); print('Paddle:',paddle.__version__); print('Device:',paddle.device.get_device()); print('Result:',float(paddle.mean(y)))"
```

A warning such as `No ccache found` is not relevant to normal inference and can be ignored.

When troubleshooting, test `torch` and `paddle` independently. `paddleocr`, `paddlex`, or `modelscope` may transitively import PyTorch, which can make a PyTorch DLL problem appear to be a PaddleOCR problem.

## 8. Dependency Files

### `requirements.txt`

Shared application dependencies only.

It intentionally does not pin/install PyTorch or PaddlePaddle backends.

### `requirements-cpu.txt`

CPU environment. Includes the shared dependencies and CPU framework packages.

### `requirements-rtx5060ti.txt`

Documents the RTX 5060 Ti profile and includes shared dependencies, but the CUDA frameworks are intentionally installed by `scripts/setup_rtx5060ti.ps1` so that each framework uses its correct package index.

### `requirements-gpu.txt`

Deprecated compatibility file. Use `scripts/setup_rtx5060ti.ps1` for new GPU installations.

## 9. Device Selection

`src/device_config.py` detects the two frameworks independently.

YOLO/Ultralytics:

```text
CUDA available -> device 0
otherwise      -> cpu
```

PaddleOCR:

```text
CUDA build + GPU available -> gpu:0
otherwise                  -> cpu
```

Run:

```powershell
python src/device_config.py
```

The summary performs real detection instead of printing a hard-coded GPU value.

## 10. Detection Model

Place the trained YOLO11 Ver3 weights at:

```text
models/passport_detector_ver3_best.pt
```

Detector classes:

```text
mrz
passport_page
```

## 11. Input

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

## 12. Running the Pipeline

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

`--fresh` removes generated outputs/checkpoints from the previous run and is only appropriate when starting from stage 1.

Resume from stage 3:

```powershell
python src/pipeline.py --start-stage 3
```

Run a selected stage range:

```powershell
python src/pipeline.py --start-stage 8 --end-stage 10
```

## 13. Outputs

Final files:

```text
outputs/final_results/passport_extraction_results.csv
outputs/final_results/passport_extraction_results.json
```

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

Evaluation numbers from development/smoke-test batches should be treated as pipeline/field-availability references unless they are compared against manually verified ground truth.

## 15. Operational Notes

- Keep `.venv/` out of Git.
- Do not install random newer Torch/Paddle versions into a working environment without a reason.
- Do not use PaddlePaddle 3.2.0 CUDA 12.6 on the RTX 5060 Ti.
- A successful real tensor operation is a stronger GPU health check than only checking device count.
- If the current environment runs the production stages successfully, avoid changing packages merely to remove harmless warnings.
- The first PaddleOCR run may download official OCR models.
