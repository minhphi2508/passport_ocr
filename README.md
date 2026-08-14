# Passport OCR Pipeline

End-to-end passport OCR pipeline focused on ICAO TD3 passports.

The pipeline combines:

- YOLO11 passport-page / MRZ detection
- safe perspective processing and landscape normalization
- PaddleOCR MRZ recognition
- ICAO TD3 parsing and checksum validation
- VIZ OCR and Date of Issue extraction
- final CSV / JSON output

## Supported environments

This repository is maintained for:

1. Windows CPU-only laptop
2. Windows + NVIDIA GeForce RTX 5060 Ti

Python 3.12 is the maintained Python version.

---

## Important RTX 5060 Ti note

On Windows, the maintained RTX 5060 Ti configuration **does not put
PyTorch GPU and PaddlePaddle GPU in the same virtual environment**.

The tested framework profiles use different CUDA runtime package sets:

```text
YOLO / PyTorch
  PyTorch       2.8.0
  Torchvision   0.23.0
  CUDA wheel    cu128

PaddleOCR
  PaddlePaddle  3.3.0 GPU
  PaddleOCR     3.7.0
  PaddleX       3.7.2
  CUDA wheel    cu129
```

Loading both GPU frameworks in one Windows Python process can produce DLL
collisions such as `WinError 127`, including failures while loading
`cublas64_12.dll` or cuDNN DLLs.

The RTX setup therefore creates two isolated environments:

```text
.venv-torch
.venv-paddle
```

The GPU launcher automatically routes each pipeline stage to the correct
Python interpreter.

---

## Project structure

```text
passport_ocr/
|-- input_images/
|-- models/
|   `-- passport_detector_ver3_best.pt
|-- outputs/
|-- scripts/
|   |-- setup_cpu.ps1
|   |-- setup_rtx5060ti.ps1
|   `-- run_rtx5060ti.ps1
|-- src/
|   |-- device_config.py
|   |-- pipeline.py
|   |-- pipeline_rtx5060ti.py
|   |-- process_passport_pages.py
|   |-- crop_mrz_batch.py
|   |-- ocr_mrz_batch.py
|   |-- parse_mrz_results.py
|   |-- validate_mrz_results.py
|   |-- crop_viz_batch.py
|   |-- preprocess_viz_batch.py
|   |-- ocr_viz_batch.py
|   |-- extract_viz_fields.py
|   |-- extract_date_of_issue.py
|   `-- build_final_results.py
|-- requirements.txt
|-- requirements-cpu.txt
|-- requirements-rtx5060ti.txt
|-- requirements-gpu.txt
`-- README.md
```

---

## Clone

```powershell
git clone https://github.com/minhphi2508/passport_ocr.git
cd passport_ocr
```

---

# CPU setup

CPU keeps the existing single-environment workflow.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_cpu.ps1
```

Activate when needed:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
python src\pipeline.py --fresh
```

---

# RTX 5060 Ti setup

## Prerequisites

Install:

- current NVIDIA driver
- Git
- Python 3.12 x64

Verify:

```powershell
nvidia-smi
py -3.12 --version
git --version
```

A separate CUDA Toolkit installation is not required for normal inference;
the maintained PyTorch and PaddlePaddle wheels provide their CUDA runtime
dependencies.

## One-command environment setup

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_rtx5060ti.ps1
```

The setup creates:

```text
.venv-torch
  -> Ultralytics / PyTorch GPU
  -> torch 2.8.0 cu128
  -> torchvision 0.23.0 cu128

.venv-paddle
  -> PaddleOCR / PaddleX
  -> paddlepaddle-gpu 3.3.0 cu129
```

It then runs real GPU tensor smoke tests for both frameworks.

Do **not** activate either environment to run the full GPU pipeline.

---

## RTX device health check

Run:

```powershell
py -3.12 src\device_config.py
```

Expected shape:

```text
========================================================================
DEVICE CONFIGURATION - DUAL GPU ENV
========================================================================
YOLO / PyTorch : GPU 0 (NVIDIA GeForce RTX 5060 Ti, sm_120, ...)
PaddleOCR      : gpu:0 (paddle=3.3.0, gpu_count=1)
...
========================================================================
```

The health check probes Torch and Paddle in separate subprocesses.

---

# Detection model

Place the trained YOLO weights at:

```text
models\passport_detector_ver3_best.pt
```

Detector classes:

```text
mrz
passport_page
```

---

# Input images

Place passport images in:

```text
input_images\
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

---

# Running the RTX 5060 Ti pipeline

Use the RTX launcher, not `python src\pipeline.py`.

List stages:

```powershell
.\scripts\run_rtx5060ti.ps1 --list-stages
```

Clean end-to-end run:

```powershell
.\scripts\run_rtx5060ti.ps1 --fresh
```

Resume:

```powershell
.\scripts\run_rtx5060ti.ps1 --start-stage 3
```

Selected range:

```powershell
.\scripts\run_rtx5060ti.ps1 --start-stage 8 --end-stage 11
```

## GPU stage routing

```text
Detect + Process Passport Pages
  -> .venv-torch
  -> PyTorch / RTX 5060 Ti

OCR MRZ
  -> .venv-paddle
  -> PaddleOCR / RTX 5060 Ti

OCR VIZ
  -> .venv-paddle
  -> PaddleOCR / RTX 5060 Ti

Other crop / parse / validation / result-building stages
  -> .venv-torch
  -> CPU where appropriate
```

The launcher reuses the normal `src/pipeline.py` stage definitions,
checkpointing and output logic. It only changes the Python interpreter used
for each stage.

---

# Outputs

Final output files:

```text
outputs\final_results\passport_extraction_results.csv
outputs\final_results\passport_extraction_results.json
```

Primary fields include:

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

---

# Troubleshooting

## `No ccache found`

A Paddle warning similar to:

```text
No ccache found
```

is not a normal inference failure and may be ignored.

## Paddle works alone but fails after importing Torch

If this succeeds:

```powershell
.\.venv-paddle\Scripts\python.exe -c "import paddle; paddle.utils.run_check()"
```

but mixing `import torch` and `import paddle` produces `WinError 127`, do not
merge the two GPU environments. The dual-environment design exists
specifically to avoid that Windows DLL collision.

## Do not use the old one-venv GPU setup

Do not create one `.venv` containing both:

```text
torch GPU
paddlepaddle-gpu
```

for the maintained RTX 5060 Ti profile.

Do not use PaddlePaddle 3.2.0 CUDA 12.6 on the maintained RTX 5060 Ti setup.

---

# Development notes

Ignore local environments:

```text
.venv/
.venv-torch/
.venv-paddle/
```

Before committing environment changes, verify:

```powershell
py -3.12 -m py_compile src\device_config.py src\pipeline_rtx5060ti.py
.\scripts\run_rtx5060ti.ps1 --list-stages
py -3.12 src\device_config.py
```

For a full GPU smoke test, place at least one image in `input_images` and run:

```powershell
.\scripts\run_rtx5060ti.ps1 --fresh
```
