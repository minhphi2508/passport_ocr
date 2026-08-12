# Passport OCR Upgrade V2

This package is designed to be applied on top of the previous sample-id / single-MRZ-detection upgrade.

## Add these new files to `src/`

- `checkpoint_utils.py`
- `consistency_checks.py`
- `mrz_geometry.py`

## Replace these files in `src/`

- `ocr_mrz_batch.py`
- `ocr_viz_batch.py`
- `build_final_results.py`
- `evaluate_final_results.py`

The full patch bundle also includes the previous upgraded versions of:

- `sample_manifest.py`
- `process_passport_pages.py`
- `crop_mrz_batch.py`
- `crop_viz_batch.py`

## What changed

1. MRZ text fragments are reconstructed into two TD3 rows using OCR geometry (Y-row clustering + X-ordering), with conservative text-order fallback.
2. MRZ/VIZ OCR checkpoints now use append-only JSONL rather than rewriting the full JSON/CSV after every image.
3. OCR checkpoint metadata includes a stage fingerprint derived from code, input artifacts and relevant package versions. Stale checkpoints are automatically invalidated per stage.
4. Final results separate `coverage_status` from `quality_status` / `quality_score`.
5. Final results run MRZ-vs-VIZ source-conflict checks and temporal/document consistency checks. Medium/high inconsistencies trigger review.
6. Evaluation reports coverage, confidence buckets, conflicts, identity-level GT accuracy, and confidence calibration when ground truth exists.

## Recommended run

If you already completed one fresh run with the previous upgrade, rerun from MRZ OCR so the new OCR/checkpoint logic propagates through all downstream stages:

```bash
python src/pipeline.py --start-stage 3
```

If you have not yet run the previous sample-id upgrade, use a fresh run:

```bash
python src/pipeline.py --fresh
```

Ground-truth evaluation remains:

```bash
python src/evaluate_final_results.py --write-ground-truth-template
python src/evaluate_final_results.py
```

## Tests included

The `tests/` directory contains unit tests for MRZ geometry reconstruction, checkpoint invalidation/recovery, and consistency checks. They use the standard-library `unittest` runner:

```bash
python -m unittest discover -s tests -v
```
