# Passport OCR Benchmark Toolkit V3

This toolkit adds a proper accuracy/failure-analysis loop on top of the V2 pipeline.

## What is new

1. `ground_truth_tools.py`
   - creates a ground-truth template from final results;
   - validates duplicate IDs and identity leakage;
   - assigns train/val/test at **identity level**, never image level.

2. `failure_audit.py`
   - compares final predictions with GT;
   - classifies each wrong field into probable failure stage:
     `passport_stage`, `mrz_ocr`, `mrz_parse`, `viz_field_extraction`, `doi_extraction`, `fusion`, etc.;
   - writes detailed and aggregate reports.

3. `build_review_bundle.py`
   - gathers source image and available passport/MRZ/VIZ crops for failed samples;
   - writes one review folder per sample with human-readable error metadata.

4. `benchmark_suite.py`
   - runs GT validation, accuracy evaluation, failure audit and review bundle in one command.

## Recommended workflow

First run your current V2 pipeline and make sure final results exist.

```bash
python src/ground_truth_tools.py create-template
```

Fill `ground_truth/passport_ground_truth.csv`, especially `identity_id` and the eight target fields.

Then assign identity-safe splits:

```bash
python src/ground_truth_tools.py assign-splits
```

Validate the GT file:

```bash
python src/ground_truth_tools.py validate
```

Run the complete benchmark:

```bash
python src/benchmark_suite.py
```

Key outputs are under `outputs/evaluation/`.

## Important

Do not tune thresholds against the test split. Use `val` for threshold/model decisions and keep `test` frozen for final reporting.

## Split metrics

`split_metrics.py` reports both image-level and identity-level accuracy for every split. During development, use `val` metrics for decisions. Treat `test` as frozen final evaluation.
