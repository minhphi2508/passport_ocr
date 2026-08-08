# Mentor #1 Passport OCR — Checkpoint (2026-08-08)

## Scope

- Foreign passport OCR, focused on ICAO TD3-style passports.
- Dataset also contains specimen / synthetic / AI-generated / non-standard passports.
- Pipeline therefore stays permissive:
  - TD3 structure/checksum are quality signals.
  - Non-standard records are not hard-rejected only because checksum/structure fails.

## Current production pipeline

1. Detect and rectify passport page.
2. Detect/crop MRZ.
3. Build MRZ variants:
   - original
   - grayscale
   - threshold
4. PaddleOCR on MRZ.
5. Parse TD3 fields.
6. Validate MRZ checksums.
7. Crop/process VIZ.
8. PaddleOCR on VIZ.
9. Extract Date of Issue.
10. Build final CSV/JSON.

Main entry point:

```powershell
python src/pipeline.py
```

## Current production change

`src/extract_date_of_issue.py` now keeps the existing V3 DOI extractor as the baseline and adds a fallback-only spatial rescue layer.

Policy:

- If V3 already finds a DOI, keep it.
- Only rescue when V3 has no DOI.
- Search all VIZ OCR variants:
  - enhanced
  - color
  - grayscale
- Use strong Date-of-Issue label + spatial proximity.
- Explicitly block obvious DOB / expiry / validity labels.

Reproducible test on the current 1061-record VIZ batch:

- Baseline HEAD DOI: 734 / 1061
- Patched DOI: 749 / 1061
- Rescued: 15
- Lost: 0
- One comparison anomaly was observed for `clear100.jpg`; it was intentionally not used to drive further tuning before closing this checkpoint.

## MRZ recovery experiments

Experimental MRZ recovery scripts were tested and removed from `src/` before packaging.

Findings:

- Arbitrary filler movement was too aggressive and produced false recoveries.
- Conservative single-filler insertion on 43-character line 2 recovered only a very small number of strong cases.
- Because the gain was small, MRZ recovery was NOT integrated into production at this checkpoint.

## MRZ OCR failure audit

On the current MRZ OCR output:

- Total MRZ OCR records: 1012
- Usable 2-line records: 922
- Selected records with fewer than 2 lines: 90
- Another OCR variant already had 2 lines: 0
- Any failed variant with raw text length >= 80: 0

Failure categories:

- very_little_text: 52
- ocr_empty_all_variants: 38

Important interpretation:

- Failures cluster across roughly 10 passport identities and then repeat across image variants.
- Some of those source identities/images are visually very poor even to a human.
- Therefore the 90 image-level failures should not be interpreted as 90 independent passport failures.
- Do not over-tune detector/OCR against these dirty variants without first filtering visibly unusable samples and reviewing identity-level failure rates.

## Crop audit

MRZ detector/crop medians:

Usable 2-line:
- detector confidence: ~0.875
- crop width: 1874
- crop height: 226
- crop aspect: ~8.30
- resized output height: 180

Very little text:
- detector confidence: ~0.848
- crop width: 1652.5
- crop height: 170.5
- crop aspect: ~9.67

OCR empty all variants:
- detector confidence: ~0.842
- crop width: 1591
- crop height: 164
- crop aspect: ~9.99

No detector/crop change was integrated before closing this checkpoint.

## Backup

Important runtime artifacts were backed up to Google Drive before cleanup, including the raw/intermediate OCR outputs needed to continue analysis without rerunning GPU inference.

Git intentionally ignores:

- `.venv/`
- `input_images/*`
- `outputs/*`

The repository should contain code/config/models only, not generated runtime outputs.

## Experimental scripts removed before packaging

The following local experiment files were intentionally deleted from `src/`:

- audit_mrz_crops.py
- audit_mrz_crops_v2.py
- audit_mrz_failures.py
- extract_date_of_issue_v4.py
- extract_date_of_issue_v41.py
- extract_date_of_issue_v42.py
- prepare_mrz_failure_review.py
- recover_mrz_v1.py
- recover_mrz_v2.py
- rerank_mrz_results.py

## Recommended next work when resuming

Do not restart by rewriting the parser.

First:

1. Review identity-level MRZ failure rates after removing visually unusable images.
2. Inspect only identities that still fail consistently on readable images.
3. Decide whether the next bottleneck is:
   - detector/crop coverage,
   - MRZ preprocessing/upscaling,
   - OCR model behavior,
   - or a small targeted parser/recovery rule.
4. Keep evaluation identity-aware because many samples are variants of the same passport.

## Current closure decision

This checkpoint is considered good enough to stop tuning temporarily.

Reason:

- The pipeline is broadly functional.
- Many apparent image-level failures are repeated variants of a small number of difficult identities.
- Additional heuristic tuning at this point has diminishing returns and higher overfitting risk.
