from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_GT = PROJECT_ROOT / "ground_truth" / "passport_ground_truth.csv"


def run(script: str, args: list[str]) -> None:
    command = [sys.executable, str(SRC_DIR / script), *args]
    print("\n" + "=" * 76)
    print("RUN:", " ".join(command))
    print("=" * 76)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {script} (return code {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command Passport OCR benchmark suite.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--skip-review-bundle", action="store_true")
    parser.add_argument("--max-review-samples", type=int, default=200)
    args = parser.parse_args()

    if not args.ground_truth.exists():
        raise FileNotFoundError(
            f"Ground truth chưa tồn tại:\n{args.ground_truth}\n\n"
            "Tạo bằng: python src/ground_truth_tools.py create-template"
        )

    run("ground_truth_tools.py", ["validate", "--input", str(args.ground_truth)])
    run("evaluate_final_results.py", ["--ground-truth", str(args.ground_truth)])
    run("split_metrics.py", ["--ground-truth", str(args.ground_truth)])
    run("failure_audit.py", ["--ground-truth", str(args.ground_truth)])

    if not args.skip_review_bundle:
        run("build_review_bundle.py", ["--max-samples", str(args.max_review_samples)])

    print("\n" + "=" * 76)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 76)
    print("Main reports:")
    print("  outputs/evaluation/end_to_end_summary.csv")
    print("  outputs/evaluation/ground_truth_sample_details.csv")
    print("  outputs/evaluation/identity_level_summary.csv")
    print("  outputs/evaluation/quality_calibration.csv")
    print("  outputs/evaluation/split_metrics.csv")
    print("  outputs/evaluation/failure_audit/failure_audit_summary.csv")
    if not args.skip_review_bundle:
        print("  outputs/evaluation/review_bundle/")


if __name__ == "__main__":
    main()
