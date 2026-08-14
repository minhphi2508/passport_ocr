from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from annotation_sampling import (
    FIELDS,
    build_annotation_queue,
    clean,
    export_ground_truth_rows,
    load_csv,
    mark_needs_review,
    near_duplicate_identity_suggestions,
    progress_summary,
    propagate_anchor,
    queue_fieldnames,
    truthy,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = PROJECT_ROOT / "outputs" / "final_results" / "passport_extraction_results.csv"
GT_DIR = PROJECT_ROOT / "ground_truth"
QUEUE_CSV = GT_DIR / "annotation_queue.csv"
IDENTITY_SUMMARY_CSV = GT_DIR / "annotation_identity_selection.csv"
IDENTITY_HINTS_CSV = GT_DIR / "identity_review_suggestions.csv"
DEFAULT_GT_CSV = GT_DIR / "passport_ground_truth.csv"
PAGE_DIR = PROJECT_ROOT / "outputs" / "passport_pages_safe" / "images"
MRZ_DIR = PROJECT_ROOT / "outputs" / "mrz_stage" / "original"
VIZ_DIR = PROJECT_ROOT / "outputs" / "viz_stage" / "color"


FIELD_LABELS = {
    "passport_number": "Passport number",
    "surname": "Surname",
    "given_names": "Given names",
    "nationality": "Nationality",
    "date_of_birth": "Date of birth",
    "sex": "Sex",
    "date_of_expiry": "Date of expiry",
    "date_of_issue": "Date of issue",
}


def prepare(target_identities: int, force: bool, max_identity_hints: int) -> None:
    if QUEUE_CSV.exists() and not force:
        raise FileExistsError(
            f"Annotation queue đã tồn tại:\n{QUEUE_CSV}\n\n"
            "Dùng --force nếu thực sự muốn tạo lại và ghi đè tiến độ annotate."
        )

    rows = load_csv(FINAL_CSV)
    queue_rows, identity_summaries = build_annotation_queue(
        rows=rows,
        target_identities=target_identities,
    )
    write_csv(QUEUE_CSV, queue_rows, queue_fieldnames())
    write_csv(IDENTITY_SUMMARY_CSV, identity_summaries)

    hints = near_duplicate_identity_suggestions(rows, max_pairs=max_identity_hints)
    if hints:
        write_csv(IDENTITY_HINTS_CSV, hints)

    summary = progress_summary(queue_rows)
    high_groups = sum(item.get("identity_confidence") == "high" for item in identity_summaries)
    medium_groups = sum(item.get("identity_confidence") == "medium" for item in identity_summaries)
    singleton_groups = sum(item.get("identity_confidence") == "single" for item in identity_summaries)

    print("=" * 76)
    print("ANNOTATION QUEUE PREPARED")
    print("=" * 76)
    print(f"Final-result samples    : {len(rows)}")
    print(f"Selected identities     : {summary['selected_identities']}")
    print(f"Selected image variants : {summary['total_selected_samples']}")
    print(f"High-confidence groups  : {high_groups}")
    print(f"Medium-confidence groups: {medium_groups}")
    print(f"Safe singleton groups   : {singleton_groups}")
    print(f"\nQueue: {QUEUE_CSV}")
    print(f"Identity summary: {IDENTITY_SUMMARY_CSV}")
    if hints:
        print(f"Possible identity merges (review only): {IDENTITY_HINTS_CSV}")
    print("\nNext: python src/annotation_assistant.py gui")



def extend_queue(target_identities: int) -> None:
    if not QUEUE_CSV.exists():
        prepare(target_identities=target_identities, force=False, max_identity_hints=200)
        return

    existing = load_csv(QUEUE_CSV)
    final_rows = load_csv(FINAL_CSV)
    fresh_rows, identity_summaries = build_annotation_queue(
        rows=final_rows,
        target_identities=target_identities,
    )

    existing_by_sample = {clean(row.get("sample_id")): row for row in existing}
    fresh_ids = {clean(row.get("sample_id")) for row in fresh_rows}
    preserve_keys = [
        "identity_id",
        "annotation_status",
        "propagated_from_sample_id",
        "split",
        "notes",
        *[f"gt_{field}" for field in FIELDS],
    ]

    for row in fresh_rows:
        old = existing_by_sample.get(clean(row.get("sample_id")))
        if old is None:
            continue
        for key in preserve_keys:
            if key in old:
                row[key] = old[key]

    # Never discard prior human work if a future sampling policy changes.
    for row in existing:
        if clean(row.get("sample_id")) not in fresh_ids:
            fresh_rows.append(row)

    write_csv(QUEUE_CSV, fresh_rows, queue_fieldnames())
    write_csv(IDENTITY_SUMMARY_CSV, identity_summaries)
    summary = progress_summary(fresh_rows)
    print("=" * 76)
    print("ANNOTATION QUEUE EXTENDED")
    print("=" * 76)
    print(f"Selected identities : {summary['selected_identities']}")
    print(f"Covered samples     : {summary['covered_samples']}/{summary['total_selected_samples']}")
    print("Existing verified GT was preserved.")
    print(f"Queue: {QUEUE_CSV}")

def print_status(queue_path: Path) -> None:
    rows = load_csv(queue_path)
    summary = progress_summary(rows)
    print("=" * 76)
    print("ANNOTATION PROGRESS")
    print("=" * 76)
    print(f"Selected identities : {summary['selected_identities']}")
    print(f"Verified identities : {summary['verified_identities']}")
    print(f"Pending identities  : {summary['pending_identities']}")
    print(f"Needs review        : {summary['needs_review_identities']}")
    print(f"Covered samples     : {summary['covered_samples']}/{summary['total_selected_samples']}")
    if summary["verified_identities"]:
        print(f"Samples per manual anchor: {summary['manual_saving_factor']:.2f}x")


def export_ground_truth(
    queue_path: Path,
    output: Path,
    assign_splits: bool,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> None:
    rows = load_csv(queue_path)
    gt_rows = export_ground_truth_rows(rows)
    if not gt_rows:
        raise RuntimeError("Chưa có identity nào được verify để export.")

    fieldnames = [
        "sample_id",
        "identity_id",
        "split",
        "filename",
        "relative_path",
        *FIELDS,
        "notes",
    ]
    write_csv(output, gt_rows, fieldnames)
    print(f"Exported ground truth: {output}")
    print(f"Samples              : {len(gt_rows)}")
    print(f"Identities           : {len({row['identity_id'] for row in gt_rows})}")

    if assign_splits:
        from ground_truth_tools import assign_splits

        assign_splits(
            path=output,
            output=output,
            train=train,
            val=val,
            test=test,
            seed=seed,
        )
        print(f"Identity-safe split đã được gán: train={train:.0%}, val={val:.0%}, test={test:.0%}.")


def _find_image_path(row: dict[str, str], variant_sample_id: str | None = None) -> Path | None:
    # Original input is best for human verification.
    relative_path = clean(row.get("relative_path"))
    if relative_path:
        candidate = PROJECT_ROOT / "input_images" / Path(relative_path)
        if candidate.exists():
            return candidate

    sample_id = variant_sample_id or clean(row.get("sample_id"))
    generated_name = clean(row.get("generated_filename")) or f"{sample_id}.jpg"

    candidates = [
        PROJECT_ROOT / "outputs" / "passport_pages_safe" / "transformed" / generated_name,
        PROJECT_ROOT / "outputs" / "passport_pages_safe" / "crops" / generated_name,
        VIZ_DIR / generated_name,
        PROJECT_ROOT / "outputs" / "mrz_stage" / "original_crops" / generated_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def launch_gui(queue_path: Path) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as error:
        raise RuntimeError(
            "Python trên máy không có tkinter. Trên Windows Python chính thức thường có sẵn tkinter."
        ) from error

    try:
        from PIL import Image, ImageTk
    except ImportError as error:
        raise RuntimeError(
            "GUI cần Pillow để hiển thị JPG. Cài trong venv bằng: pip install pillow"
        ) from error

    rows = load_csv(queue_path)
    anchors = [row for row in rows if truthy(row.get("is_anchor"))]
    if not anchors:
        raise RuntimeError("Annotation queue không có anchor rows.")

    # Work on unfinished anchors first while preserving queue order.
    anchors.sort(
        key=lambda row: (
            1 if clean(row.get("annotation_status")) == "verified" else 0,
            1 if clean(row.get("annotation_status")) == "needs_review" else 0,
        )
    )

    class AnnotationApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("Passport OCR — Verify-first Ground Truth")
            self.root.geometry("1500x900")
            self.index = 0
            self.preview_member_index = 0
            self.photo: Any = None
            self.field_vars: dict[str, tk.StringVar] = {}
            self.pred_vars: dict[str, tk.StringVar] = {}
            self.identity_var = tk.StringVar()
            self.notes_var = tk.StringVar()
            self.propagate_var = tk.BooleanVar(value=True)
            self.status_var = tk.StringVar()
            self.meta_var = tk.StringVar()
            self.preview_var = tk.StringVar()

            self._build_ui()
            self._bind_keys()
            self.load_current()

        def _build_ui(self) -> None:
            top = ttk.Frame(self.root, padding=8)
            top.pack(fill="x")
            ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(side="left")
            ttk.Button(top, text="Previous", command=self.previous).pack(side="right", padx=4)
            ttk.Button(top, text="Next", command=self.next).pack(side="right", padx=4)

            main = ttk.Panedwindow(self.root, orient="horizontal")
            main.pack(fill="both", expand=True, padx=8, pady=4)

            image_frame = ttk.Frame(main, padding=8)
            form_frame = ttk.Frame(main, padding=8)
            main.add(image_frame, weight=3)
            main.add(form_frame, weight=2)

            self.image_label = ttk.Label(image_frame, anchor="center")
            self.image_label.pack(fill="both", expand=True)
            ttk.Label(image_frame, textvariable=self.preview_var).pack(fill="x", pady=(6, 0))
            preview_buttons = ttk.Frame(image_frame)
            preview_buttons.pack(fill="x", pady=4)
            ttk.Button(preview_buttons, text="◀ Variant", command=lambda: self.change_preview(-1)).pack(side="left")
            ttk.Button(preview_buttons, text="Variant ▶", command=lambda: self.change_preview(1)).pack(side="left", padx=4)

            ttk.Label(form_frame, textvariable=self.meta_var, justify="left", wraplength=560).grid(
                row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10)
            )

            ttk.Label(form_frame, text="Identity ID").grid(row=1, column=0, sticky="w")
            ttk.Entry(form_frame, textvariable=self.identity_var, width=45).grid(
                row=1, column=1, columnspan=2, sticky="ew", pady=2
            )

            ttk.Label(form_frame, text="Field", font=("Segoe UI", 9, "bold")).grid(row=2, column=0, sticky="w")
            ttk.Label(form_frame, text="Prediction", font=("Segoe UI", 9, "bold")).grid(row=2, column=1, sticky="w")
            ttk.Label(form_frame, text="Ground truth (edit only if wrong)", font=("Segoe UI", 9, "bold")).grid(
                row=2, column=2, sticky="w"
            )

            start_row = 3
            for offset, field in enumerate(FIELDS):
                row_index = start_row + offset
                self.pred_vars[field] = tk.StringVar()
                self.field_vars[field] = tk.StringVar()
                ttk.Label(form_frame, text=FIELD_LABELS[field]).grid(row=row_index, column=0, sticky="w", pady=3)
                pred_entry = ttk.Entry(form_frame, textvariable=self.pred_vars[field], state="readonly", width=28)
                pred_entry.grid(row=row_index, column=1, sticky="ew", padx=(0, 6), pady=3)
                ttk.Entry(form_frame, textvariable=self.field_vars[field], width=34).grid(
                    row=row_index, column=2, sticky="ew", pady=3
                )

            actions_row = start_row + len(FIELDS)
            ttk.Checkbutton(
                form_frame,
                text="Propagate GT to all image variants in this suggested identity",
                variable=self.propagate_var,
            ).grid(row=actions_row, column=0, columnspan=3, sticky="w", pady=(10, 4))

            ttk.Label(form_frame, text="Notes").grid(row=actions_row + 1, column=0, sticky="w")
            ttk.Entry(form_frame, textvariable=self.notes_var).grid(
                row=actions_row + 1, column=1, columnspan=2, sticky="ew", pady=3
            )

            buttons = ttk.Frame(form_frame)
            buttons.grid(row=actions_row + 2, column=0, columnspan=3, sticky="ew", pady=(12, 4))
            ttk.Button(buttons, text="Reset GT = prediction", command=self.reset_predictions).pack(side="left")
            ttk.Button(buttons, text="Needs review", command=self.save_needs_review).pack(side="right", padx=4)
            ttk.Button(buttons, text="VERIFY + NEXT", command=self.verify_next).pack(side="right", padx=4)
            ttk.Button(buttons, text="Save", command=self.save_only).pack(side="right", padx=4)

            help_text = (
                "Shortcuts: Ctrl+Enter = VERIFY + NEXT | Ctrl+S = Save | Ctrl+R = Needs review | "
                "Alt+Left/Right = Previous/Next | Alt+Up/Down = preview variants"
            )
            ttk.Label(form_frame, text=help_text, wraplength=560).grid(
                row=actions_row + 3, column=0, columnspan=3, sticky="w", pady=(8, 0)
            )

            form_frame.columnconfigure(1, weight=1)
            form_frame.columnconfigure(2, weight=1)

        def _bind_keys(self) -> None:
            self.root.bind("<Control-Return>", lambda _event: self.verify_next())
            self.root.bind("<Control-s>", lambda _event: self.save_only())
            self.root.bind("<Control-r>", lambda _event: self.save_needs_review())
            self.root.bind("<Alt-Left>", lambda _event: self.previous())
            self.root.bind("<Alt-Right>", lambda _event: self.next())
            self.root.bind("<Alt-Up>", lambda _event: self.change_preview(-1))
            self.root.bind("<Alt-Down>", lambda _event: self.change_preview(1))

        def current(self) -> dict[str, str]:
            return anchors[self.index]

        def group_members(self) -> list[dict[str, str]]:
            group_id = clean(self.current().get("suggested_identity_id"))
            return [row for row in rows if clean(row.get("suggested_identity_id")) == group_id]

        def load_current(self) -> None:
            row = self.current()
            self.preview_member_index = 0
            self.identity_var.set(clean(row.get("identity_id")) or clean(row.get("suggested_identity_id")))
            self.notes_var.set(clean(row.get("notes")))
            for field in FIELDS:
                self.pred_vars[field].set(clean(row.get(f"pred_{field}")))
                self.field_vars[field].set(clean(row.get(f"gt_{field}")) or clean(row.get(f"pred_{field}")))

            # High-confidence passport-number groups propagate by default. Medium groups require explicit opt-in.
            self.propagate_var.set(clean(row.get("identity_confidence")) == "high")
            group_size = clean(row.get("group_size")) or "1"
            self.meta_var.set(
                f"sample_id: {clean(row.get('sample_id'))}\n"
                f"file: {clean(row.get('relative_path')) or clean(row.get('filename'))}\n"
                f"identity suggestion: {clean(row.get('suggested_identity_id'))} "
                f"({clean(row.get('identity_confidence'))}, {clean(row.get('identity_evidence'))}) | group size={group_size}\n"
                f"difficulty: {clean(row.get('annotation_difficulty'))} | risk={clean(row.get('annotation_risk_score'))} | "
                f"quality={clean(row.get('quality_status'))} | coverage={clean(row.get('coverage_status'))}\n"
                f"review reasons: {clean(row.get('review_reasons')) or '-'}\n"
                f"source conflicts: {clean(row.get('source_conflict_fields')) or '-'}"
            )
            self.status_var.set(
                f"Identity {self.index + 1}/{len(anchors)} | status={clean(row.get('annotation_status'))}"
            )
            self.show_preview()

        def show_preview(self) -> None:
            members = self.group_members()
            if not members:
                return
            self.preview_member_index %= len(members)
            member = members[self.preview_member_index]
            image_path = _find_image_path(member, clean(member.get("sample_id")))
            self.preview_var.set(
                f"Variant {self.preview_member_index + 1}/{len(members)} — "
                f"{clean(member.get('relative_path')) or clean(member.get('filename'))}"
            )
            if image_path is None:
                self.image_label.configure(text="Không tìm thấy ảnh local cho sample này.", image="")
                self.photo = None
                return

            try:
                image = Image.open(image_path).convert("RGB")
                max_w, max_h = 850, 720
                image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=self.photo, text="")
            except Exception as error:
                self.image_label.configure(text=f"Không mở được ảnh:\n{image_path}\n{error}", image="")
                self.photo = None

        def change_preview(self, delta: int) -> None:
            members = self.group_members()
            if not members:
                return
            self.preview_member_index = (self.preview_member_index + delta) % len(members)
            self.show_preview()

        def reset_predictions(self) -> None:
            for field in FIELDS:
                self.field_vars[field].set(self.pred_vars[field].get())

        def _persist(self) -> None:
            write_csv(queue_path, rows, queue_fieldnames())

        def _gt_values(self) -> dict[str, str]:
            return {field: clean(self.field_vars[field].get()) for field in FIELDS}

        def save_only(self) -> None:
            row = self.current()
            row["identity_id"] = clean(self.identity_var.get()) or clean(row.get("suggested_identity_id"))
            row["notes"] = clean(self.notes_var.get())
            for field in FIELDS:
                row[f"gt_{field}"] = clean(self.field_vars[field].get())
            self._persist()
            self.status_var.set(f"Saved {clean(row.get('sample_id'))}")

        def verify_next(self) -> None:
            row = self.current()
            identity_id = clean(self.identity_var.get()) or clean(row.get("suggested_identity_id"))
            gt_values = self._gt_values()
            if not any(gt_values.values()):
                if not messagebox.askyesno(
                    "Empty ground truth",
                    "Tất cả field đang trống. Vẫn mark identity này là verified?",
                ):
                    return

            propagate = bool(self.propagate_var.get())
            affected = propagate_anchor(
                rows=rows,
                anchor_sample_id=clean(row.get("sample_id")),
                identity_id=identity_id,
                gt_values=gt_values,
                notes=clean(self.notes_var.get()),
                propagate=propagate,
            )
            self._persist()
            self.status_var.set(f"Verified. GT applied to {affected} sample(s).")
            self.next()

        def save_needs_review(self) -> None:
            row = self.current()
            mark_needs_review(
                rows,
                clean(row.get("sample_id")),
                clean(self.notes_var.get()) or "manual_identity_or_field_review_needed",
            )
            self._persist()
            self.next()

        def previous(self) -> None:
            if self.index > 0:
                self.index -= 1
                self.load_current()

        def next(self) -> None:
            if self.index < len(anchors) - 1:
                self.index += 1
                self.load_current()
            else:
                summary = progress_summary(rows)
                messagebox.showinfo(
                    "Annotation queue",
                    f"Đã đến cuối queue.\n\n"
                    f"Verified identities: {summary['verified_identities']}/{summary['selected_identities']}\n"
                    f"Covered samples: {summary['covered_samples']}/{summary['total_selected_samples']}\n\n"
                    "Bạn có thể đóng GUI và chạy export.",
                )

    root = tk.Tk()
    AnnotationApp(root)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify-first ground-truth annotation assistant for Passport OCR."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Select identities and pre-fill annotation queue from predictions.")
    p_prepare.add_argument("--target-identities", type=int, default=40)
    p_prepare.add_argument("--force", action="store_true")
    p_prepare.add_argument("--max-identity-hints", type=int, default=200)

    p_extend = sub.add_parser("extend", help="Increase target identities without losing existing annotations.")
    p_extend.add_argument("--target-identities", type=int, required=True)

    p_gui = sub.add_parser("gui", help="Open local verify-first annotation GUI.")
    p_gui.add_argument("--queue", type=Path, default=QUEUE_CSV)

    p_status = sub.add_parser("status")
    p_status.add_argument("--queue", type=Path, default=QUEUE_CSV)

    p_export = sub.add_parser("export")
    p_export.add_argument("--queue", type=Path, default=QUEUE_CSV)
    p_export.add_argument("--output", type=Path, default=DEFAULT_GT_CSV)
    p_export.add_argument(
        "--assign-splits",
        action="store_true",
        help="Assign identity-safe splits immediately after export.",
    )
    p_export.add_argument("--train", type=float, default=0.0)
    p_export.add_argument("--val", type=float, default=0.5)
    p_export.add_argument("--test", type=float, default=0.5)
    p_export.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.target_identities, args.force, args.max_identity_hints)
    elif args.command == "extend":
        extend_queue(args.target_identities)
    elif args.command == "gui":
        launch_gui(args.queue)
    elif args.command == "status":
        print_status(args.queue)
    elif args.command == "export":
        export_ground_truth(
            args.queue,
            args.output,
            args.assign_splits,
            args.train,
            args.val,
            args.test,
            args.seed,
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
