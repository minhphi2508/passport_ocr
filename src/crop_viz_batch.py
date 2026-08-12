from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from device_config import YOLO_DEVICE


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "passport_detector_ver3_best.pt"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "passport_pages_safe"
    / "transformed"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "viz_stage"
)

COLOR_DIR = (
    OUTPUT_ROOT
    / "color"
)

GRAYSCALE_DIR = (
    OUTPUT_ROOT
    / "grayscale"
)

DEBUG_DIR = (
    OUTPUT_ROOT
    / "debug"
)

CSV_PATH = (
    OUTPUT_ROOT
    / "viz_crop_results.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50


# ============================================================
# VIZ CROP CONFIG
# ============================================================

# Khi có MRZ:
# VIZ kết thúc phía trên MRZ một khoảng nhỏ.
MRZ_TOP_MARGIN_RATIO = 0.04

# QUAN TRỌNG:
#
# Khi không detect được MRZ, KHÔNG crop theo tỷ lệ nữa.
# Toàn bộ passport page sẽ được đưa sang VIZ OCR.
#
# Lý do:
# - ảnh có thể thực sự không chứa MRZ do khách crop mất
# - không muốn tiếp tục làm mất VIZ information
# - VIZ OCR có thể xử lý cả full page


# ============================================================
# UTILITIES
# ============================================================

def safe_output_name(
    image_path: Path,
    input_root: Path,
) -> str:

    relative_path = (
        image_path.relative_to(
            input_root
        )
    )

    parts = list(
        relative_path.parts
    )

    stem = Path(
        parts[-1]
    ).stem

    safe_stem = "__".join(
        parts[:-1]
        + [
            stem
        ]
    )

    return (
        f"{safe_stem}.jpg"
    )


def create_grayscale_variant(
    viz_crop: np.ndarray,
) -> np.ndarray:

    gray = cv2.cvtColor(
        viz_crop,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(
            8,
            8,
        ),
    )

    return clahe.apply(
        gray
    )


# ============================================================
# MRZ DETECTION
# ============================================================

def detect_best_mrz(
    model: YOLO,
    image_path: Path,
) -> dict[str, object] | None:

    result = model.predict(
        source=str(
            image_path
        ),
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=YOLO_DEVICE,
        save=False,
        verbose=False,
    )[0]

    candidates: list[
        dict[
            str,
            object,
        ]
    ] = []

    if (
        result.boxes
        is not None
    ):

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            class_name = (
                result.names[
                    class_id
                ]
            )

            if (
                class_name
                != "mrz"
            ):
                continue

            confidence = float(
                box.conf.item()
            )

            (
                x1,
                y1,
                x2,
                y2,
            ) = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )

            candidates.append(
                {
                    "confidence":
                        confidence,

                    "bbox": (
                        int(
                            round(
                                x1
                            )
                        ),
                        int(
                            round(
                                y1
                            )
                        ),
                        int(
                            round(
                                x2
                            )
                        ),
                        int(
                            round(
                                y2
                            )
                        ),
                    ),
                }
            )

    del result

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item:
            item[
                "confidence"
            ],
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(
    model: YOLO,
    image_path: Path,
) -> dict[str, object]:

    output_name = (
        safe_output_name(
            image_path,
            INPUT_DIR,
        )
    )

    color_path = (
        COLOR_DIR
        / output_name
    )

    grayscale_path = (
        GRAYSCALE_DIR
        / output_name
    )

    debug_path = (
        DEBUG_DIR
        / output_name
    )

    image = cv2.imread(
        str(
            image_path
        )
    )

    relative_path = str(
        image_path.relative_to(
            INPUT_DIR
        )
    )

    if image is None:

        return {
            "filename":
                image_path.name,

            "relative_path":
                relative_path,

            "status":
                "image_read_failed",

            "mrz_detected":
                False,

            "mrz_confidence":
                None,

            "viz_mode":
                None,

            "viz_top":
                None,

            "viz_bottom":
                None,

            "viz_width":
                None,

            "viz_height":
                None,

            "error":
                "OpenCV không đọc được ảnh.",
        }

    (
        height,
        width,
    ) = image.shape[:2]

    mrz = detect_best_mrz(
        model=model,
        image_path=image_path,
    )

    debug_image = (
        image.copy()
    )

    # ========================================================
    # MODE A
    # MRZ DETECTED
    #
    # Giữ behavior cũ:
    # crop vùng phía trên MRZ.
    # ========================================================

    if (
        mrz
        is not None
    ):

        (
            x1,
            y1,
            x2,
            y2,
        ) = mrz[
            "bbox"
        ]

        margin = int(
            round(
                height
                * MRZ_TOP_MARGIN_RATIO
            )
        )

        viz_top = 0

        viz_bottom = max(
            1,
            y1 - margin,
        )

        viz_crop = image[
            viz_top:viz_bottom,
            0:width,
        ]

        viz_mode = (
            "mrz_based"
        )

        mrz_detected = True

        mrz_confidence = (
            float(
                mrz[
                    "confidence"
                ]
            )
        )

        # ----------------------------------------------------
        # DEBUG MRZ
        # ----------------------------------------------------

        cv2.rectangle(
            debug_image,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                0,
                255,
            ),
            3,
        )

        cv2.line(
            debug_image,
            (
                0,
                viz_bottom,
            ),
            (
                width - 1,
                viz_bottom,
            ),
            (
                0,
                255,
                0,
            ),
            3,
        )

        cv2.putText(
            debug_image,
            (
                f"MRZ "
                f"{mrz_confidence:.3f}"
            ),
            (
                x1,
                max(
                    25,
                    y1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            debug_image,
            "VIZ: MRZ-BASED",
            (
                20,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # MODE B
    # NO MRZ
    #
    # NEW:
    # Không dùng 78% fallback.
    # Giữ TOÀN BỘ passport page.
    # ========================================================

    else:

        viz_top = 0
        viz_bottom = height

        viz_crop = (
            image.copy()
        )

        viz_mode = (
            "full_page_no_mrz"
        )

        mrz_detected = False
        mrz_confidence = None

        cv2.rectangle(
            debug_image,
            (
                0,
                0,
            ),
            (
                width - 1,
                height - 1,
            ),
            (
                0,
                255,
                255,
            ),
            4,
        )

        cv2.putText(
            debug_image,
            "VIZ: FULL PAGE - NO MRZ",
            (
                20,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                0,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # VALIDATE CROP
    # ========================================================

    if (
        viz_crop.size
        == 0
    ):

        return {
            "filename":
                image_path.name,

            "relative_path":
                relative_path,

            "status":
                "empty_viz_crop",

            "mrz_detected":
                mrz_detected,

            "mrz_confidence":
                mrz_confidence,

            "viz_mode":
                viz_mode,

            "viz_top":
                viz_top,

            "viz_bottom":
                viz_bottom,

            "viz_width":
                0,

            "viz_height":
                0,

            "error":
                "VIZ crop rỗng.",
        }

    # ========================================================
    # CREATE VARIANTS
    # ========================================================

    grayscale = (
        create_grayscale_variant(
            viz_crop
        )
    )

    # ========================================================
    # WRITE FILES
    # ========================================================

    color_success = (
        cv2.imwrite(
            str(
                color_path
            ),
            viz_crop,
        )
    )

    grayscale_success = (
        cv2.imwrite(
            str(
                grayscale_path
            ),
            grayscale,
        )
    )

    debug_success = (
        cv2.imwrite(
            str(
                debug_path
            ),
            debug_image,
        )
    )

    if not (
        color_success
        and grayscale_success
        and debug_success
    ):

        return {
            "filename":
                image_path.name,

            "relative_path":
                relative_path,

            "status":
                "write_failed",

            "mrz_detected":
                mrz_detected,

            "mrz_confidence":
                mrz_confidence,

            "viz_mode":
                viz_mode,

            "viz_top":
                viz_top,

            "viz_bottom":
                viz_bottom,

            "viz_width":
                viz_crop.shape[1],

            "viz_height":
                viz_crop.shape[0],

            "error":
                "Không ghi được một hoặc nhiều output.",
        }

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "filename":
            image_path.name,

        "relative_path":
            relative_path,

        "status":
            "success",

        "mrz_detected":
            mrz_detected,

        "mrz_confidence":
            (
                round(
                    mrz_confidence,
                    6,
                )
                if (
                    mrz_confidence
                    is not None
                )
                else None
            ),

        "viz_mode":
            viz_mode,

        "viz_top":
            viz_top,

        "viz_bottom":
            viz_bottom,

        "viz_width":
            viz_crop.shape[1],

        "viz_height":
            viz_crop.shape[0],

        "color_path":
            str(
                color_path
            ),

        "grayscale_path":
            str(
                grayscale_path
            ),

        "debug_path":
            str(
                debug_path
            ),

        "error":
            None,
    }


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(
    rows:
        list[
            dict[
                str,
                object,
            ]
        ],
) -> None:

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = (
            csv.DictWriter(
                csv_file,
                fieldnames=(
                    fieldnames
                ),
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Không thấy model:\n"
            f"{MODEL_PATH}"
        )

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Không thấy input folder:\n"
            f"{INPUT_DIR}"
        )

    image_paths = sorted(
        path
        for path in INPUT_DIR.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:

        raise RuntimeError(
            f"Không có ảnh trong:\n"
            f"{INPUT_DIR}"
        )

    for directory in (
        OUTPUT_ROOT,
        COLOR_DIR,
        GRAYSCALE_DIR,
        DEBUG_DIR,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        f"Tổng passport page : "
        f"{len(image_paths)}"
    )

    print(
        f"Input              : "
        f"{INPUT_DIR}"
    )

    print()

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    rows = []

    for (
        index,
        image_path,
    ) in enumerate(
        image_paths,
        start=1,
    ):

        try:

            row = (
                process_one_image(
                    model=model,
                    image_path=image_path,
                )
            )

        except Exception as error:

            row = {
                "filename":
                    image_path.name,

                "relative_path":
                    str(
                        image_path.relative_to(
                            INPUT_DIR
                        )
                    ),

                "status":
                    "unexpected_error",

                "mrz_detected":
                    None,

                "mrz_confidence":
                    None,

                "viz_mode":
                    None,

                "error":
                    repr(
                        error
                    ),
            }

        rows.append(
            row
        )

        write_csv(
            rows
        )

        print(
            f"["
            f"{index:>4}"
            f"/"
            f"{len(image_paths)}"
            f"] "
            f"{row['relative_path']} "
            f"-> "
            f"{row['status']} "
            f"| mode="
            f"{row.get('viz_mode')}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    status_counts = {}
    mode_counts = {}

    for row in rows:

        status = str(
            row[
                "status"
            ]
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

        mode = row.get(
            "viz_mode"
        )

        if mode:

            mode_counts[
                str(mode)
            ] = (
                mode_counts.get(
                    str(mode),
                    0,
                )
                + 1
            )

    print()
    print(
        "=" * 76
    )

    print(
        "KẾT QUẢ CROP VIZ"
    )

    print(
        "=" * 76
    )

    print()
    print(
        "STATUS"
    )

    print(
        "-" * 76
    )

    for (
        status,
        count,
    ) in sorted(
        status_counts.items()
    ):

        print(
            f"{status:<32}: "
            f"{count}"
        )

    print()
    print(
        "VIZ MODES"
    )

    print(
        "-" * 76
    )

    for (
        mode,
        count,
    ) in sorted(
        mode_counts.items()
    ):

        print(
            f"{mode:<32}: "
            f"{count}"
        )

    print()

    print(
        f"Color     : "
        f"{COLOR_DIR}"
    )

    print(
        f"Grayscale : "
        f"{GRAYSCALE_DIR}"
    )

    print(
        f"Debug     : "
        f"{DEBUG_DIR}"
    )

    print(
        f"CSV       : "
        f"{CSV_PATH}"
    )


if __name__ == "__main__":
    main()