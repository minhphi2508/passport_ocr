from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from device_config import YOLO_DEVICE


# ============================================================
# CẤU HÌNH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "passport_detector_ver3_best.pt"
)

INPUT_DIR = PROJECT_ROOT / "input_images"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "passport_pages_safe"
)

CROP_DIR = OUTPUT_ROOT / "crops"
TRANSFORMED_DIR = OUTPUT_ROOT / "transformed"
DEBUG_DIR = OUTPUT_ROOT / "debug"
CSV_PATH = OUTPUT_ROOT / "processing_results.csv"

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
CROP_PADDING_RATIO = 0.03

# Cố ý đặt nghiêm ngặt:
# perspective sai nguy hiểm hơn không perspective.
MIN_QUAD_AREA_RATIO = 0.72
MIN_MRZ_INSIDE_RATIO = 0.95
MAX_CONTOURS_TO_CHECK = 15


# ============================================================
# ORIENTATION RETRY CONFIG
# ============================================================

ORIENTATION_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# ============================================================
# HÀM CƠ BẢN
# ============================================================

def safe_output_name(
    image_path: Path,
    input_root: Path,
) -> str:
    relative_path = image_path.relative_to(
        input_root
    )

    parts = list(
        relative_path.parts
    )

    stem = Path(
        parts[-1]
    ).stem

    safe_stem = "__".join(
        parts[:-1] + [stem]
    )

    return f"{safe_stem}.jpg"


def add_padding(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    padding_x = int(
        round(
            width
            * padding_ratio
        )
    )

    padding_y = int(
        round(
            height
            * padding_ratio
        )
    )

    return (
        max(
            0,
            x1 - padding_x,
        ),
        max(
            0,
            y1 - padding_y,
        ),
        min(
            image_width,
            x2 + padding_x,
        ),
        min(
            image_height,
            y2 + padding_y,
        ),
    )


def order_points(
    points: np.ndarray,
) -> np.ndarray:
    points = points.astype(
        np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    sums = points.sum(
        axis=1
    )

    differences = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    ordered[0] = points[
        np.argmin(
            sums
        )
    ]

    ordered[1] = points[
        np.argmin(
            differences
        )
    ]

    ordered[2] = points[
        np.argmax(
            sums
        )
    ]

    ordered[3] = points[
        np.argmax(
            differences
        )
    ]

    return ordered


def perspective_warp(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    ordered = order_points(
        corners
    )

    (
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ) = ordered

    top_width = np.linalg.norm(
        top_right
        - top_left
    )

    bottom_width = np.linalg.norm(
        bottom_right
        - bottom_left
    )

    left_height = np.linalg.norm(
        bottom_left
        - top_left
    )

    right_height = np.linalg.norm(
        bottom_right
        - top_right
    )

    output_width = int(
        round(
            max(
                top_width,
                bottom_width,
            )
        )
    )

    output_height = int(
        round(
            max(
                left_height,
                right_height,
            )
        )
    )

    if (
        output_width < 100
        or output_height < 100
    ):
        raise ValueError(
            "Perspective output quá nhỏ."
        )

    destination = np.array(
        [
            [
                0,
                0,
            ],
            [
                output_width - 1,
                0,
            ],
            [
                output_width - 1,
                output_height - 1,
            ],
            [
                0,
                output_height - 1,
            ],
        ],
        dtype=np.float32,
    )

    matrix = (
        cv2.getPerspectiveTransform(
            ordered,
            destination,
        )
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (
            output_width,
            output_height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=(
            cv2.BORDER_REPLICATE
        ),
    )


def rotate_to_landscape(
    image: np.ndarray,
) -> tuple[
    np.ndarray,
    bool,
]:
    height, width = (
        image.shape[:2]
    )

    if height > width:
        return (
            cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            ),
            True,
        )

    return (
        image,
        False,
    )


# ============================================================
# DETECTION
# ============================================================

def detect_objects(
    model: YOLO,
    image_source: Path | np.ndarray,
) -> dict[str, object] | None:

    source = (
        str(
            image_source
        )
        if isinstance(
            image_source,
            Path,
        )
        else image_source
    )

    result = model.predict(
        source=source,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=YOLO_DEVICE,
        save=False,
        verbose=False,
    )[0]

    passport_candidates = []
    mrz_candidates = []

    if result.boxes is not None:
        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            class_name = (
                result.names[
                    class_id
                ]
            )

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

            item = {
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

            if (
                class_name
                == "passport_page"
            ):
                passport_candidates.append(
                    item
                )

            elif (
                class_name
                == "mrz"
            ):
                mrz_candidates.append(
                    item
                )

    del result

    if not passport_candidates:
        return None

    best_passport = max(
        passport_candidates,
        key=lambda item:
            item[
                "confidence"
            ],
    )

    best_mrz = (
        max(
            mrz_candidates,
            key=lambda item:
                item[
                    "confidence"
                ],
        )
        if mrz_candidates
        else None
    )

    return {
        "passport":
            best_passport,

        "mrz":
            best_mrz,
    }


# ============================================================
# ORIENTATION RETRY
# ============================================================

def detect_with_orientation_retry(
    model: YOLO,
    image: np.ndarray,
) -> tuple[
    dict[str, object] | None,
    np.ndarray,
    bool,
    int,
]:
    """
    Detect passport ở orientation gốc trước.

    Chỉ khi không detect được passport_page
    mới thử lần lượt:
    90°, 180°, 270°.

    Nếu nhiều orientation đều detect được,
    chọn orientation có passport confidence
    cao nhất.
    """

    # --------------------------------------------------------
    # 1. ORIGINAL
    # --------------------------------------------------------

    detections = detect_objects(
        model=model,
        image_source=image,
    )

    if detections is not None:
        return (
            detections,
            image,
            False,
            0,
        )

    # --------------------------------------------------------
    # 2. RETRY ORIENTATIONS
    # --------------------------------------------------------

    candidates = []

    for (
        angle,
        rotation_code,
    ) in (
        ORIENTATION_ROTATIONS.items()
    ):

        rotated = cv2.rotate(
            image,
            rotation_code,
        )

        rotated_detections = (
            detect_objects(
                model=model,
                image_source=rotated,
            )
        )

        if (
            rotated_detections
            is None
        ):
            continue

        passport_confidence = float(
            rotated_detections[
                "passport"
            ][
                "confidence"
            ]
        )

        candidates.append(
            {
                "angle":
                    angle,

                "image":
                    rotated,

                "detections":
                    rotated_detections,

                "passport_confidence":
                    passport_confidence,
            }
        )

    # --------------------------------------------------------
    # 3. VẪN KHÔNG DETECT ĐƯỢC
    # --------------------------------------------------------

    if not candidates:
        return (
            None,
            image,
            True,
            0,
        )

    # --------------------------------------------------------
    # 4. CHỌN ORIENTATION TỐT NHẤT
    # --------------------------------------------------------

    best = max(
        candidates,
        key=lambda item:
            item[
                "passport_confidence"
            ],
    )

    return (
        best[
            "detections"
        ],
        best[
            "image"
        ],
        True,
        int(
            best[
                "angle"
            ]
        ),
    )


# ============================================================
# TẠO EDGE MAP
# ============================================================

def create_edge_map(
    image: np.ndarray,
) -> np.ndarray:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (
            5,
            5,
        ),
        0,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    enhanced = clahe.apply(
        blurred
    )

    edges = cv2.Canny(
        enhanced,
        40,
        130,
    )

    kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                5,
                5,
            ),
        )
    )

    edges = (
        cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
    )

    return cv2.dilate(
        edges,
        kernel,
        iterations=1,
    )


# ============================================================
# VALIDATION TỨ GIÁC
# ============================================================

def bbox_to_crop_coordinates(
    bbox:
        tuple[
            int,
            int,
            int,
            int,
        ],

    crop_bbox:
        tuple[
            int,
            int,
            int,
            int,
        ],

) -> tuple[
    int,
    int,
    int,
    int,
]:
    (
        bx1,
        by1,
        bx2,
        by2,
    ) = bbox

    (
        crop_x1,
        crop_y1,
        _,
        _,
    ) = crop_bbox

    return (
        bx1
        - crop_x1,

        by1
        - crop_y1,

        bx2
        - crop_x1,

        by2
        - crop_y1,
    )


def polygon_bbox_intersection_ratio(
    polygon: np.ndarray,

    bbox:
        tuple[
            int,
            int,
            int,
            int,
        ],

    image_shape:
        tuple[
            int,
            int,
        ],

) -> float:

    height, width = (
        image_shape
    )

    polygon_mask = (
        np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )
    )

    bbox_mask = (
        np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )
    )

    polygon_int = (
        order_points(
            polygon
        )
        .astype(
            np.int32
        )
    )

    cv2.fillConvexPoly(
        polygon_mask,
        polygon_int,
        255,
    )

    (
        x1,
        y1,
        x2,
        y2,
    ) = bbox

    x1 = max(
        0,
        min(
            width,
            x1,
        ),
    )

    x2 = max(
        0,
        min(
            width,
            x2,
        ),
    )

    y1 = max(
        0,
        min(
            height,
            y1,
        ),
    )

    y2 = max(
        0,
        min(
            height,
            y2,
        ),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return 0.0

    bbox_mask[
        y1:y2,
        x1:x2,
    ] = 255

    bbox_area = int(
        np.count_nonzero(
            bbox_mask
        )
    )

    if bbox_area == 0:
        return 0.0

    intersection = (
        cv2.bitwise_and(
            polygon_mask,
            bbox_mask,
        )
    )

    intersection_area = int(
        np.count_nonzero(
            intersection
        )
    )

    return (
        intersection_area
        / bbox_area
    )


def quad_is_valid(
    corners: np.ndarray,

    crop_shape:
        tuple[
            int,
            int,
        ],

    mrz_bbox_in_crop:
        tuple[
            int,
            int,
            int,
            int,
        ]
        | None,

) -> tuple[
    bool,
    str,
]:

    (
        crop_height,
        crop_width,
    ) = crop_shape

    crop_area = float(
        crop_width
        * crop_height
    )

    ordered = order_points(
        corners
    )

    quad_area = abs(
        cv2.contourArea(
            ordered.astype(
                np.float32
            )
        )
    )

    area_ratio = (
        quad_area
        / crop_area
    )

    if (
        area_ratio
        < MIN_QUAD_AREA_RATIO
    ):
        return (
            False,
            (
                "quad_area_too_small:"
                f"{area_ratio:.3f}"
            ),
        )

    (
        top_left,
        top_right,
        bottom_right,
        bottom_left,
    ) = ordered

    top_y = min(
        top_left[1],
        top_right[1],
    )

    bottom_y = max(
        bottom_left[1],
        bottom_right[1],
    )

    left_x = min(
        top_left[0],
        bottom_left[0],
    )

    right_x = max(
        top_right[0],
        bottom_right[0],
    )

    # Tứ giác phải phủ gần toàn bộ
    # chiều rộng và chiều cao crop.

    if (
        left_x
        > crop_width
        * 0.18
    ):
        return (
            False,
            "left_edge_too_far_inside",
        )

    if (
        right_x
        < crop_width
        * 0.82
    ):
        return (
            False,
            "right_edge_too_far_inside",
        )

    if (
        top_y
        > crop_height
        * 0.18
    ):
        return (
            False,
            "top_edge_too_far_inside",
        )

    if (
        bottom_y
        < crop_height
        * 0.82
    ):
        return (
            False,
            "bottom_edge_too_far_inside",
        )

    if (
        mrz_bbox_in_crop
        is not None
    ):

        mrz_inside_ratio = (
            polygon_bbox_intersection_ratio(
                polygon=ordered,
                bbox=(
                    mrz_bbox_in_crop
                ),
                image_shape=(
                    crop_height,
                    crop_width,
                ),
            )
        )

        if (
            mrz_inside_ratio
            < MIN_MRZ_INSIDE_RATIO
        ):
            return (
                False,
                (
                    "mrz_not_fully_inside:"
                    f"{mrz_inside_ratio:.3f}"
                ),
            )

        # Cạnh dưới của tứ giác
        # tuyệt đối không được nằm
        # phía trên MRZ.

        (
            _,
            _,
            _,
            mrz_y2,
        ) = mrz_bbox_in_crop

        if (
            bottom_y
            < (
                mrz_y2
                - crop_height
                * 0.02
            )
        ):
            return (
                False,
                "quad_ends_above_mrz",
            )

    return (
        True,
        "valid",
    )


def find_page_corners(
    crop: np.ndarray,

    mrz_bbox_in_crop:
        tuple[
            int,
            int,
            int,
            int,
        ]
        | None,

) -> tuple[
    np.ndarray | None,
    np.ndarray,
    str,
]:

    edges = create_edge_map(
        crop
    )

    contours, _ = (
        cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    )

    if not contours:
        return (
            None,
            edges,
            "no_contour",
        )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[
        :MAX_CONTOURS_TO_CHECK
    ]

    rejection_reasons = []

    for contour in contours:

        perimeter = (
            cv2.arcLength(
                contour,
                True,
            )
        )

        for epsilon_ratio in (
            0.012,
            0.015,
            0.020,
            0.025,
            0.030,
            0.040,
        ):

            approximation = (
                cv2.approxPolyDP(
                    contour,
                    (
                        epsilon_ratio
                        * perimeter
                    ),
                    True,
                )
            )

            if (
                len(
                    approximation
                )
                != 4
            ):
                continue

            if not (
                cv2.isContourConvex(
                    approximation
                )
            ):
                continue

            corners = (
                approximation.reshape(
                    4,
                    2,
                )
            )

            (
                valid,
                reason,
            ) = quad_is_valid(
                corners=corners,
                crop_shape=(
                    crop.shape[:2]
                ),
                mrz_bbox_in_crop=(
                    mrz_bbox_in_crop
                ),
            )

            if valid:
                return (
                    corners,
                    edges,
                    "valid_quad",
                )

            rejection_reasons.append(
                reason
            )

    if rejection_reasons:
        return (
            None,
            edges,
            rejection_reasons[0],
        )

    return (
        None,
        edges,
        "no_valid_quad",
    )


# ============================================================
# DEBUG
# ============================================================

def save_debug_image(
    crop: np.ndarray,
    edges: np.ndarray,
    corners: np.ndarray | None,

    mrz_bbox:
        tuple[
            int,
            int,
            int,
            int,
        ]
        | None,

    output_path: Path,
) -> None:

    visualization = (
        crop.copy()
    )

    if (
        mrz_bbox
        is not None
    ):

        (
            x1,
            y1,
            x2,
            y2,
        ) = mrz_bbox

        cv2.rectangle(
            visualization,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                255,
                0,
                0,
            ),
            3,
        )

        cv2.putText(
            visualization,
            "MRZ",
            (
                x1,
                max(
                    25,
                    y1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                255,
                0,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    if corners is not None:

        ordered = (
            order_points(
                corners
            )
            .astype(
                np.int32
            )
        )

        cv2.polylines(
            visualization,
            [
                ordered
            ],
            True,
            (
                0,
                255,
                0,
            ),
            4,
        )

    edge_bgr = (
        cv2.cvtColor(
            edges,
            cv2.COLOR_GRAY2BGR,
        )
    )

    if (
        edge_bgr.shape[0]
        != visualization.shape[0]
    ):

        edge_bgr = cv2.resize(
            edge_bgr,
            (
                edge_bgr.shape[1],
                visualization.shape[0],
            ),
        )

    combined = np.hstack(
        [
            visualization,
            edge_bgr,
        ]
    )

    cv2.imwrite(
        str(
            output_path
        ),
        combined,
    )


# ============================================================
# XỬ LÝ MỘT ẢNH
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

    crop_path = (
        CROP_DIR
        / output_name
    )

    transformed_path = (
        TRANSFORMED_DIR
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

    if image is None:
        return {
            "filename":
                image_path.name,

            "relative_path":
                str(
                    image_path.relative_to(
                        INPUT_DIR
                    )
                ),

            "status":
                "image_read_failed",

            "validation_reason":
                None,

            "orientation_retry_used":
                False,

            "orientation_angle":
                None,

            "error":
                (
                    "OpenCV không "
                    "đọc được ảnh."
                ),
        }

    # ========================================================
    # DETECT + ORIENTATION RETRY
    # ========================================================

    (
        detections,
        working_image,
        orientation_retry_used,
        orientation_angle,
    ) = (
        detect_with_orientation_retry(
            model=model,
            image=image,
        )
    )

    if detections is None:
        return {
            "filename":
                image_path.name,

            "relative_path":
                str(
                    image_path.relative_to(
                        INPUT_DIR
                    )
                ),

            "status":
                "passport_page_not_detected",

            "validation_reason":
                None,

            "orientation_retry_used":
                orientation_retry_used,

            "orientation_angle":
                None,

            "error":
                None,
        }

    # Từ đây trở đi bbox YOLO thuộc
    # hệ tọa độ của working_image.

    image = working_image

    (
        image_height,
        image_width,
    ) = image.shape[:2]

    passport = (
        detections[
            "passport"
        ]
    )

    mrz = (
        detections[
            "mrz"
        ]
    )

    # ========================================================
    # PASSPORT CROP
    # ========================================================

    padded_bbox = (
        add_padding(
            bbox=(
                passport[
                    "bbox"
                ]
            ),
            image_width=(
                image_width
            ),
            image_height=(
                image_height
            ),
            padding_ratio=(
                CROP_PADDING_RATIO
            ),
        )
    )

    (
        x1,
        y1,
        x2,
        y2,
    ) = padded_bbox

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return {
            "filename":
                image_path.name,

            "relative_path":
                str(
                    image_path.relative_to(
                        INPUT_DIR
                    )
                ),

            "status":
                "empty_crop",

            "validation_reason":
                None,

            "orientation_retry_used":
                orientation_retry_used,

            "orientation_angle":
                orientation_angle,

            "error":
                "Crop rỗng.",
        }

    cv2.imwrite(
        str(
            crop_path
        ),
        crop,
    )

    # ========================================================
    # MRZ COORDINATE IN CROP
    # ========================================================

    mrz_bbox_in_crop = None

    if mrz is not None:

        mrz_bbox_in_crop = (
            bbox_to_crop_coordinates(
                bbox=(
                    mrz[
                        "bbox"
                    ]
                ),
                crop_bbox=(
                    padded_bbox
                ),
            )
        )

    # ========================================================
    # SAFE PERSPECTIVE
    # ========================================================

    (
        corners,
        edges,
        validation_reason,
    ) = find_page_corners(
        crop=crop,
        mrz_bbox_in_crop=(
            mrz_bbox_in_crop
        ),
    )

    perspective_applied = False
    rotated = False

    if corners is not None:

        try:

            transformed = (
                perspective_warp(
                    crop,
                    corners,
                )
            )

            perspective_applied = (
                True
            )

            status = (
                "perspective_success"
            )

        except (
            cv2.error,
            ValueError,
        ) as error:

            transformed = (
                crop.copy()
            )

            status = (
                "perspective_failed_fallback"
            )

            validation_reason = (
                repr(
                    error
                )
            )

    else:

        transformed = (
            crop.copy()
        )

        status = (
            "safe_fallback_crop"
        )

    # ========================================================
    # FINAL LANDSCAPE NORMALIZATION
    # ========================================================

    (
        transformed,
        rotated,
    ) = rotate_to_landscape(
        transformed
    )

    cv2.imwrite(
        str(
            transformed_path
        ),
        transformed,
    )

    # ========================================================
    # DEBUG
    # ========================================================

    save_debug_image(
        crop=crop,
        edges=edges,
        corners=corners,
        mrz_bbox=(
            mrz_bbox_in_crop
        ),
        output_path=(
            debug_path
        ),
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "filename":
            image_path.name,

        "relative_path":
            str(
                image_path.relative_to(
                    INPUT_DIR
                )
            ),

        "status":
            status,

        "validation_reason":
            validation_reason,

        "orientation_retry_used":
            orientation_retry_used,

        "orientation_angle":
            orientation_angle,

        "passport_confidence":
            round(
                float(
                    passport[
                        "confidence"
                    ]
                ),
                6,
            ),

        "mrz_confidence":
            (
                round(
                    float(
                        mrz[
                            "confidence"
                        ]
                    ),
                    6,
                )
                if mrz is not None
                else None
            ),

        "perspective_applied":
            perspective_applied,

        "rotated_to_landscape":
            rotated,

        "crop_width":
            crop.shape[1],

        "crop_height":
            crop.shape[0],

        "output_width":
            transformed.shape[1],

        "output_height":
            transformed.shape[0],

        "crop_path":
            str(
                crop_path
            ),

        "transformed_path":
            str(
                transformed_path
            ),

        "debug_path":
            str(
                debug_path
            ),

        "error":
            None,
    }


# ============================================================
# MAIN
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

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main() -> None:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không thấy model:\n"
            f"{MODEL_PATH}"
        )

    image_paths = sorted(
        path
        for path in (
            INPUT_DIR.rglob(
                "*"
            )
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
        CROP_DIR,
        TRANSFORMED_DIR,
        DEBUG_DIR,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    rows = []

    print(
        f"Tổng ảnh: "
        f"{len(image_paths)}"
    )

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
                    image_path=(
                        image_path
                    ),
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

                "validation_reason":
                    None,

                "orientation_retry_used":
                    None,

                "orientation_angle":
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
            f"("
            f"{row.get('validation_reason')}"
            f") "
            f"orientation="
            f"{row.get('orientation_angle')}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    status_counts = {}

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

    orientation_retry_count = sum(
        1
        for row in rows
        if (
            row.get(
                "orientation_retry_used"
            )
            is True
        )
    )

    orientation_rescued_count = sum(
        1
        for row in rows
        if (
            row.get(
                "orientation_retry_used"
            )
            is True
            and row.get(
                "status"
            )
            not in {
                "passport_page_not_detected",
                "image_read_failed",
                "unexpected_error",
            }
        )
    )

    orientation_angle_counts = {
        90: 0,
        180: 0,
        270: 0,
    }

    for row in rows:

        angle = row.get(
            "orientation_angle"
        )

        if (
            angle
            in orientation_angle_counts
        ):
            orientation_angle_counts[
                angle
            ] += 1

    print(
        "\n"
        + "=" * 72
    )

    print(
        "KẾT QUẢ PERSPECTIVE AN TOÀN"
    )

    print(
        "=" * 72
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
        "ORIENTATION RETRY"
    )

    print(
        "-" * 72
    )

    print(
        f"Retry used                      : "
        f"{orientation_retry_count}"
    )

    print(
        f"Successfully rescued            : "
        f"{orientation_rescued_count}"
    )

    print(
        f"Selected 90°                    : "
        f"{orientation_angle_counts[90]}"
    )

    print(
        f"Selected 180°                   : "
        f"{orientation_angle_counts[180]}"
    )

    print(
        f"Selected 270°                   : "
        f"{orientation_angle_counts[270]}"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()