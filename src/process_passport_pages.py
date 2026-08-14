from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from device_config import YOLO_DEVICE
from sample_manifest import build_manifest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "passport_detector_ver3_best.pt"
INPUT_DIR = PROJECT_ROOT / "input_images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "passport_pages_safe"
CROP_DIR = OUTPUT_ROOT / "crops"
TRANSFORMED_DIR = OUTPUT_ROOT / "transformed"
DEBUG_DIR = OUTPUT_ROOT / "debug"
CSV_PATH = OUTPUT_ROOT / "processing_results.csv"
MANIFEST_CSV = OUTPUT_ROOT / "input_manifest.csv"

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
CROP_PADDING_RATIO = 0.03

MIN_QUAD_AREA_RATIO = 0.72
MIN_MRZ_INSIDE_RATIO = 0.95
MAX_CONTOURS_TO_CHECK = 15

ORIENTATION_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def add_padding(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    padding_x = int(round(width * padding_ratio))
    padding_y = int(round(height * padding_ratio))

    return (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(image_width, x2 + padding_x),
        min(image_height, y2 + padding_y),
    )


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[2] = points[np.argmax(sums)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    ordered = order_points(corners)
    top_left, top_right, bottom_right, bottom_left = ordered

    output_width = int(
        round(
            max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
        )
    )
    output_height = int(
        round(
            max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )
        )
    )

    if output_width < 100 or output_height < 100:
        raise ValueError("Perspective output quá nhỏ.")

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def rotate_to_landscape(image: np.ndarray) -> tuple[np.ndarray, bool]:
    # Orientation V8: semantic orientation is resolved earlier by
    # run_passport_gate() from passport/MRZ geometry.
    # Do not apply a second width/height-only rotation here.
    return image, False


def detect_objects(
    model: YOLO,
    image_source: Path | np.ndarray,
) -> dict[str, Any]:
    source = str(image_source) if isinstance(image_source, Path) else image_source
    result = model.predict(
        source=source,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=YOLO_DEVICE,
        save=False,
        verbose=False,
    )[0]

    passport_candidates: list[dict[str, Any]] = []
    mrz_candidates: list[dict[str, Any]] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            item = {
                "confidence": confidence,
                "bbox": (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ),
            }

            if class_name == "passport_page":
                passport_candidates.append(item)
            elif class_name == "mrz":
                mrz_candidates.append(item)

    del result

    return {
        "passport": (
            max(passport_candidates, key=lambda item: item["confidence"])
            if passport_candidates
            else None
        ),
        "mrz": (
            max(mrz_candidates, key=lambda item: item["confidence"])
            if mrz_candidates
            else None
        ),
    }


def rotate_bbox(
    bbox: tuple[int, int, int, int],
    angle: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    if angle == 0:
        return x1, y1, x2, y2

    if angle == 90:
        return (
            image_height - y2,
            x1,
            image_height - y1,
            x2,
        )

    if angle == 180:
        return (
            image_width - x2,
            image_height - y2,
            image_width - x1,
            image_height - y1,
        )

    if angle == 270:
        return (
            y1,
            image_width - x2,
            y2,
            image_width - x1,
        )

    raise ValueError(
        f"Unsupported orientation angle: {angle}"
    )


def rotate_detections(
    detections: dict[str, Any],
    angle: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for key in ("passport", "mrz"):
        item = detections.get(key)

        if item is None:
            output[key] = None
            continue

        copied = dict(item)

        copied["bbox"] = rotate_bbox(
            tuple(item["bbox"]),
            angle,
            image_width,
            image_height,
        )

        output[key] = copied

    return output


def rotate_image(
    image: np.ndarray,
    angle: int,
) -> np.ndarray:
    if angle == 0:
        return image

    return cv2.rotate(
        image,
        ORIENTATION_ROTATIONS[angle],
    )


def orientation_geometry_score(
    detections: dict[str, Any],
) -> float:
    passport = detections.get("passport")
    mrz = detections.get("mrz")

    if passport is None:
        return -10000.0

    px1, py1, px2, py2 = passport["bbox"]

    pw = max(
        1.0,
        float(px2 - px1),
    )
    ph = max(
        1.0,
        float(py2 - py1),
    )

    passport_ratio = pw / ph

    score = 0.0

    # Upright TD3 data pages should be landscape.
    if passport_ratio >= 1.10:
        score += 65.0
    else:
        score -= 65.0

    score += (
        min(passport_ratio, 2.5)
        * 10.0
    )

    score += (
        float(
            passport.get("confidence")
            or 0.0
        )
        * 5.0
    )

    if mrz is None:
        return round(score, 6)

    mx1, my1, mx2, my2 = mrz["bbox"]

    mw = max(
        1.0,
        float(mx2 - mx1),
    )
    mh = max(
        1.0,
        float(my2 - my1),
    )

    mrz_ratio = mw / mh

    # Upright MRZ should be horizontally wide.
    if mrz_ratio >= 2.0:
        score += 65.0
    else:
        score -= 65.0

    score += (
        min(mrz_ratio, 10.0)
        * 4.0
    )

    score += (
        float(
            mrz.get("confidence")
            or 0.0
        )
        * 5.0
    )

    passport_cx = (
        px1 + px2
    ) / 2.0

    passport_cy = (
        py1 + py2
    ) / 2.0

    mrz_cx = (
        mx1 + mx2
    ) / 2.0

    mrz_cy = (
        my1 + my2
    ) / 2.0

    rel_x = (
        mrz_cx - px1
    ) / pw

    rel_y = (
        mrz_cy - py1
    ) / ph

    width_ratio = (
        mw / pw
    )

    # Distinguishes upright from 180 degrees.
    if mrz_cy > passport_cy:
        score += 70.0
    else:
        score -= 70.0

    # MRZ center is normally close to the lower edge.
    score += max(
        -70.0,
        70.0
        - abs(rel_y - 0.82)
        * 180.0,
    )

    # MRZ should span much of the page width and be centered.
    score += (
        min(
            max(width_ratio, 0.0),
            1.0,
        )
        * 35.0
    )

    score += max(
        -25.0,
        25.0
        - abs(rel_x - 0.50)
        * 70.0,
    )

    if (
        px1 <= mrz_cx <= px2
        and py1 <= mrz_cy <= py2
    ):
        score += 20.0
    else:
        score -= 30.0

    return round(score, 6)


def build_orientation_candidate(
    angle: int,
    image: np.ndarray,
    detections: dict[str, Any],
) -> dict[str, Any]:
    passport = detections.get(
        "passport"
    )

    mrz = detections.get(
        "mrz"
    )

    return {
        "angle": angle,
        "image": image,
        "detections": detections,
        "passport_detected": (
            passport is not None
        ),
        "passport_confidence": (
            float(passport["confidence"])
            if passport is not None
            else None
        ),
        "mrz_detected": (
            mrz is not None
        ),
        "mrz_confidence": (
            float(mrz["confidence"])
            if mrz is not None
            else None
        ),
        "orientation_score": (
            orientation_geometry_score(
                detections
            )
        ),
    }


def geometry_candidates_from_original(
    image: np.ndarray,
    detections: dict[str, Any],
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]

    candidates: list[
        dict[str, Any]
    ] = []

    for angle in (
        0,
        90,
        180,
        270,
    ):
        rotated_image = rotate_image(
            image,
            angle,
        )

        rotated_detections = (
            rotate_detections(
                detections,
                angle,
                width,
                height,
            )
        )

        candidates.append(
            build_orientation_candidate(
                angle=angle,
                image=rotated_image,
                detections=(
                    rotated_detections
                ),
            )
        )

    return candidates


def run_passport_gate(
    model: YOLO,
    image: np.ndarray,
) -> dict[str, Any]:
    original_detections = (
        detect_objects(
            model,
            image,
        )
    )

    original = (
        build_orientation_candidate(
            angle=0,
            image=image,
            detections=(
                original_detections
            ),
        )
    )

    # Fast path:
    # if passport + MRZ already exist, rotate their bboxes mathematically
    # and choose semantic orientation without three extra YOLO calls.
    if (
        original[
            "passport_detected"
        ]
        and original[
            "mrz_detected"
        ]
    ):
        candidates = (
            geometry_candidates_from_original(
                image,
                original_detections,
            )
        )

        best = max(
            candidates,
            key=lambda candidate: (
                candidate[
                    "orientation_score"
                ],
                candidate[
                    "passport_confidence"
                ]
                or 0.0,
                candidate[
                    "mrz_confidence"
                ]
                or 0.0,
            ),
        )

        return {
            "passport_gate_status": (
                "passport_confirmed"
            ),
            "passport_gate_reason": (
                "mrz_geometry_orientation"
            ),
            "orientation_retry_used": (
                int(best["angle"])
                != 0
            ),
            "orientation_angle": int(
                best["angle"]
            ),
            "selected_candidate": best,
        }

    # Retry path when the original orientation does not expose
    # both passport and MRZ.
    candidates = [original]

    for (
        angle,
        rotation_code,
    ) in (
        ORIENTATION_ROTATIONS
        .items()
    ):
        rotated = cv2.rotate(
            image,
            rotation_code,
        )

        candidates.append(
            build_orientation_candidate(
                angle=angle,
                image=rotated,
                detections=(
                    detect_objects(
                        model,
                        rotated,
                    )
                ),
            )
        )

    strong = [
        candidate
        for candidate
        in candidates
        if (
            candidate[
                "passport_detected"
            ]
            and candidate[
                "mrz_detected"
            ]
        )
    ]

    if strong:
        best = max(
            strong,
            key=lambda candidate: (
                candidate[
                    "orientation_score"
                ],
                candidate[
                    "passport_confidence"
                ]
                or 0.0,
                candidate[
                    "mrz_confidence"
                ]
                or 0.0,
            ),
        )

        return {
            "passport_gate_status": (
                "passport_confirmed"
            ),
            "passport_gate_reason": (
                "orientation_retry_mrz_geometry"
            ),
            "orientation_retry_used": True,
            "orientation_angle": int(
                best["angle"]
            ),
            "selected_candidate": best,
        }

    passport_only = [
        candidate
        for candidate
        in candidates
        if candidate[
            "passport_detected"
        ]
    ]

    if passport_only:
        # No MRZ means no reliable 0-vs-180 semantic cue.
        # Prefer landscape page geometry, then confidence.
        best = max(
            passport_only,
            key=lambda candidate: (
                candidate[
                    "orientation_score"
                ],
                candidate[
                    "passport_confidence"
                ]
                or 0.0,
            ),
        )

        return {
            "passport_gate_status": (
                "passport_candidate"
            ),
            "passport_gate_reason": (
                "passport_detected_without_mrz"
            ),
            "orientation_retry_used": (
                int(best["angle"])
                != 0
            ),
            "orientation_angle": int(
                best["angle"]
            ),
            "selected_candidate": best,
        }

    return {
        "passport_gate_status": (
            "no_passport_evidence"
        ),
        "passport_gate_reason": (
            "no_passport_detected_any_orientation"
        ),
        "orientation_retry_used": True,
        "orientation_angle": None,
        "selected_candidate": None,
    }


def create_edge_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edges = cv2.Canny(enhanced, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.dilate(edges, kernel, iterations=1)


def bbox_to_crop_coordinates(
    bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = bbox
    crop_x1, crop_y1, _, _ = crop_bbox
    return bx1 - crop_x1, by1 - crop_y1, bx2 - crop_x1, by2 - crop_y1


def polygon_bbox_intersection_ratio(
    polygon: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> float:
    height, width = image_shape
    polygon_mask = np.zeros((height, width), dtype=np.uint8)
    bbox_mask = np.zeros((height, width), dtype=np.uint8)
    polygon_int = order_points(polygon).astype(np.int32)
    cv2.fillConvexPoly(polygon_mask, polygon_int, 255)

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask[y1:y2, x1:x2] = 255
    bbox_area = int(np.count_nonzero(bbox_mask))
    if bbox_area == 0:
        return 0.0

    intersection = cv2.bitwise_and(polygon_mask, bbox_mask)
    return int(np.count_nonzero(intersection)) / bbox_area


def quad_is_valid(
    corners: np.ndarray,
    crop_shape: tuple[int, int],
    mrz_bbox_in_crop: tuple[int, int, int, int] | None,
) -> tuple[bool, str]:
    crop_height, crop_width = crop_shape
    crop_area = float(crop_width * crop_height)
    ordered = order_points(corners)
    quad_area = abs(cv2.contourArea(ordered.astype(np.float32)))
    area_ratio = quad_area / crop_area

    if area_ratio < MIN_QUAD_AREA_RATIO:
        return False, f"quad_area_too_small:{area_ratio:.3f}"

    top_left, top_right, bottom_right, bottom_left = ordered
    top_y = min(top_left[1], top_right[1])
    bottom_y = max(bottom_left[1], bottom_right[1])
    left_x = min(top_left[0], bottom_left[0])
    right_x = max(top_right[0], bottom_right[0])

    if left_x > crop_width * 0.18:
        return False, "left_edge_too_far_inside"
    if right_x < crop_width * 0.82:
        return False, "right_edge_too_far_inside"
    if top_y > crop_height * 0.18:
        return False, "top_edge_too_far_inside"
    if bottom_y < crop_height * 0.82:
        return False, "bottom_edge_too_far_inside"

    if mrz_bbox_in_crop is not None:
        inside_ratio = polygon_bbox_intersection_ratio(
            polygon=ordered,
            bbox=mrz_bbox_in_crop,
            image_shape=(crop_height, crop_width),
        )
        if inside_ratio < MIN_MRZ_INSIDE_RATIO:
            return False, f"mrz_not_fully_inside:{inside_ratio:.3f}"

        _, _, _, mrz_y2 = mrz_bbox_in_crop
        if bottom_y < mrz_y2 - crop_height * 0.02:
            return False, "quad_ends_above_mrz"

    return True, "valid"


def find_page_corners(
    crop: np.ndarray,
    mrz_bbox_in_crop: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray | None, np.ndarray, str]:
    edges = create_edge_map(crop)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, edges, "no_contour"

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS_TO_CHECK]
    rejection_reasons: list[str] = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.012, 0.015, 0.020, 0.025, 0.030, 0.040):
            approximation = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True,
            )
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue

            corners = approximation.reshape(4, 2)
            valid, reason = quad_is_valid(
                corners=corners,
                crop_shape=crop.shape[:2],
                mrz_bbox_in_crop=mrz_bbox_in_crop,
            )
            if valid:
                return corners, edges, "valid_quad"
            rejection_reasons.append(reason)

    if rejection_reasons:
        return None, edges, rejection_reasons[0]
    return None, edges, "no_valid_quad"


def save_debug_image(
    crop: np.ndarray,
    edges: np.ndarray,
    corners: np.ndarray | None,
    mrz_bbox: tuple[int, int, int, int] | None,
    output_path: Path,
) -> None:
    visualization = crop.copy()

    if mrz_bbox is not None:
        x1, y1, x2, y2 = mrz_bbox
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            visualization,
            "MRZ",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if corners is not None:
        ordered = order_points(corners).astype(np.int32)
        cv2.polylines(visualization, [ordered], True, (0, 255, 0), 4)

    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if edge_bgr.shape[0] != visualization.shape[0]:
        edge_bgr = cv2.resize(
            edge_bgr,
            (edge_bgr.shape[1], visualization.shape[0]),
        )

    combined = np.hstack([visualization, edge_bgr])
    cv2.imwrite(str(output_path), combined)


def transformed_mrz_metadata(
    model: YOLO,
    transformed: np.ndarray,
) -> dict[str, Any]:
    """
    Detect MRZ exactly once on the final transformed page.

    crop_mrz_batch.py and crop_viz_batch.py consume these coordinates and do
    not load/run YOLO again.
    """
    mrz = detect_objects(model, transformed).get("mrz")

    if mrz is None:
        return {
            "transformed_mrz_detected": False,
            "transformed_mrz_confidence": None,
            "transformed_mrz_x1": None,
            "transformed_mrz_y1": None,
            "transformed_mrz_x2": None,
            "transformed_mrz_y2": None,
        }

    x1, y1, x2, y2 = mrz["bbox"]
    return {
        "transformed_mrz_detected": True,
        "transformed_mrz_confidence": round(float(mrz["confidence"]), 6),
        "transformed_mrz_x1": x1,
        "transformed_mrz_y1": y1,
        "transformed_mrz_x2": x2,
        "transformed_mrz_y2": y2,
    }


def process_one_image(
    model: YOLO,
    sample_id: str,
    source_filename: str,
    relative_path: str,
) -> dict[str, Any]:
    image_path = INPUT_DIR / Path(relative_path)
    output_name = f"{sample_id}.jpg"
    crop_path = CROP_DIR / output_name
    transformed_path = TRANSFORMED_DIR / output_name
    debug_path = DEBUG_DIR / output_name

    base = {
        "sample_id": sample_id,
        "source_filename": source_filename,
        "relative_path": relative_path,
        "generated_filename": output_name,
    }

    image = cv2.imread(str(image_path))
    if image is None:
        return {
            **base,
            "status": "image_read_failed",
            "passport_gate_status": "image_read_failed",
            "passport_gate_reason": "opencv_read_failed",
            "orientation_retry_used": False,
            "orientation_angle": None,
            "passport_confidence": None,
            "mrz_detected": False,
            "mrz_confidence": None,
            "perspective_applied": False,
            "rotated_to_landscape": False,
            **transformed_mrz_metadata_empty(),
            "crop_path": None,
            "transformed_path": None,
            "debug_path": None,
            "error": "OpenCV không đọc được ảnh.",
        }

    gate_result = run_passport_gate(model, image)
    selected_candidate = gate_result["selected_candidate"]

    if selected_candidate is None:
        return {
            **base,
            "status": "no_passport_evidence",
            "passport_gate_status": gate_result["passport_gate_status"],
            "passport_gate_reason": gate_result["passport_gate_reason"],
            "orientation_retry_used": gate_result["orientation_retry_used"],
            "orientation_angle": gate_result["orientation_angle"],
            "passport_confidence": None,
            "mrz_detected": False,
            "mrz_confidence": None,
            "perspective_applied": False,
            "rotated_to_landscape": False,
            **transformed_mrz_metadata_empty(),
            "crop_path": None,
            "transformed_path": None,
            "debug_path": None,
            "error": None,
        }

    working_image = selected_candidate["image"]
    detections = selected_candidate["detections"]
    passport = detections["passport"]
    mrz = detections["mrz"]
    image_height, image_width = working_image.shape[:2]

    padded_bbox = add_padding(
        bbox=passport["bbox"],
        image_width=image_width,
        image_height=image_height,
        padding_ratio=CROP_PADDING_RATIO,
    )
    x1, y1, x2, y2 = padded_bbox
    crop = working_image[y1:y2, x1:x2]

    if crop.size == 0:
        return {
            **base,
            "status": "empty_crop",
            "passport_gate_status": gate_result["passport_gate_status"],
            "passport_gate_reason": gate_result["passport_gate_reason"],
            "orientation_retry_used": gate_result["orientation_retry_used"],
            "orientation_angle": gate_result["orientation_angle"],
            "passport_confidence": selected_candidate["passport_confidence"],
            "mrz_detected": selected_candidate["mrz_detected"],
            "mrz_confidence": selected_candidate["mrz_confidence"],
            "perspective_applied": False,
            "rotated_to_landscape": False,
            **transformed_mrz_metadata_empty(),
            "crop_path": None,
            "transformed_path": None,
            "debug_path": None,
            "error": "Crop rỗng.",
        }

    cv2.imwrite(str(crop_path), crop)

    mrz_bbox_in_crop = None
    if mrz is not None:
        mrz_bbox_in_crop = bbox_to_crop_coordinates(mrz["bbox"], padded_bbox)

    corners, edges, validation_reason = find_page_corners(
        crop=crop,
        mrz_bbox_in_crop=mrz_bbox_in_crop,
    )

    perspective_applied = False
    if corners is not None:
        try:
            transformed = perspective_warp(crop, corners)
            perspective_applied = True
            status = "perspective_success"
        except (cv2.error, ValueError) as error:
            transformed = crop.copy()
            status = "perspective_failed_fallback"
            validation_reason = repr(error)
    else:
        transformed = crop.copy()
        status = "safe_fallback_crop"

    transformed, rotated = rotate_to_landscape(transformed)
    cv2.imwrite(str(transformed_path), transformed)

    save_debug_image(
        crop=crop,
        edges=edges,
        corners=corners,
        mrz_bbox=mrz_bbox_in_crop,
        output_path=debug_path,
    )

    mrz_after_transform = transformed_mrz_metadata(model, transformed)

    return {
        **base,
        "status": status,
        "passport_gate_status": gate_result["passport_gate_status"],
        "passport_gate_reason": gate_result["passport_gate_reason"],
        "orientation_retry_used": gate_result["orientation_retry_used"],
        "orientation_angle": gate_result["orientation_angle"],
        "passport_confidence": (
            round(float(selected_candidate["passport_confidence"]), 6)
            if selected_candidate["passport_confidence"] is not None
            else None
        ),
        "mrz_detected": bool(selected_candidate["mrz_detected"]),
        "mrz_confidence": (
            round(float(selected_candidate["mrz_confidence"]), 6)
            if selected_candidate["mrz_confidence"] is not None
            else None
        ),
        "perspective_applied": perspective_applied,
        "rotated_to_landscape": rotated,
        "validation_reason": validation_reason,
        "crop_width": crop.shape[1],
        "crop_height": crop.shape[0],
        "output_width": transformed.shape[1],
        "output_height": transformed.shape[0],
        **mrz_after_transform,
        "crop_path": str(crop_path),
        "transformed_path": str(transformed_path),
        "debug_path": str(debug_path),
        "error": None,
    }


def transformed_mrz_metadata_empty() -> dict[str, Any]:
    return {
        "transformed_mrz_detected": False,
        "transformed_mrz_confidence": None,
        "transformed_mrz_x1": None,
        "transformed_mrz_y1": None,
        "transformed_mrz_x2": None,
        "transformed_mrz_y2": None,
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row})
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không thấy model:\n{MODEL_PATH}")

    for directory in (OUTPUT_ROOT, CROP_DIR, TRANSFORMED_DIR, DEBUG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(INPUT_DIR, MANIFEST_CSV)
    model = YOLO(str(MODEL_PATH))
    rows: list[dict[str, Any]] = []

    print(f"Input images : {len(manifest)}")
    print(f"Manifest     : {MANIFEST_CSV}")

    for index, item in enumerate(manifest, start=1):
        try:
            row = process_one_image(
                model=model,
                sample_id=item["sample_id"],
                source_filename=item["source_filename"],
                relative_path=item["relative_path"],
            )
        except Exception as error:
            row = {
                "sample_id": item["sample_id"],
                "source_filename": item["source_filename"],
                "relative_path": item["relative_path"],
                "generated_filename": f"{item['sample_id']}.jpg",
                "status": "unexpected_error",
                "passport_gate_status": "unexpected_error",
                "passport_gate_reason": None,
                "orientation_retry_used": None,
                "orientation_angle": None,
                "passport_confidence": None,
                "mrz_detected": None,
                "mrz_confidence": None,
                "perspective_applied": False,
                "rotated_to_landscape": False,
                **transformed_mrz_metadata_empty(),
                "crop_path": None,
                "transformed_path": None,
                "debug_path": None,
                "error": repr(error),
            }

        rows.append(row)
        write_csv(rows)

        print(
            f"[{index:>4}/{len(manifest)}] "
            f"{item['relative_path']} -> {row['status']} "
            f"| gate={row['passport_gate_status']} "
            f"| transformed_mrz={row.get('transformed_mrz_detected')}"
        )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\n" + "=" * 76)
    print("PASSPORT PAGE PROCESSING SUMMARY")
    print("=" * 76)
    for status, count in sorted(status_counts.items()):
        print(f"{status:<36}: {count}")
    print(f"\nCSV      : {CSV_PATH}")
    print(f"Manifest : {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
