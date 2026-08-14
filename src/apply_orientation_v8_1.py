from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "src" / "process_passport_pages.py"


OLD_LANDSCAPE = """def rotate_to_landscape(image: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    if height > width:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), True
    return image, False
"""


NEW_LANDSCAPE = """def rotate_to_landscape(image: np.ndarray) -> tuple[np.ndarray, bool]:
    # Orientation V8: semantic orientation is resolved earlier by
    # run_passport_gate() from passport/MRZ geometry.
    # Do not apply a second width/height-only rotation here.
    return image, False
"""


START_MARKER = "def build_orientation_candidate("
END_MARKER = "\ndef create_edge_map("


NEW_ORIENTATION_BLOCK = r"""def rotate_bbox(
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

"""


def apply_patch() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"Không thấy target:\n{TARGET}"
        )

    text = TARGET.read_text(
        encoding="utf-8"
    )

    if (
        "mrz_geometry_orientation"
        in text
    ):
        print(
            "Orientation V8 đã được "
            "áp dụng trước đó."
        )
        return

    if (
        OLD_LANDSCAPE
        not in text
    ):
        raise RuntimeError(
            "Không tìm thấy "
            "rotate_to_landscape() "
            "của version repo hiện tại."
        )

    start = text.find(
        START_MARKER
    )

    end = text.find(
        END_MARKER,
        start,
    )

    if (
        start < 0
        or end < 0
    ):
        raise RuntimeError(
            "Không tìm thấy orientation "
            "block cần thay."
        )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    backup = TARGET.with_name(
        TARGET.stem
        + (
            ".before_orientation_v8_"
            f"{timestamp}"
        )
        + TARGET.suffix
    )

    shutil.copy2(
        TARGET,
        backup,
    )

    # IMPORTANT:
    # Replace the later orientation block FIRST while the precomputed
    # start/end offsets are still valid. Only then replace the earlier
    # rotate_to_landscape block. Reversing this order shifts offsets and
    # can corrupt the file.
    text = (
        text[:start]
        + NEW_ORIENTATION_BLOCK
        + text[end:]
    )

    if OLD_LANDSCAPE not in text:
        raise RuntimeError(
            "rotate_to_landscape block disappeared unexpectedly "
            "after orientation replacement."
        )

    text = text.replace(
        OLD_LANDSCAPE,
        NEW_LANDSCAPE,
        1,
    )

    TARGET.write_text(
        text,
        encoding="utf-8",
    )

    print("=" * 76)
    print(
        "ORIENTATION V8 PATCH APPLIED"
    )
    print("=" * 76)
    print(
        f"Target : {TARGET}"
    )
    print(
        f"Backup : {backup}"
    )
    print(
        "Semantic orientation: "
        "passport + MRZ geometry"
    )
    print(
        "Blind portrait -> 90 CW "
        "rotation: disabled"
    )


if __name__ == "__main__":
    apply_patch()
