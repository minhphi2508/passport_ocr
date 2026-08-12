from __future__ import annotations

import re
from statistics import median
from typing import Any

import numpy as np


MIN_GEOMETRY_ROW_GAP_RATIO = 0.35
MIN_GEOMETRY_LINE_LENGTH = 20


def normalize_mrz_piece(text: str) -> str:
    text = str(text).upper().strip()
    text = text.replace("«", "<").replace("‹", "<").replace("＜", "<")
    text = text.replace(" ", "")
    return re.sub(r"[^A-Z0-9<]", "", text)


def box_from_value(value: Any) -> list[float] | None:
    if value is None:
        return None

    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None

    if array.ndim == 1 and array.size >= 4:
        x1, y1, x2, y2 = array[:4].tolist()
        return [float(x1), float(y1), float(x2), float(y2)]

    if array.ndim >= 2 and array.shape[-1] >= 2:
        points = array.reshape(-1, array.shape[-1])[:, :2]
        return [
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ]

    return None


def merge_text_order(fragments: list[str]) -> list[str]:
    cleaned = [normalize_mrz_piece(fragment) for fragment in fragments]
    cleaned = [fragment for fragment in cleaned if fragment]

    if not cleaned:
        return []

    if len(cleaned) == 2:
        return cleaned

    total_text = "".join(cleaned)
    if len(total_text) >= 80:
        best_split = min(
            range(1, len(total_text)),
            key=lambda index: (
                abs(index - 44) + abs((len(total_text) - index) - 44)
            ),
        )
        return [total_text[:best_split], total_text[best_split:]]

    return cleaned


def reconstruct_mrz_lines(items: list[dict[str, Any]]) -> tuple[list[str], str]:
    """
    Reconstruct TD3 rows from OCR geometry.

    The spatial route is used only when there is a clear vertical gap and both
    reconstructed rows contain meaningful text. Otherwise it falls back to the
    previous text-order logic instead of forcing a false two-row split.
    """
    raw_texts = [str(item.get("text") or "") for item in items]

    geometry_items: list[dict[str, Any]] = []
    for item in items:
        cleaned = normalize_mrz_piece(str(item.get("text") or ""))
        box = item.get("box")
        if not cleaned or not isinstance(box, list) or len(box) < 4:
            continue

        x1, y1, x2, y2 = (float(value) for value in box[:4])
        geometry_items.append(
            {
                "text": cleaned,
                "x1": x1,
                "cy": (y1 + y2) / 2.0,
                "height": max(1.0, y2 - y1),
            }
        )

    if len(geometry_items) >= 2:
        by_y = sorted(geometry_items, key=lambda item: item["cy"])
        gaps = [
            by_y[index + 1]["cy"] - by_y[index]["cy"]
            for index in range(len(by_y) - 1)
        ]

        split_index = int(np.argmax(gaps)) + 1
        largest_gap = gaps[split_index - 1]
        typical_height = median(item["height"] for item in by_y)

        if largest_gap >= max(2.0, typical_height * MIN_GEOMETRY_ROW_GAP_RATIO):
            upper_items = by_y[:split_index]
            lower_items = by_y[split_index:]

            upper = "".join(
                item["text"] for item in sorted(upper_items, key=lambda x: x["x1"])
            )
            lower = "".join(
                item["text"] for item in sorted(lower_items, key=lambda x: x["x1"])
            )

            if (
                len(upper) >= MIN_GEOMETRY_LINE_LENGTH
                and len(lower) >= MIN_GEOMETRY_LINE_LENGTH
            ):
                return [upper, lower], "geometry_two_rows"

    return merge_text_order(raw_texts), "text_order_fallback"
