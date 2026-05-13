"""License plate OCR (EasyOCR) + privacy blur for non-violating vehicles."""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import easyocr  # type: ignore
except ImportError:
    easyocr = None  # type: ignore


# 한국 번호판 패턴: 신형 3자리(예: 12가1234), 구형 2자리(예: 12가1234), 신형 영업/특수도 포함
KOREAN_PLATE_RE = re.compile(r"\d{2,3}[가-힣]\d{4}")
KOREAN_PLATE_LOOSE_RE = re.compile(r"\d{2,3}\s?[가-힣]\s?\d{3,4}")


@dataclass
class PlateResult:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # within original frame


class PlateReader:
    """Lazy-init EasyOCR reader. Singleton-style usage from app code."""
    _reader = None

    @classmethod
    def get(cls):
        if easyocr is None:
            raise RuntimeError("easyocr is not installed.")
        if cls._reader is None:
            cls._reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
        return cls._reader


def _normalize_plate(text: str) -> str | None:
    cleaned = text.replace(" ", "").replace("-", "")
    m = KOREAN_PLATE_RE.search(cleaned)
    if m:
        return m.group(0)
    m = KOREAN_PLATE_LOOSE_RE.search(text)
    if m:
        return m.group(0).replace(" ", "")
    return None


def read_plate(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> PlateResult | None:
    """Run OCR on the lower half of the vehicle bbox (where plates typically sit)."""
    if easyocr is None:
        return None
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    # focus on bottom 55% of bbox where plate is most likely
    plate_y1 = int(y1 + (y2 - y1) * 0.45)
    crop = frame[plate_y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # upscale small crops to help OCR
    if crop.shape[0] < 80:
        scale = 80 / max(1, crop.shape[0])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    reader = PlateReader.get()
    results = reader.readtext(crop, detail=1, paragraph=False)
    best: PlateResult | None = None
    for box, text, conf in results:
        plate = _normalize_plate(text)
        if plate is None:
            continue
        if best is None or conf > best.confidence:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            best = PlateResult(
                text=plate,
                confidence=float(conf),
                bbox=(x1 + min(xs), plate_y1 + min(ys), x1 + max(xs), plate_y1 + max(ys)),
            )
    return best


def blur_region(frame: np.ndarray, bbox: tuple[int, int, int, int], ksize: int = 35) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    k = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame


def anonymize_snapshot(
    frame: np.ndarray,
    detections,
    keep_track_id: int | None,
    person_class_id: int = 0,
) -> np.ndarray:
    """Blur every vehicle except `keep_track_id`, plus all people."""
    out = frame.copy()
    for d in detections:
        if d.cls_id == person_class_id:
            blur_region(out, d.bbox, ksize=45)
            continue
        if d.track_id is not None and d.track_id == keep_track_id:
            continue
        blur_region(out, d.bbox, ksize=35)
    return out
