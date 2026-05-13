"""YOLO-based vehicle detection + rule-based violation analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # allow import-time when ultralytics absent (e.g., during static checks)
    YOLO = None  # type: ignore

from .tracker import TrackPoint, TrackRegistry, VehicleTrack


# COCO classes we care about
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COCO_TRAFFIC_LIGHT_CLASS = 9
COCO_PERSON_CLASS = 0


VIOLATION_LABELS = {
    "lane": "차선 위반",
    "rapid_lane_change": "급차선 변경",
    "illegal_uturn": "불법 유턴",
    "traffic_light": "신호 위반",
}


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None


@dataclass
class Violation:
    type_key: str            # one of VIOLATION_LABELS keys
    type_label: str          # Korean label
    track_id: int
    frame_idx: int
    time_sec: float
    confidence: float        # 0-100
    bbox: tuple[int, int, int, int]
    snapshot: np.ndarray | None = None  # BGR frame at violation moment
    note: str = ""

    def summary(self) -> str:
        return f"{self.type_label} (#{self.track_id} @ {self.time_sec:.1f}s, {self.confidence:.0f}%)"


@dataclass
class LaneLines:
    """Two main lane lines (left and right of ego lane) in image coords."""
    left: tuple[int, int, int, int] | None = None   # x1,y1,x2,y2
    right: tuple[int, int, int, int] | None = None

    def x_at(self, y: int, side: str) -> float | None:
        line = self.left if side == "left" else self.right
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if y2 == y1:
            return float(x1)
        t = (y - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    def lane_of(self, x: float, y: int) -> int:
        """Return -1 unknown, 0 left of ego, 1 ego, 2 right of ego."""
        lx = self.x_at(y, "left")
        rx = self.x_at(y, "right")
        if lx is None and rx is None:
            return -1
        if lx is not None and x < lx:
            return 0
        if rx is not None and x > rx:
            return 2
        return 1


def detect_lanes(frame: np.ndarray) -> LaneLines:
    """Cheap Hough-based ego-lane line estimation. Returns LaneLines."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)

    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.45), int(h * 0.6)),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.95), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=80)
    if lines is None:
        return LaneLines()

    left_pts: list[tuple[int, int]] = []
    right_pts: list[tuple[int, int]] = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4:
            continue
        if slope < 0 and x1 < w * 0.55 and x2 < w * 0.55:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and x1 > w * 0.45 and x2 > w * 0.45:
            right_pts.extend([(x1, y1), (x2, y2)])

    def fit(pts: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        if len(pts) < 4:
            return None
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        # fit x = m*y + b so vertical-ish lines work
        m, b = np.polyfit(ys, xs, 1)
        y1, y2 = h, int(h * 0.62)
        return int(m * y1 + b), y1, int(m * y2 + b), y2

    return LaneLines(left=fit(left_pts), right=fit(right_pts))


def classify_traffic_light(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """Return 'red'/'green'/'yellow'/'unknown' from the traffic light ROI."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    masks = {
        "red": cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        | cv2.inRange(hsv, (160, 100, 100), (180, 255, 255)),
        "yellow": cv2.inRange(hsv, (15, 100, 100), (35, 255, 255)),
        "green": cv2.inRange(hsv, (40, 80, 80), (90, 255, 255)),
    }
    scores = {k: int(np.count_nonzero(m)) for k, m in masks.items()}
    best = max(scores, key=scores.get)
    if scores[best] < 30:
        return "unknown"
    return best


class RoadEyeAnalyzer:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 640) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Run `pip install ultralytics`.")
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.registry = TrackRegistry()
        self.lanes: LaneLines = LaneLines()
        self._light_state: str = "unknown"
        self._light_bbox: tuple[int, int, int, int] | None = None
        self._violations: list[Violation] = []
        self._reported: set[tuple[int, str]] = set()  # (track_id, type_key) once each

    # ---- public API ----
    def analyze_video(
        self,
        src_path: str,
        fps_hint: float | None = None,
        progress=None,
        max_seconds: float | None = None,
        sample_stride: int = 2,
    ) -> list[Violation]:
        from .utils import iter_frames, probe_video  # local import to avoid cycle

        meta = probe_video(src_path)
        fps = fps_hint or meta.fps or 30.0
        max_frames = int(max_seconds * fps) if max_seconds else meta.frame_count

        # YOLO's built-in tracker handles ID assignment; we run frame-by-frame.
        cap = cv2.VideoCapture(src_path)
        idx = 0
        try:
            while idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % sample_stride == 0:
                    self._process_frame(frame, idx, fps)
                idx += 1
                if progress is not None and meta.frame_count > 0:
                    progress(min(1.0, idx / min(max_frames, meta.frame_count)))
        finally:
            cap.release()
        return self._violations

    def detections_on_frame(self, frame: np.ndarray) -> list[Detection]:
        """One-shot detection (no tracking, no rules). Used for preview overlays."""
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        return self._results_to_detections(results)

    # ---- internals ----
    def _process_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> None:
        # refresh lane estimate every ~1s
        if frame_idx % max(1, int(fps)) == 0:
            self.lanes = detect_lanes(frame)

        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )
        dets = self._results_to_detections(results)

        # traffic light state
        light = next((d for d in dets if d.cls_id == COCO_TRAFFIC_LIGHT_CLASS), None)
        if light is not None:
            self._light_state = classify_traffic_light(frame, light.bbox)
            self._light_bbox = light.bbox

        for d in dets:
            if d.cls_id not in COCO_VEHICLE_CLASSES or d.track_id is None:
                continue
            x1, y1, x2, y2 = d.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bottom_y = y2
            lane = self.lanes.lane_of(cx, bottom_y)
            pt = TrackPoint(frame=frame_idx, cx=cx, cy=cy, bx1=x1, by1=y1, bx2=x2, by2=y2)
            track = self.registry.update(d.track_id, d.cls_name, pt, lane)
            self._evaluate_rules(track, frame, frame_idx, fps)

        self.registry.prune(frame_idx)

    def _results_to_detections(self, results) -> list[Detection]:
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(clss)
        names = r.names if hasattr(r, "names") else {}
        for box, conf, cls, tid in zip(xyxy, confs, clss, ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            out.append(Detection(
                cls_id=int(cls),
                cls_name=str(names.get(int(cls), str(cls))),
                conf=float(conf),
                bbox=(x1, y1, x2, y2),
                track_id=int(tid) if tid is not None else None,
            ))
        return out

    def _record(self, v: Violation, frame: np.ndarray) -> None:
        key = (v.track_id, v.type_key)
        if key in self._reported:
            return
        self._reported.add(key)
        v.snapshot = frame.copy()
        self._violations.append(v)

    def _evaluate_rules(self, track: VehicleTrack, frame: np.ndarray, frame_idx: int, fps: float) -> None:
        # need a minimum history before evaluating
        if len(track.history) < 6:
            return

        # 1) Lane violation: any lane index transition involving 0<->1 or 1<->2
        if track.lane_crossings(window=int(fps * 1.5)) >= 1:
            # require it to have been in ego lane at some point — filters out side-of-road parked cars
            recent_lanes = [l for l in list(track.lane_history)[-int(fps * 2):] if l >= 0]
            if 1 in recent_lanes and (0 in recent_lanes or 2 in recent_lanes):
                v = Violation(
                    type_key="lane",
                    type_label=VIOLATION_LABELS["lane"],
                    track_id=track.track_id,
                    frame_idx=frame_idx,
                    time_sec=frame_idx / fps,
                    confidence=72.0,
                    bbox=(int(track.history[-1].bx1), int(track.history[-1].by1),
                          int(track.history[-1].bx2), int(track.history[-1].by2)),
                    note="차량 중심이 차선 라인을 가로질렀습니다.",
                )
                self._record(v, frame)

        # 2) Rapid lane change: 2+ crossings within ~2s
        if track.lane_crossings(window=int(fps * 2.0)) >= 2:
            v = Violation(
                type_key="rapid_lane_change",
                type_label=VIOLATION_LABELS["rapid_lane_change"],
                track_id=track.track_id,
                frame_idx=frame_idx,
                time_sec=frame_idx / fps,
                confidence=64.0,
                bbox=(int(track.history[-1].bx1), int(track.history[-1].by1),
                      int(track.history[-1].bx2), int(track.history[-1].by2)),
                note="2초 이내에 차선을 2회 이상 가로질렀습니다.",
            )
            self._record(v, frame)

        # 3) Illegal U-turn: vertical direction reversal + horizontal sweep
        if track.heading_flip(lookback=int(fps * 3.0)):
            recent = list(track.history)[-int(fps * 3.0):]
            if recent:
                dx = abs(recent[-1].cx - recent[0].cx)
                if dx > frame.shape[1] * 0.15:
                    v = Violation(
                        type_key="illegal_uturn",
                        type_label=VIOLATION_LABELS["illegal_uturn"],
                        track_id=track.track_id,
                        frame_idx=frame_idx,
                        time_sec=frame_idx / fps,
                        confidence=55.0,
                        bbox=(int(track.history[-1].bx1), int(track.history[-1].by1),
                              int(track.history[-1].bx2), int(track.history[-1].by2)),
                        note="진행 방향이 반전되며 큰 횡방향 이동이 감지되었습니다.",
                    )
                    self._record(v, frame)

        # 4) Traffic light violation: light=red AND vehicle moves strongly toward camera in lower frame
        if self._light_state == "red":
            disp = track.displacement_y(n=int(fps * 1.0))
            last = track.history[-1]
            if disp > 25 and last.by2 > frame.shape[0] * 0.6:
                v = Violation(
                    type_key="traffic_light",
                    type_label=VIOLATION_LABELS["traffic_light"],
                    track_id=track.track_id,
                    frame_idx=frame_idx,
                    time_sec=frame_idx / fps,
                    confidence=50.0,
                    bbox=(int(last.bx1), int(last.by1), int(last.bx2), int(last.by2)),
                    note="신호등이 적색인 상태에서 진행이 감지되었습니다 (휴리스틱).",
                )
                self._record(v, frame)


def draw_overlay(
    frame: np.ndarray,
    detections: Iterable[Detection],
    lanes: LaneLines | None = None,
    highlight_track_id: int | None = None,
) -> np.ndarray:
    out = frame.copy()
    if lanes is not None:
        for line in (lanes.left, lanes.right):
            if line is not None:
                cv2.line(out, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 3)
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        color = (0, 0, 255) if d.track_id == highlight_track_id else (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.cls_name}"
        if d.track_id is not None:
            label += f"#{d.track_id}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def process_video_with_yolo(input_video: str, output_video: str, model_path: str = "yolov8n.pt"):
    """Process a video with YOLO detection and save annotated video."""
    analyzer = RoadEyeAnalyzer(model_path=model_path)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = analyzer.detections_on_frame(frame)
        annotated = draw_overlay(frame, detections)
        out.write(annotated)
    cap.release()
    out.release()
