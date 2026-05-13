"""Per-vehicle trajectory history. YOLO's internal tracker assigns IDs; we keep the time series."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class TrackPoint:
    frame: int
    cx: float
    cy: float
    bx1: float
    by1: float
    bx2: float
    by2: float


@dataclass
class VehicleTrack:
    track_id: int
    cls_name: str
    history: Deque[TrackPoint] = field(default_factory=lambda: deque(maxlen=240))  # ~8s @ 30fps
    lane_history: Deque[int] = field(default_factory=lambda: deque(maxlen=240))
    last_seen_frame: int = -1

    def add(self, point: TrackPoint, lane: int) -> None:
        self.history.append(point)
        self.lane_history.append(lane)
        self.last_seen_frame = point.frame

    def displacement_y(self, n: int = 10) -> float:
        """Vertical displacement over the last n frames. Positive = moving down (toward camera)."""
        if len(self.history) < 2:
            return 0.0
        recent = list(self.history)[-n:]
        return recent[-1].cy - recent[0].cy

    def heading_flip(self, lookback: int = 30) -> bool:
        """True if vertical motion direction reversed within lookback frames (U-turn cue)."""
        if len(self.history) < lookback:
            return False
        recent = list(self.history)[-lookback:]
        half = len(recent) // 2
        first_dy = recent[half - 1].cy - recent[0].cy
        second_dy = recent[-1].cy - recent[half].cy
        return first_dy * second_dy < 0 and abs(first_dy) > 6 and abs(second_dy) > 6

    def lane_crossings(self, window: int = 60) -> int:
        """Count lane index changes in the last window frames (excludes -1 unknowns)."""
        if len(self.lane_history) < 2:
            return 0
        recent = [l for l in list(self.lane_history)[-window:] if l >= 0]
        if len(recent) < 2:
            return 0
        crossings = 0
        for a, b in zip(recent[:-1], recent[1:]):
            if a != b:
                crossings += 1
        return crossings


class TrackRegistry:
    """Holds VehicleTrack objects keyed by YOLO track_id."""

    def __init__(self) -> None:
        self.tracks: dict[int, VehicleTrack] = {}

    def update(self, track_id: int, cls_name: str, point: TrackPoint, lane: int) -> VehicleTrack:
        track = self.tracks.get(track_id)
        if track is None:
            track = VehicleTrack(track_id=track_id, cls_name=cls_name)
            self.tracks[track_id] = track
        track.add(point, lane)
        return track

    def prune(self, current_frame: int, ttl_frames: int = 90) -> None:
        stale = [tid for tid, t in self.tracks.items() if current_frame - t.last_seen_frame > ttl_frames]
        for tid in stale:
            del self.tracks[tid]
