"""Video I/O and clip extraction helpers."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe_video(path: str | os.PathLike) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMeta(fps=fps, width=width, height=height, frame_count=frame_count)
    finally:
        cap.release()


def iter_frames(path: str | os.PathLike, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame). stride>1 skips frames for speed."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def save_clip(
    src_path: str | os.PathLike,
    out_path: str | os.PathLike,
    start_frame: int,
    end_frame: int,
    transform=None,
) -> str:
    """Save a subclip [start_frame, end_frame) to out_path. Optional per-frame transform(frame)->frame."""
    meta = probe_video(src_path)
    start_frame = max(0, start_frame)
    end_frame = min(meta.frame_count, end_frame)
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")

    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, meta.fps, (meta.width, meta.height))
    try:
        for _ in range(end_frame - start_frame):
            ok, frame = cap.read()
            if not ok:
                break
            if transform is not None:
                frame = transform(frame)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return str(out_path)


def save_frame_image(frame: np.ndarray, out_path: str | os.PathLike) -> str:
    cv2.imwrite(str(out_path), frame)
    return str(out_path)


def seconds_to_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def tempdir_for_session() -> Path:
    base = Path(tempfile.gettempdir()) / "roadeye"
    base.mkdir(parents=True, exist_ok=True)
    return base


def clip_window(center_frame: int, fps: float, pre_sec: float = 3.0, post_sec: float = 3.0) -> tuple[int, int]:
    pre = int(pre_sec * fps)
    post = int(post_sec * fps)
    return max(0, center_frame - pre), center_frame + post


def convert_jpg_sequence_to_mp4(jpg_folder: str | os.PathLike, output_mp4: str | os.PathLike, fps: float = 5.0) -> bool:
    """Convert a sequence of JPG images in a folder to an MP4 video."""
    jpg_folder = Path(jpg_folder)
    images = sorted(jpg_folder.glob("*.jpg"))
    if not images:
        return False
    # Read first image to get dimensions
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        return False
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))
    try:
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                video.write(frame)
    finally:
        video.release()
    return True
