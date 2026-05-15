"""Convert JPG/JPEG/PNG image sequences into MP4 videos at a fixed FPS.

This script is Unicode-safe on Windows for paths containing Korean or other
non-ASCII characters.

Usage:
    python convert_jpgs_to_videos_unicode.py --source "01.데이터\2.Validation\원천데이터\VS\A\BLUE" \
        --output outputs/jpg_videos\A_BLUE --num-videos 1 --fps 5
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert image sequences into MP4 videos at a fixed frame rate."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source image folder containing JPG/JPEG/PNG files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/jpg_videos",
        help="Output folder for generated MP4 files.",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=5,
        help="Approximate number of MP4 files to generate.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second for each generated video.",
    )
    return parser.parse_args()


def gather_images(source_dir: Path) -> list[Path]:
    allowed_ext = {".jpg", ".jpeg", ".png"}
    return sorted(
        [path for path in source_dir.rglob("*") if path.suffix.lower() in allowed_ext]
    )


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_video(images: list[Path], output_path: Path, fps: float) -> None:
    first_frame = imread_unicode(images[0])
    if first_frame is None:
        raise RuntimeError(f"Cannot read image: {images[0]}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path}")

    for image_path in images:
        frame = imread_unicode(image_path)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists() or not source_dir.is_dir():
        raise SystemExit(f"Source folder not found: {source_dir}")

    images = gather_images(source_dir)
    if not images:
        raise SystemExit(f"No JPG/JPEG/PNG images found in: {source_dir}")

    num_videos = max(1, args.num_videos)
    chunk_size = ceil(len(images) / num_videos)

    print(f"Found {len(images)} images. Generating {num_videos} videos at {args.fps} FPS.")

    for index in range(num_videos):
        start = index * chunk_size
        end = start + chunk_size
        chunk = images[start:end]
        if not chunk:
            break

        output_path = output_dir / f"video_{index + 1:02d}.mp4"
        print(f"Writing {output_path} with {len(chunk)} frames...")
        write_video(chunk, output_path, args.fps)

    print("Finished generating videos.")


if __name__ == "__main__":
    main()
