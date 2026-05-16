"""One-shot script: re-encode sample videos from mp4v to H.264 (browser-compatible).

Backs up originals as *_orig.mp4, replaces in-place.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

# Force UTF-8 stdout/stderr on Windows so ffmpeg output and unicode prints don't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_DIR = Path(__file__).parent / "outputs" / "traffic_violation_videos"
VIDEOS = [VIDEO_DIR / f"{i}.mp4" for i in (1, 2, 3)]


def reencode(src: Path) -> None:
    backup = src.with_name(f"{src.stem}_orig.mp4")
    tmp_out = src.with_name(f"{src.stem}_h264.mp4")

    if not src.exists():
        print(f"[skip] {src} — not found")
        return

    # back up the original if not already backed up
    if not backup.exists():
        shutil.copy2(src, backup)
        print(f"[backup] {src.name} -> {backup.name}")

    cmd = [
        FFMPEG, "-y", "-i", str(src),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",  # drop audio (none expected anyway)
        str(tmp_out),
    ]
    print(f"[encode] {src.name} -> H.264")
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"  ffmpeg failed for {src.name}:")
        print((result.stderr or "")[-800:])
        sys.exit(1)

    # replace original with re-encoded
    src.unlink()
    tmp_out.rename(src)
    print(f"  ok -- {src.name} re-encoded ({src.stat().st_size / 1024:.0f} KB)")


for vp in VIDEOS:
    reencode(vp)

print("\nDone. Originals saved as *_orig.mp4 (gitignored).")
