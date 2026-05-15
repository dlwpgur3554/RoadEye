"""로드아이 (RoadEye) — Streamlit MVP

블랙박스 영상을 업로드하면 AI가 위반 차량을 감지하고
어떤 위반을 했는지 보여줍니다.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

# Force headless OpenCV mode BEFORE importing cv2
os.environ['OPENCV_FFMPEG_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    import cv2
except ImportError as e:
    if "libGL.so.1" in str(e):
        print(f"Warning: OpenCV libGL error - {e}", file=sys.stderr)
        raise
    raise

import streamlit as st

from modules.detector import (
    RoadEyeAnalyzer,
    Violation,
    VIOLATION_LABELS,
    draw_overlay,
)
from modules.ocr import anonymize_snapshot, read_plate
from modules.utils import probe_video, seconds_to_timestamp, tempdir_for_session


st.set_page_config(page_title="로드아이 RoadEye", page_icon="🚨", layout="wide")


# ---------- styling ----------
st.markdown("""
<style>
.violation-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #fff;
}
.badge-lane { background: #ef4444; }
.badge-rapid_lane_change { background: #f97316; }
.badge-illegal_uturn { background: #a855f7; }
.badge-traffic_light { background: #dc2626; }
.plate-chip {
    display:inline-block;
    background:#0f172a;
    color:#fbbf24;
    border:2px solid #fbbf24;
    border-radius:6px;
    padding:6px 12px;
    font-family: 'Courier New', monospace;
    font-weight: 700;
    letter-spacing: 2px;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)


# ---------- header ----------
st.title("🚨 로드아이 (RoadEye)")
st.caption("AI로 [ 도로 위 무법자 ]를 없앤다면 — 블랙박스 영상 위반 자동 감지 MVP")
st.divider()


# ---------- analyzer cache ----------
@st.cache_resource
def get_analyzer() -> RoadEyeAnalyzer:
    return RoadEyeAnalyzer(model_path="yolov8n.pt", conf=0.35, imgsz=640)


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ 분석 설정")
    max_seconds = st.slider("분석 길이 제한 (초)", 5, 120, 30, step=5,
                            help="긴 영상은 처리 시간이 늘어납니다. 데모용으로 30초 정도가 적절합니다.")
    sample_stride = st.slider("프레임 스킵", 1, 5, 2,
                              help="값이 크면 빠르지만 짧은 위반을 놓칠 수 있습니다.")
    do_ocr = st.checkbox("번호판 OCR 수행", value=True)
    do_blur = st.checkbox("주변 차량 블러 (개인정보 보호)", value=True)
    st.markdown("---")
    st.markdown("**감지 가능 위반**")
    for label in VIOLATION_LABELS.values():
        st.markdown(f"- {label}")


# ---------- upload ----------
uploaded = st.file_uploader(
    "블랙박스 영상을 업로드하세요 (mp4 / avi / mov, ~30초 권장)",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded is None:
    st.info("👆 영상을 업로드한 뒤 **분석 시작** 버튼을 눌러주세요.")
    st.stop()


# save upload to a temp file (cv2 needs a path, not a stream)
session_dir = tempdir_for_session() / uuid.uuid4().hex
session_dir.mkdir(parents=True, exist_ok=True)
src_path = session_dir / f"input_{uploaded.name}"
src_path.write_bytes(uploaded.getbuffer())

meta = probe_video(src_path)
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("길이", f"{meta.duration_sec:.1f} s")
col_b.metric("FPS", f"{meta.fps:.1f}")
col_c.metric("해상도", f"{meta.width}×{meta.height}")
col_d.metric("프레임 수", f"{meta.frame_count}")

st.video(str(src_path))

run = st.button("🔍 분석 시작", type="primary", use_container_width=True)
if not run:
    st.stop()


# ---------- run analysis ----------
try:
    analyzer = get_analyzer()
except Exception as exc:
    st.error(f"AI 모델 로딩에 실패했습니다: {exc}")
    st.stop()

analyzer.registry.tracks.clear()
analyzer._violations.clear()
analyzer._reported.clear()

progress = st.progress(0)
status = st.empty()
status.text("영상 분석 중…")
start = time.time()


def on_progress(p: float):
    progress.progress(min(1.0, p))
    status.text(f"영상 분석 중… {int(min(1.0, p) * 100)}%")


violations: list[Violation] = analyzer.analyze_video(
    str(src_path),
    progress=on_progress,
    max_seconds=float(max_seconds),
    sample_stride=int(sample_stride),
)
elapsed = time.time() - start
progress.progress(1.0, text=f"완료 — {elapsed:.1f}초 소요")


# ---------- results ----------
st.divider()
if not violations:
    st.warning("⚠️ 위반이 감지되지 않았습니다. 영상을 바꾸거나 분석 길이/프레임 스킵을 조정해보세요.")
    st.stop()

st.success(f"✅ 위반 {len(violations)}건 감지 — 처리 시간 {elapsed:.1f}초")

# build a quick re-detection cache for the snapshot frames (to allow plate OCR + blur)
for i, v in enumerate(violations, start=1):
        with st.container():
            col_img, col_info = st.columns([1.2, 1])

            with col_img:
                snap = v.snapshot
                if snap is None:
                    st.write("(스냅샷 없음)")
                else:
                    dets = analyzer.detections_on_frame(snap)
                    annotated = draw_overlay(snap, dets, lanes=analyzer.lanes,
                                             highlight_track_id=v.track_id)
                    if do_blur:
                        annotated = anonymize_snapshot(annotated, dets, keep_track_id=v.track_id)
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption=f"감지 시점 스냅샷 ({seconds_to_timestamp(v.time_sec)})",
                             use_container_width=True)

            with col_info:
                st.markdown(f"**{v.type_label}**  #{i}")
                st.markdown(f"**감지 시각**: `{seconds_to_timestamp(v.time_sec)}` (frame {v.frame_idx})")
                st.markdown(f"**차량 ID**: `#{v.track_id}`")
                st.progress(v.confidence / 100.0, text=f"AI 신뢰도 {v.confidence:.0f}%")
                if v.note:
                    st.caption(v.note)

                plate_text = "—"
                if do_ocr and v.snapshot is not None:
                    try:
                        plate = read_plate(v.snapshot, v.bbox)
                        if plate is not None:
                            plate_text = plate.text
                    except Exception as exc:
                        plate_text = f"OCR 오류: {exc}"
                st.markdown("**차량 번호판**")
                st.markdown(f'`{plate_text}`')
st.divider()
st.caption("⚠️ MVP 데모: 위반 룰은 룰베이스 휴리스틱이며, 정확도는 데모 수준입니다. "
           "실제 신고 자동화에는 추가 검증 단계가 필요합니다.")
