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


# ---------- get sample videos ----------
SAMPLES_DIR = Path(__file__).parent / "outputs" / "traffic_violation_videos"
CATEGORIES = {
    "신호위반": "🚦",
    "중앙선침범": "↔️",
    "안전모미착용": "🚫",
    "진로변경위반": "🔄",
}


def get_sample_videos():
    """Returns dict of category -> list of video paths"""
    videos = {}
    if SAMPLES_DIR.exists():
        for cat in CATEGORIES.keys():
            cat_dir = SAMPLES_DIR / cat
            if cat_dir.exists():
                videos[cat] = sorted(cat_dir.glob("*.mp4"))
    return videos


# ---------- sample videos UI ----------
st.markdown("### 📺 샘플 영상 선택")
st.caption("아래 샘플 영상을 클릭하면 자동으로 분석됩니다")

samples = get_sample_videos()
if not samples or all(not v for v in samples.values()):
    st.warning("샘플 영상이 없습니다. 먼저 영상을 생성하세요.")
    st.stop()

# Create tabs for each category
tabs = st.tabs([f"{CATEGORIES.get(cat, '')} {cat}" for cat in CATEGORIES.keys()])

selected_video = None
for tab, cat in zip(tabs, CATEGORIES.keys()):
    with tab:
        videos = samples.get(cat, [])
        if not videos:
            st.info(f"{cat} 샘플이 없습니다")
            continue
        
        cols = st.columns(min(3, len(videos)))
        for idx, video_path in enumerate(videos):
            with cols[idx % len(cols)]:
                if st.button(f"📹 영상 {idx+1}", key=f"select_{cat}_{idx}", use_container_width=True):
                    st.session_state["selected_video"] = str(video_path)
                    selected_video = str(video_path)

# Get selected video from session state
if "selected_video" not in st.session_state:
    st.info("👆 분석할 영상을 선택해주세요")
    st.stop()

selected_video = st.session_state["selected_video"]

# Show selected video preview
st.divider()
st.markdown("### 📹 선택된 영상")
st.video(selected_video)

meta = probe_video(Path(selected_video))
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("길이", f"{meta.duration_sec:.1f} s")
col_b.metric("FPS", f"{meta.fps:.1f}")
col_c.metric("해상도", f"{meta.width}×{meta.height}")
col_d.metric("프레임 수", f"{meta.frame_count}")

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
    selected_video,
    progress=on_progress,
    max_seconds=120.0,
    sample_stride=2,
)
elapsed = time.time() - start
progress.progress(1.0, text=f"완료 — {elapsed:.1f}초 소요")


# ---------- results ----------
st.divider()
if not violations:
    st.warning("⚠️ 위반이 감지되지 않았습니다.")
    st.stop()

st.success(f"✅ 위반 {len(violations)}건 감지 — 처리 시간 {elapsed:.1f}초")

# Display violations
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
            if v.snapshot is not None:
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
