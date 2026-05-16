"""로드아이 (RoadEye) — Streamlit MVP

블랙박스 샘플 영상을 클릭하면 AI가 위반 차량을 감지하고
어떤 위반을 했는지 보여줍니다.
"""
from __future__ import annotations

import base64
import os
import sys
import time
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
from modules.utils import probe_video, seconds_to_timestamp


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
st.caption("📊 데이터 출처: **AI Hub — 교통법규 위반 상황 데이터** (NIA · 경찰청 제공, 2022년 구축)")
st.divider()


# ---------- analyzer cache ----------
@st.cache_resource
def get_analyzer() -> RoadEyeAnalyzer:
    return RoadEyeAnalyzer(model_path="yolov8n.pt", conf=0.35, imgsz=640)


# ---------- video sources ----------
VIDEOS_DIR = Path(__file__).parent / "outputs" / "traffic_violation_videos"
VIDEO_FILES = [VIDEOS_DIR / f"{i}.mp4" for i in range(1, 4)]


@st.cache_data(show_spinner=False)
def get_thumbnail_base64(video_path: str) -> str:
    """Extract a representative frame and encode as base64 JPEG."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Seek to ~30% for a representative frame (avoids black opening frames)
        target_frame = max(0, total // 3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            return ""
    finally:
        cap.release()

    h, w = frame.shape[:2]
    target_w = 320
    scale = target_w / w
    frame = cv2.resize(frame, (target_w, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_click_picker() -> None:
    """Render clickable video cards. Click sets ?video=N via st.query_params."""
    st.markdown("""
    <style>
      .roadeye-thumb {
        aspect-ratio: 16 / 9;
        overflow: hidden;
        border-radius: 10px;
        border: 2px solid #334155;
        background: #0f172a;
        transition: border-color .15s, transform .15s;
        margin-bottom: 6px;
      }
      .roadeye-thumb:hover {
        border-color: #fbbf24;
        transform: translateY(-2px);
      }
      .roadeye-thumb img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
      }
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns(len(VIDEO_FILES))
    for idx, (col, video) in enumerate(zip(cols, VIDEO_FILES), start=1):
        if not video.exists():
            continue
        thumb = get_thumbnail_base64(str(video))
        with col:
            if thumb:
                st.markdown(
                    f'<div class="roadeye-thumb">'
                    f'<img src="data:image/jpeg;base64,{thumb}" alt="영상 {idx}"/>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if st.button(f"📹 영상 {idx} 분석", key=f"pick_video_{idx}",
                         use_container_width=True, type="primary"):
                st.query_params["video"] = str(idx)
                st.rerun()


# ---------- main flow ----------
selected_id_raw = st.query_params.get("video")
selected_id: int | None = None
if selected_id_raw is not None:
    try:
        n = int(selected_id_raw)
        if 1 <= n <= len(VIDEO_FILES) and VIDEO_FILES[n - 1].exists():
            selected_id = n
    except ValueError:
        selected_id = None

if selected_id is None:
    # Show drag-drop landing UI
    missing = [str(v) for v in VIDEO_FILES if not v.exists()]
    if missing:
        st.warning(
            "다음 샘플 영상이 없습니다. 먼저 영상을 준비해주세요:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
        )
        st.stop()

    st.markdown("### 📺 분석할 영상을 선택하세요")
    st.caption("아래 영상 카드를 클릭하면 자동으로 위반 분석이 시작됩니다")
    render_click_picker()
    st.stop()


# ---------- selected video preview ----------
selected_video = str(VIDEO_FILES[selected_id - 1])

top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown(f"### 📹 영상 {selected_id} 분석 중")
with top_right:
    if st.button("🔄 다른 영상 선택", use_container_width=True):
        st.query_params.clear()
        st.rerun()

with open(selected_video, "rb") as _vf:
    st.video(_vf.read(), format="video/mp4")

meta = probe_video(Path(selected_video))
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("길이", f"{meta.duration_sec:.1f} s")
col_b.metric("FPS", f"{meta.fps:.1f}")
col_c.metric("해상도", f"{meta.width}×{meta.height}")
col_d.metric("프레임 수", f"{meta.frame_count}")

st.divider()


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

with st.expander("📊 사용 데이터 안내 — AI Hub 「교통법규 위반 상황 데이터」"):
    st.markdown("""
본 서비스는 **AI Hub**에서 공개한 공공 데이터셋 **「교통법규 위반 상황 데이터」** 를
학습/검증 및 데모용 샘플 영상으로 활용하여 제작되었습니다.

**데이터 개요**
- **출처**: 과학기술정보통신부 · 한국지능정보사회진흥원(NIA) **AI Hub**
- **원천 데이터 제공**: **경찰청** 스마트 국민제보 시스템
- **구축 연도 / 규모**: 2022년 · 이미지 약 **1,490,000장**
- **위반 카테고리(4종)**: 신호위반, 안전모 미착용, 중앙선 침범, 진로변경 위반
- **라이선스**: AI Hub 데이터 이용약관 준수 (학습 목적, 비영리 데모)

**데이터 구성 비율**
| 위반 유형 | 비율 |
|---|---|
| 신호위반 | 45.26% |
| 안전모 미착용 | 1.12% |
| 중앙선 침범 | 30.18% |
| 진로변경 위반 | 23.44% |

**활용 목적**
- 도로 주행 차량의 교통법규 위반 여부 자동 판별 데모
- 룰베이스 + YOLO 객체 탐지 기반 MVP 프로토타입
- 기존 공공 신고 시스템의 효율 증대 가능성 탐색

> 본 페이지의 분석 결과는 데모용이며, 실제 단속·신고의 근거가 될 수 없습니다.
> 데이터 원본은 [AI Hub](https://aihub.or.kr) 에서 확인하실 수 있습니다.
""")
