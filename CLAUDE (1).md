# 로드아이 (RoadEye) MVP — Development Spec

> Claude Code 작업 명세서. 이 파일을 읽고 단계별로 진행해주세요.

---

## 📋 1. 프로젝트 컨텍스트

- **대회**: 2026 멋쟁이사자처럼 대학 14기 아이디어톤
- **마감**: 2026년 5월 18일 15:00 (남은 시간: 약 5일)
- **컨셉**: AI로 **[ 도로 위 무법자 ]** 를 없앤다면 — NPU 기반 AI 공익 신고 자동화 시스템
- **서비스명**: 로드아이 (RoadEye, 가칭)
- **개발 인력**: 학생 1~2명, 풀타임

## 🎯 2. MVP 목표

**한 줄 정의**: 블랙박스 영상을 업로드하면 AI가 위반 차량을 자동 감지하고, 번호판 추출·주변 블러 처리 후 안전신문고 신고서를 자동 작성하는 웹 데모.

**최종 산출물**:
1. **배포된 웹 앱** (Streamlit Cloud) — 심사위원이 QR로 접근 가능
2. **GitHub 레포지토리** (코드 공개)
3. **시연 영상 1편** (mp4, 100초 영상에 삽입용)

**평가 기준 (멋사 14기 운영 가이드)**:
- 개발은 가산점, 핵심 기능 1개 이상 구현·배포 시 가점
- 작동 여부만 판단, 기능 수·퀄리티는 평가 X
- → **"작동하는 데모"** 1개면 충분. 완벽함보다 작동을 우선.

---

## 📦 3. 기술 스택

| 영역 | 라이브러리 | 비고 |
|---|---|---|
| Python | 3.10+ | 필수 |
| 객체 감지 | `ultralytics` (YOLOv8) | 사전학습 모델, 추가 학습 X |
| 영상 처리 | `opencv-python` | 영상 입출력, 블러 |
| OCR | `easyocr` | 한국어 번호판 인식 (`['ko', 'en']`) |
| 웹 UI | `streamlit` | 빠른 프로토타입 + 무료 배포 |
| PDF 생성 | `reportlab` or `weasyprint` | 신고서 PDF |
| 영상 다운로드 | `yt-dlp` | 샘플 영상 수집용 (개발 단계만) |

### `requirements.txt`
```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
easyocr>=1.7.0
streamlit>=1.28.0
reportlab>=4.0.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## 📁 4. 프로젝트 구조

```
roadeye-mvp/
├── app.py                  # Streamlit 메인 진입점
├── modules/
│   ├── __init__.py
│   ├── detector.py         # YOLO 차량·차선 감지 + 위반 룰
│   ├── tracker.py          # 차량 추적
│   ├── ocr.py              # 번호판 OCR + 블러
│   ├── report.py           # 신고서 PDF 생성
│   └── utils.py            # 공통 유틸 (영상 로딩, 클립 자르기 등)
├── samples/                # 샘플 영상 (gitignore)
├── outputs/                # 분석 결과 임시 저장 (gitignore)
├── .streamlit/
│   └── config.toml         # Streamlit 설정
├── .gitignore
├── requirements.txt
├── README.md
└── packages.txt            # Streamlit Cloud 시스템 패키지 (필요 시)
```

---

## 🛠 5. 개발 단계 (5일 일정)

### Day 1 (5/13) — 환경 + YOLO 기본 동작 확인

**목표**: 영상 1개를 YOLO에 통과시켜 객체 박스가 그려진 결과 영상 생성.

**작업**:
1. 가상환경 + 라이브러리 설치
2. 한문철TV 등에서 샘플 영상 5~10개 다운로드 (yt-dlp)
   - 위반 유형 다양하게: 차선 위반, 꼬리물기, 보행자 보호 위반, 보복운전
3. `detector.py` 에 기본 함수 작성:
   - `detect_objects(frame) -> list` : 프레임 1장에서 차량·사람·번호판 박스 반환
4. 영상 한 개 통과시켜 결과 영상 저장 테스트

**완료 기준**: `output.mp4` 에 차량 박스가 그려진 영상이 생성됨.

**참고 코드**:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # nano 모델, 빠름
results = model(frame)
annotated = results[0].plot()
```

---

### Day 2 (5/14) — 위반 감지 룰 + 클립 자동 추출

**목표**: 영상에서 위반 시점을 자동 감지하고, ±10초 클립 자동 저장.

**작업**:
1. **차선 감지** (`detector.py`):
   - OpenCV Hough Line Transform 또는 YOLOv8 segmentation 모델
   - 차선 위치(좌·우)를 매 프레임 추적
2. **차량 추적** (`tracker.py`):
   - YOLOv8 자체 트래킹 사용 (`model.track()`)
   - 각 차량에 ID 부여, 위치 시계열 보관
3. **위반 룰 작성** (`detector.py`):
   - **차선 위반**: 차량 중심이 차선 라인을 가로지름 (좌→우 또는 우→좌)
   - **꼬리물기**: 앞차와의 픽셀 거리 < 임계값 (간이 기준, 정확도 X 의도)
   - **급차선 변경**: 짧은 시간(2초) 내 차선 가로지름 + 깜빡이 미감지(시각적 룰 어려우면 시간 기준만)
4. **클립 자동 추출** (`utils.py`):
   - 위반 시점 ±10초 (총 20초) 클립 mp4로 저장
   - 파일명: `violation_<type>_<timestamp>.mp4`

**완료 기준**: 샘플 영상 1개에서 위반 N건 감지되고 클립이 `outputs/` 폴더에 저장됨.

**주의사항**:
- 위반 룰은 **완벽한 정확도 X**. 데모용으로 50~70% 인식되면 충분.
- 한국 도로 환경 학습 데이터가 없으니 룰베이스로 처리.
- AI 신뢰도(confidence)는 룰별 가중치로 산출 (예: "차선 위반 87%").

---

### Day 3 (5/15) — 번호판 OCR + 블러 + 신고서 PDF

**목표**: 위반 차량 번호판 추출 + 주변 차량 블러 + 신고서 PDF 생성.

**작업**:
1. **번호판 OCR** (`ocr.py`):
   - EasyOCR 인스턴스: `easyocr.Reader(['ko', 'en'])`
   - 위반 차량 박스 안에서 번호판 후보 영역 OCR
   - 한국 번호판 패턴 정규식 필터링: `r'\d{2,3}[가-힣]\d{4}'`
2. **블러 처리** (`ocr.py`):
   - 위반 차량 외 다른 차량 박스에 가우시안 블러 적용
   - `cv2.GaussianBlur(roi, (51,51), 0)`
   - 보행자 얼굴도 블러
3. **신고서 PDF 생성** (`report.py`):
   - reportlab으로 안전신문고 양식 모방
   - 항목: 위반 유형, 일시, 장소(GPS 데모로 임의 입력), 차량 번호, AI 신뢰도, 첨부 영상 캡처
   - 출력: `report_<timestamp>.pdf`

**완료 기준**: 위반 클립 → 번호판 추출 + 블러된 미리보기 이미지 + 신고서 PDF 1개 자동 생성.

**OCR 정확도 팁**:
- 한국 번호판 폰트는 표준이라 인식률 70~85%
- 정확도 낮으면 박스 확대 후 OCR 재시도
- 다중 후보 → 정규식 필터링

---

### Day 4 (5/16) — Streamlit UI + 무료 배포

**목표**: 사용자가 영상 업로드하면 자동 분석되는 웹 앱을 Streamlit Cloud에 배포.

**작업**:
1. **`app.py` 작성** (~150줄):
   - 헤더: 로드아이 소개
   - 파일 업로더 (`st.file_uploader`)
   - 진행바 (`st.progress`)
   - 결과 카드 (`st.container(border=True)`):
     - 미리보기 비디오 (`st.video`)
     - 위반 유형 / 시각 / 번호판 / 신뢰도
     - "신고서 다운로드" 버튼
2. **Streamlit Cloud 배포**:
   - GitHub 레포 푸시 (Public)
   - https://share.streamlit.io 접속
   - "New app" → 레포 선택 → 자동 배포
   - 배포 URL 확보
3. **QR 코드 생성**:
   - https://qr.io 또는 `qrcode` 라이브러리
   - PNG로 저장 (발표 PPT 삽입용)

**완료 기준**: 배포된 URL로 접속해 영상 업로드 → 결과 표시까지 정상 작동.

**Streamlit 배포 주의**:
- `requirements.txt` 정확히 작성
- 시스템 패키지 필요 시 `packages.txt`에 추가:
  ```
  libgl1
  libglib2.0-0
  ```
- 메모리 한계 1GB → YOLOv8n (nano) 사용 권장
- 영상 처리 시간 길면 진행바·진행 메시지 필수

**UI 시안 (참고)**:
```python
st.title("🚨 로드아이")
st.markdown("AI로 [ 도로 위 무법자 ]를 없앤다면")
st.markdown("---")

uploaded = st.file_uploader("블랙박스 영상을 업로드하세요", type=['mp4','avi','mov'])

if uploaded:
    with st.spinner("AI가 영상을 분석 중입니다..."):
        violations = analyze(uploaded)
    
    st.success(f"✅ 분석 완료 — 위반 {len(violations)}건 감지")
    
    for i, v in enumerate(violations):
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.video(v['clip'])
            with col2:
                st.markdown(f"**위반 유형**: {v['type']}")
                st.markdown(f"**감지 시각**: {v['time']}")
                st.markdown(f"**차량 번호**: `{v['plate']}`")
                st.progress(v['confidence'] / 100, text=f"AI 신뢰도 {v['confidence']}%")
                st.download_button(
                    "📄 신고서 PDF 다운로드",
                    data=v['report_pdf'],
                    file_name=f"신고서_{i}.pdf",
                    key=f"dl_{i}"
                )
```

---

### Day 5 (5/17) — 테스트 + 시연 영상 + 폴리싱

**작업**:
1. **다양한 영상으로 테스트** (5~10개)
   - 잘 작동하는 영상 2~3개 골라 발표용으로 확정
2. **시연 영상 녹화** (1~2분):
   - 화면 녹화 도구로 전체 흐름 시연
   - 영상 업로드 → 분석 → 결과 → 신고서 다운로드
3. **README.md 작성**:
   - 프로젝트 소개, 기술 스택, 실행 방법, 데모 URL, 팀 소개
4. **최종 점검**:
   - 배포 URL 정상 접속
   - QR 코드 인쇄·PPT 삽입
   - 100초 영상에 시연 클립 삽입

---

## 🆘 6. 백업 플랜 (시간 부족 시 단계별 축소)

진행이 늦어질 경우 우선순위에 따라 축소:

### 🟢 Level 1 (필수, Day 3까지 완료해야 함)
- YOLO 객체 감지 작동
- 영상 1개 입력 → 객체 박스 결과 영상 출력
- **이것만 있어도 가산점 받을 수 있음**

### 🟡 Level 2 (가능하면, Day 4까지)
- 위반 감지 룰 (차선 위반 정도)
- 클립 자동 추출
- 간단한 Streamlit UI (로컬 실행)

### 🔴 Level 3 (시간 충분 시)
- 번호판 OCR + 블러
- 신고서 PDF
- Streamlit Cloud 배포 + QR

---

## 🚀 7. Streamlit Cloud 배포 체크리스트

- [ ] GitHub 레포 Public으로 설정
- [ ] `requirements.txt` 작성 (버전 명시)
- [ ] `packages.txt` 작성 (libgl1, libglib2.0-0)
- [ ] `.streamlit/config.toml` 작성 (선택)
- [ ] 메인 파일이 `app.py`인지 확인
- [ ] https://share.streamlit.io 접속 → "New app"
- [ ] 레포 선택 → 자동 빌드 (5~10분)
- [ ] 빌드 로그에서 에러 확인
- [ ] 배포 URL 메모

**자주 발생하는 에러**:
- `ImportError: libGL.so.1` → `packages.txt`에 `libgl1` 추가
- 메모리 초과 → YOLOv8s 대신 YOLOv8n 사용
- 영상 처리 시간 초과 → 영상 길이 60초 이하로 제한

---

## 📚 8. 주요 참고 자료

- Ultralytics YOLOv8 문서: https://docs.ultralytics.com
- Streamlit 문서: https://docs.streamlit.io
- Streamlit Cloud 배포: https://docs.streamlit.io/streamlit-community-cloud
- EasyOCR GitHub: https://github.com/JaidedAI/EasyOCR
- 안전신문고: https://www.safetyreport.go.kr (신고 양식 참고용)
- 한문철TV (샘플 영상): https://youtube.com/@hanmunchultv

---

## 🎬 9. 샘플 영상 수집 (Day 1 작업)

`yt-dlp` 한 줄로 다운로드:

```bash
# 단일 영상
yt-dlp -f "best[height<=720][ext=mp4]" -o "samples/%(title)s.%(ext)s" "URL"

# 짧게 잘라서 받기 (위반 발생 부분 30초만)
yt-dlp -f "best[height<=720][ext=mp4]" \
       --download-sections "*0:30-1:00" \
       -o "samples/%(title)s.%(ext)s" "URL"
```

**저작권 주의**: 데모·교육 용도이고 발표에서 출처 명시 시 OK. 배포 영상에는 본인 촬영 영상 또는 저작권 자유 영상 사용 권장. Streamlit Cloud에는 영상 자체를 올리지 말고, **사용자가 업로드**하도록 설계.

---

## 🎯 10. 발표용 메시지 (개발 부분)

이 데모를 IR DECK과 100초 영상에서 어떻게 활용할지:

**100초 영상**:
- 0:30~0:50 구간에 라이브 데모 시연 클립 삽입
- "AI가 위반을 어떻게 감지하는지 직접 보세요"

**IR DECK (5분 발표)**:
- 데모 슬라이드에 QR 코드 + 배포 URL
- "심사위원께서 직접 영상을 업로드해 체험해보실 수 있습니다"

**발표 멘트**:
> "현재 데모 화면은 QR 코드로 직접 접속하실 수 있습니다. AI 모델은 차량용 NPU에 탑재 가능하도록 경량화되어 있으며, 이스라엘 Nexar가 검증한 모델 구조와 동일합니다."

---

## ❓ 11. 우선순위 명확화

**막혔을 때 결정 기준**:

1. **Level 1 필수가 안 되면**: 다른 모든 것 멈추고 Level 1부터.
2. **Level 1은 되는데 Level 2가 막히면**: 위반 룰을 가장 단순한 1개만 (차선 위반).
3. **OCR이 안 되면**: 번호판 자리에 "AI 인식 중..." 또는 임시 텍스트 표시.
4. **배포가 안 되면**: 로컬 실행 화면 녹화 영상으로 대체.

**핵심**: **완벽함보다 작동.** 다 안 돼도 "AI가 영상에서 차량을 감지하는 화면" 1개만 있으면 OK.

---

## 📌 12. 빠른 시작 명령어

```bash
# 1. 프로젝트 생성
mkdir roadeye-mvp && cd roadeye-mvp
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install ultralytics opencv-python easyocr streamlit reportlab yt-dlp

# 3. requirements.txt 생성
pip freeze > requirements.txt

# 4. 샘플 영상 디렉토리
mkdir samples outputs

# 5. 첫 코드 작성
touch app.py
mkdir modules && touch modules/__init__.py modules/detector.py modules/ocr.py modules/report.py modules/utils.py

# 6. Streamlit 첫 실행 테스트
echo 'import streamlit as st; st.write("Hello RoadEye!")' > app.py
streamlit run app.py
```

---

## ✅ 13. 완료 체크리스트 (제출 전)

- [ ] Streamlit Cloud에 정상 배포되어 URL 접속 가능
- [ ] 샘플 영상 업로드 → 위반 감지 결과 정상 표시
- [ ] 시연 영상 1편 (mp4) 녹화 완료
- [ ] GitHub 레포 Public + README 작성
- [ ] QR 코드 PNG 생성 (PPT 삽입용)
- [ ] 100초 영상에 데모 클립 삽입 완료
- [ ] IR DECK에 배포 URL·QR 추가

---

**작성**: 2026년 5월 13일  
**프로젝트**: 로드아이 MVP (멋사 14기 아이디어톤)  
**문서 버전**: v1.0

이 문서를 읽었으면 **Day 1부터 순서대로** 진행해주세요. 시간이 부족하면 백업 플랜의 Level 1부터 우선 확보하세요.
