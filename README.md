# 🚨 로드아이 (RoadEye) MVP

> 블랙박스 영상 → AI가 위반 차량 자동 감지 → 위반 유형 표시

**2026 멋쟁이사자처럼 대학 14기 아이디어톤** 출품작.

---

## 빠른 시작 (로컬)

```powershell
# Windows PowerShell
cd roadeye-mvp
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

> Windows에서 `python` 명령이 Microsoft Store 스텁으로 잡혀 있으면 `py -m venv` 를 쓰세요. 활성화 후엔 `python` / `pip` 가 venv 의 것을 가리킵니다.
> PowerShell 실행 정책 오류가 나면: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

브라우저가 자동으로 열리며, `http://localhost:8501` 로 접근할 수 있습니다.

> 첫 실행 시 YOLO(`yolov8n.pt`)와 EasyOCR 한국어 모델이 자동 다운로드됩니다 (~250MB).

---

## 사용법

1. 사이드바에서 분석 길이 / 프레임 스킵 / OCR·블러 옵션 조정
2. `블랙박스 영상을 업로드하세요` 에 mp4 파일 드롭
3. `🔍 분석 시작` 클릭
4. 진행바 완료 후 카드별로 표시:
   - 위반 유형 배지 (차선 위반 / 급차선 변경 / 불법유턴 / 신호 위반)
   - 감지 시각, 차량 ID, AI 신뢰도
   - 감지 시점 스냅샷 (위반 차량 빨간 박스 + 주변 차량 블러)
   - 추출된 번호판

---
## JPG 이미지 시퀀스를 MP4로 변환하기

`01.데이터` 안에 JPG 이미지가 있을 때, 5 FPS로 묶어 약 5개의 MP4 파일을 만들려면 다음 스크립트를 사용하세요.

```powershell
cd roadeye-mvp
python convert_jpgs_to_videos.py --source "01.데이터\your_image_folder" --output outputs/jpg_videos --num-videos 5 --fps 5
```

- 출력 비디오 파일은 `outputs/jpg_videos/video_01.mp4` 등으로 생성됩니다.
- 이미지 파일 이름 순서대로 묶습니다.
- 프레임 크기가 서로 다르면 첫 번째 이미지 크기에 맞춰 리사이즈합니다.

---
## 감지 위반 유형

| 키 | 라벨 | 룰 (간략) | 정확도 |
|---|---|---|---|
| `lane` | 차선 위반 | 차량 중심이 에고 차선 경계를 가로지름 | 데모 수준 |
| `rapid_lane_change` | 급차선 변경 | 2초 이내 차선 2회 이상 가로지름 | 데모 수준 |
| `illegal_uturn` | 불법 유턴 | 수직 진행 방향 반전 + 큰 횡방향 이동 | 휴리스틱 (낮음) |
| `traffic_light` | 신호 위반 | 신호등 ROI HSV 분석 = red + 차량 전진 | 휴리스틱 (낮음) |

> 룰베이스 휴리스틱이며 한국 도로 환경 학습 데이터 없이 동작합니다. 발표 시연용으로 잘 작동하는 영상을 골라 사용하세요.

---

## 프로젝트 구조

```
roadeye-mvp/
├── app.py                  # Streamlit 진입점
├── modules/
│   ├── detector.py         # YOLO + 위반 룰 + 차선 감지
│   ├── tracker.py          # 차량 ID별 궤적 히스토리
│   ├── ocr.py              # 번호판 OCR + 블러
│   └── utils.py            # 영상 입출력
├── requirements.txt
├── packages.txt            # Streamlit Cloud용 시스템 패키지
└── .streamlit/config.toml
```

---

## Streamlit Cloud 배포 (선택)

1. 이 폴더를 GitHub Public 레포로 푸시
2. https://share.streamlit.io 접속 → `New app`
3. 레포 / 브랜치 / 메인 파일 (`app.py`) 선택
4. 빌드 완료 후 배포 URL 확보

> 메모리 한계로 인해 YOLOv8n(nano)와 영상 길이 30초 이하를 권장합니다.

---

## 알려진 제약

- 차선 감지는 Hough 기반이라 야간·우천 영상에서 정확도 저하
- 신호 위반과 불법유턴은 데모 수준의 룰. 실서비스에는 별도 모델 필요
- EasyOCR 첫 실행 시 모델 다운로드로 1~2분 소요

---

## 라이선스

데모 / 교육 용도. 외부 영상 사용 시 출처를 표기하세요.
