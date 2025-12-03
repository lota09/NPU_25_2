# 🔥 불꽃 감지 프로젝트 - 빠른 시작 가이드

## 📌 프로젝트 개요

YOLO(You Only Look Once)를 사용하여 불꽃을 감지하는 AI 모델을 훈련하고, 
assets 폴더의 3개 동영상(bucket11.mp4, printer31.mp4, roomfire41.mp4)에 대해 
객체인식 처리를 수행하는 프로젝트입니다.

## 🚀 가장 쉬운 실행 방법 (Windows)

### 1️⃣ 한 번에 실행하기

**PowerShell (권장 - 한글 깨짐 없음):**
```powershell
cd monoculus
.\run_fire_detection.ps1
```

**명령 프롬프트 (CMD):**
```cmd
cd monoculus
.\run_fire_detection.bat
```

메뉴에서 **"3. 전체 과정 실행"** 선택 → 자동으로 훈련 + 동영상 처리

### 2️⃣ 단계별 실행하기

**PowerShell (권장):**
```powershell
cd monoculus

# Step 1: 모델 훈련
.\train_fire.ps1

# Step 2: 동영상 처리
.\process_videos.ps1
```

**명령 프롬프트 (CMD):**
```cmd
cd monoculus

REM Step 1: 모델 훈련
.\train_fire.bat

REM Step 2: 동영상 처리
.\process_videos.bat
```

## 💻 Python으로 직접 실행하기

### 준비 (최초 1회만)

```powershell
# 필요한 패키지 설치
pip install ultralytics torch torchvision opencv-python numpy

# GPU 확인 (선택사항)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 1: 모델 훈련

```powershell
cd monoculus

# 기본 설정 (권장)
python train_fire_detection.py

# 빠른 테스트 (30분)
python train_fire_detection.py --epochs 50 --name fire_quick

# 고성능 모델 (3시간)
python train_fire_detection.py --model yolov8s.pt --epochs 150 --name fire_advanced
```

**예상 소요 시간:**
- GPU 있음: 1-3시간
- CPU만: 10-20시간

### Step 2: 동영상 처리

```powershell
# 자동으로 최신 모델과 모든 동영상 처리
python process_fire_videos.py `
    --model fire_detection_runs/fire_model/weights/best.pt `
    --video-dir assets

# 특정 동영상만 처리
python process_fire_videos.py `
    --model fire_detection_runs/fire_model/weights/best.pt `
    --videos assets/bucket11.mp4 assets/printer31.mp4 assets/roomfire41.mp4
```

## 📁 생성되는 파일들

```
monoculus/
├── fire_detection_runs/           # 훈련 결과 (자동 생성)
│   └── fire_model/
│       ├── weights/
│       │   ├── best.pt           # ⭐ 최고 성능 모델
│       │   └── last.pt           # 마지막 에포크 모델
│       ├── results.png           # 훈련 그래프
│       ├── confusion_matrix.png  # 혼동 행렬
│       └── ...
│
└── fire_detected_videos/          # 처리된 동영상 (자동 생성)
    ├── bucket11_fire_detected.mp4    # ⭐ 결과 동영상 1
    ├── printer31_fire_detected.mp4   # ⭐ 결과 동영상 2
    └── roomfire41_fire_detected.mp4  # ⭐ 결과 동영상 3
```

## ✅ 체크리스트

### 실행 전
- [ ] `monoculus` 폴더에 있음
- [ ] `assets/home fire/train` 폴더에 훈련 데이터 있음
- [ ] `assets` 폴더에 동영상 3개 있음
- [ ] Python과 필요한 패키지 설치됨

### 실행 후
- [ ] `fire_detection_runs/fire_model/weights/best.pt` 생성됨
- [ ] `fire_detected_videos/` 폴더에 3개 동영상 생성됨
- [ ] 동영상에서 불꽃이 빨간 박스로 표시됨

## 🎯 최종 결과물

**3개의 처리된 동영상:**
1. ✅ `bucket11_fire_detected.mp4` - 불꽃 감지 처리됨
2. ✅ `printer31_fire_detected.mp4` - 불꽃 감지 처리됨  
3. ✅ `roomfire41_fire_detected.mp4` - 불꽃 감지 처리됨

각 동영상에는:
- 🔥 불꽃 위치에 빨간 박스
- 🔥 "FIRE DETECTED!" 경고 텍스트
- 📊 신뢰도 점수 표시

## 🔧 문제 해결

### GPU 메모리 부족
```powershell
python train_fire_detection.py --batch 8
```

### 훈련이 너무 느림
```powershell
# 에포크 수 줄이기
python train_fire_detection.py --epochs 50
```

### 감지가 잘 안됨
```powershell
# 신뢰도 임계값 낮추기
python process_fire_videos.py --model <모델경로> --video-dir assets --confidence 0.4
```

## 📖 상세 가이드

더 자세한 내용은 다음 문서를 참고하세요:
- `README_FIRE_DETECTION.md` - 전체 프로젝트 문서
- `EXECUTION_GUIDE.md` - 상세 실행 가이드

## 💡 핵심 명령어 요약

**PowerShell (권장):**
```powershell
# 전체 프로세스
.\run_fire_detection.ps1

# 또는 개별 실행
.\train_fire.ps1              # 모델 훈련
.\process_videos.ps1          # 동영상 처리
```

**Python 직접 실행:**
```powershell
# 1. 모델 훈련
python train_fire_detection.py

# 2. 동영상 처리  
python process_fire_videos.py --model fire_detection_runs/fire_model/weights/best.pt --video-dir assets
```

---

**🎉 준비 완료! 이제 실행하세요!**
