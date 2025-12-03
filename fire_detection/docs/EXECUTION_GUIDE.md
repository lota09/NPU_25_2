# 불꽃 감지 프로젝트 실행 가이드

## 📋 전체 워크플로우

### 준비 단계
```powershell
# 1. 필요한 패키지 설치
pip install ultralytics torch torchvision opencv-python numpy

# 2. GPU 확인 (선택사항, 하지만 권장)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 1단계: 모델 훈련 🎯

#### 방법 1: 배치 파일 사용 (가장 쉬움, Windows)
```powershell
# monoculus 폴더에서 실행
.\train_fire.bat
```

메뉴에서 선택:
- **옵션 1**: 빠른 훈련 (30분, 테스트용)
- **옵션 2**: 표준 훈련 (1시간, 권장) ⭐
- **옵션 3**: 고성능 훈련 (3시간, 최고 성능)
- **옵션 4**: 사용자 지정

#### 방법 2: Python 직접 실행
```powershell
# 기본 설정 (권장)
python train_fire_detection.py

# 빠른 테스트
python train_fire_detection.py --model yolov8n.pt --epochs 50 --name fire_quick

# 고성능 모델
python train_fire_detection.py --model yolov8s.pt --epochs 150 --name fire_advanced

# GPU 메모리 부족 시
python train_fire_detection.py --batch 8
```

**예상 소요 시간:**
- yolov8n, 50 epochs: ~30분 (GPU)
- yolov8n, 100 epochs: ~1시간 (GPU)
- yolov8s, 150 epochs: ~3시간 (GPU)

### 2단계: 동영상 처리 🎬

#### 방법 1: 배치 파일 사용 (가장 쉬움, Windows)
```powershell
# monoculus 폴더에서 실행
.\process_videos.bat
```

자동으로:
- 최신 훈련 모델 찾기
- assets 폴더의 모든 동영상 처리
- 신뢰도 임계값 선택

#### 방법 2: Python 직접 실행
```powershell
# assets 폴더의 모든 동영상 처리
python process_fire_videos.py `
    --model fire_detection_runs/fire_model/weights/best.pt `
    --video-dir assets

# 특정 동영상만 처리
python process_fire_videos.py `
    --model fire_detection_runs/fire_model/weights/best.pt `
    --videos assets/bucket11.mp4 assets/printer31.mp4 assets/roomfire41.mp4

# 신뢰도 임계값 조정
python process_fire_videos.py `
    --model fire_detection_runs/fire_model/weights/best.pt `
    --video-dir assets `
    --confidence 0.7
```

**처리 속도 (GPU 기준):**
- 1080p 동영상: ~10-15 FPS
- 720p 동영상: ~20-30 FPS
- 480p 동영상: ~40-60 FPS

### 3단계: 결과 확인 ✅

처리된 동영상 위치:
```
fire_detected_videos/
├── bucket11_fire_detected.mp4
├── printer31_fire_detected.mp4
└── roomfire41_fire_detected.mp4
```

훈련 결과 분석:
```
fire_detection_runs/fire_model/
├── weights/
│   ├── best.pt              # 최고 성능 모델 (이것을 사용하세요!)
│   └── last.pt              # 마지막 에포크 모델
├── results.png              # 훈련/검증 손실 그래프
├── confusion_matrix.png     # 혼동 행렬
├── F1_curve.png            # F1 스코어 곡선
└── PR_curve.png            # Precision-Recall 곡선
```

## 🚀 빠른 시작 (전체 과정)

```powershell
# monoculus 폴더로 이동
cd monoculus

# 1. 모델 훈련 (표준 설정)
python train_fire_detection.py

# 2. 3개 동영상 처리
python process_fire_videos.py `
    --model fire_detection_runs/fire_model/weights/best.pt `
    --videos assets/bucket11.mp4 assets/printer31.mp4 assets/roomfire41.mp4

# 3. 결과 확인
explorer fire_detected_videos
```

## 📊 성능 메트릭 해석

훈련 완료 후 출력되는 메트릭:

| 메트릭 | 설명 | 좋은 값 |
|--------|------|---------|
| **mAP50** | IoU 50%에서의 평균 정밀도 | >0.7 |
| **mAP50-95** | IoU 50-95%에서의 평균 정밀도 | >0.5 |
| **Precision** | 감지된 것 중 실제 불꽃의 비율 | >0.8 |
| **Recall** | 실제 불꽃 중 감지된 것의 비율 | >0.7 |

**예시:**
```
📈 성능 메트릭:
   - mAP50: 0.8234      ✅ 좋음!
   - mAP50-95: 0.5678   ✅ 양호
   - Precision: 0.8756  ✅ 매우 좋음!
   - Recall: 0.7890     ✅ 좋음!
```

## 🔧 문제 해결

### 문제 1: GPU 메모리 부족
```
RuntimeError: CUDA out of memory
```

**해결 방법:**
```powershell
# 배치 크기 줄이기
python train_fire_detection.py --batch 8

# 또는 이미지 크기 줄이기
python train_fire_detection.py --imgsz 416
```

### 문제 2: 훈련이 너무 느림 (CPU 사용 중)

**확인:**
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**해결 방법:**
- CUDA가 False면 GPU 드라이버 및 CUDA 설치 필요
- GPU 없이도 훈련 가능하지만 매우 느림 (10-20배)

### 문제 3: 감지 성능이 좋지 않음

**체크리스트:**
1. ✅ 훈련 데이터가 충분한가? (최소 수백 장)
2. ✅ 에포크를 충분히 했는가? (100+ 권장)
3. ✅ mAP50이 0.7 이상인가?

**개선 방법:**
```powershell
# 더 많은 에포크로 재훈련
python train_fire_detection.py --epochs 200

# 더 큰 모델 사용
python train_fire_detection.py --model yolov8s.pt --epochs 150

# 신뢰도 임계값 낮추기
python process_fire_videos.py --model <모델경로> --video-dir assets --confidence 0.4
```

### 문제 4: 동영상이 처리되지 않음

**확인:**
```powershell
# assets 폴더의 동영상 확인
dir assets\*.mp4
```

**해결 방법:**
- 동영상이 assets 폴더에 있는지 확인
- 파일 경로에 한글이나 특수문자가 없는지 확인
- 동영상 코덱이 지원되는지 확인 (MP4 권장)

## 💡 고급 팁

### 1. 훈련 중간에 진행 상황 확인

```powershell
# results.png 파일 열기 (실시간 업데이트됨)
start fire_detection_runs\fire_model\results.png
```

### 2. 웹캠으로 실시간 감지

```python
from ultralytics import YOLO

model = YOLO('fire_detection_runs/fire_model/weights/best.pt')
results = model.predict(source=0, show=True, conf=0.5)
```

### 3. 단일 이미지 테스트

```powershell
# 이미지 하나로 빠르게 테스트
python -c "from ultralytics import YOLO; YOLO('fire_detection_runs/fire_model/weights/best.pt').predict('test_image.jpg', show=True)"
```

### 4. 모델 내보내기 (배포용)

```powershell
# ONNX 형식 (다른 플랫폼에서 사용)
python -c "from ultralytics import YOLO; YOLO('fire_detection_runs/fire_model/weights/best.pt').export(format='onnx')"

# TensorRT 형식 (NVIDIA GPU 최적화)
python -c "from ultralytics import YOLO; YOLO('fire_detection_runs/fire_model/weights/best.pt').export(format='engine')"
```

## 📝 체크리스트

### 훈련 전
- [ ] 데이터셋이 올바른 위치에 있음 (`assets/home fire/train`, `val`, `test`)
- [ ] `fire_dataset.yaml` 파일이 올바르게 설정됨
- [ ] 필요한 패키지가 모두 설치됨
- [ ] GPU가 인식됨 (선택사항)

### 훈련 중
- [ ] 훈련이 정상적으로 시작됨
- [ ] Loss가 감소하는 추세
- [ ] mAP가 증가하는 추세
- [ ] GPU 메모리 사용률 정상

### 훈련 후
- [ ] `best.pt` 파일이 생성됨
- [ ] mAP50 > 0.7
- [ ] Precision, Recall > 0.7
- [ ] 혼동 행렬 확인

### 동영상 처리 전
- [ ] 훈련된 모델 경로 확인
- [ ] assets 폴더에 동영상 존재 확인
- [ ] 출력 폴더 경로 확인

### 동영상 처리 후
- [ ] 모든 동영상이 처리됨
- [ ] 불꽃이 올바르게 감지됨
- [ ] 결과 동영상 재생 가능
- [ ] 감지율이 합리적 (너무 많거나 적지 않음)

## 🎯 최종 목표

✅ **3개의 동영상 모두 처리 완료**
- `bucket11_fire_detected.mp4`
- `printer31_fire_detected.mp4`
- `roomfire41_fire_detected.mp4`

✅ **불꽃이 정확하게 감지됨**
- 빨간 박스로 표시
- "🔥 FIRE DETECTED!" 텍스트 표시
- 신뢰도 점수 표시

✅ **성능이 만족스러움**
- mAP50 > 0.7
- 실시간에 가까운 처리 속도 (GPU 사용 시)
- False Positive가 적음
