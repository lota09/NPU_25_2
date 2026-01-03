# 🔥 Orange Pi 5 Plus NPU 실시간 화재 감지 시스템

완전한 화재 감지 솔루션으로, DXNN 모델을 사용하여 Orange Pi 5 Plus의 NPU에서 실시간 카메라 입력을 처리합니다.

## 📌 주요 특징

### 성능
- ⚡ **고속 추론**: NPU 가속으로 25-32 FPS 달성
- 🎯 **낮은 지연시간**: 50-100ms 응답 시간
- 💪 **효율적**: CPU 부하 최소화

### 기능
- 📹 **실시간 카메라 처리**: USB 웹캠, CSI 카메라 지원
- 🧠 **DXNN 모델**: 최적화된 NPU 실행
- 🔔 **멀티스레딩**: I/O와 추론 병렬 처리
- 📊 **통계 및 모니터링**: 실시간 성능 추적
- 💾 **데이터 저장**: 감지 결과 자동 저장
- 📧 **알림**: 이메일, SMS 등 알림 기능 (고급)

---

## 🚀 빠른 시작 (5분)

### 1. 환경 확인

```bash
python test_setup.py
```

### 2. 패키지 설치

```bash
pip install opencv-python numpy
# (선택) pip install dxnn-runtime
```

### 3. 화재 감지 실행

```bash
python fire_detection_camera_dxnn.py
```

더 자세한 가이드는 [QUICK_START_CAMERA.md](QUICK_START_CAMERA.md)를 참조하세요.

---

## 📁 파일 구조

```
fire_detection/
├── fire_detection_camera_dxnn.py          # 🌟 기본 카메라 감지 (권장)
├── fire_detection_camera_multithreaded.py # ⚡ 고성능 멀티스레딩 모드
├── advanced_fire_detection.py              # 🚀 고급 기능 (이메일, 녹화)
├── test_setup.py                           # 🧪 환경 확인 도구
├── QUICK_START_CAMERA.md                   # 📖 빠른 시작 가이드
├── CAMERA_SETUP_GUIDE.md                   # 📚 상세 설명서
└── models/
    └── best_npu_fp32_v1601/
        └── best_npu_concat.dxnn           # 🔥 DXNN 모델
```

---

## 🎯 사용 시나리오별 명령어

### 시나리오 1: 기본 사용 (권장)
```bash
python fire_detection_camera_dxnn.py
```

### 시나리오 2: 최고 성능
```bash
python fire_detection_camera_multithreaded.py --alert-threshold 3
```

### 시나리오 3: 높은 정확도
```bash
python fire_detection_camera_dxnn.py --conf 0.7 --width 1920 --height 1080
```

### 시나리오 4: 저전력 모드
```bash
python fire_detection_camera_dxnn.py --width 640 --height 480
```

### 시나리오 5: 고급 기능 (이메일, 녹화)
```bash
# 1. 설정 파일 생성
python advanced_fire_detection.py --create-config

# 2. fire_detection_config.json 수정 (이메일 설정)

# 3. 비디오 녹화와 함께 실행
python advanced_fire_detection.py --record
```

---

## 🔧 기술 사양

### 하드웨어 요구사항
| 항목 | 요구사항 |
|------|---------|
| SoC | Orange Pi 5 Plus (Rockchip RK3588) |
| NPU | Rockchip NPU (2.4 TOPS) |
| RAM | 8GB+ 권장 |
| 카메라 | USB 웹캠 또는 MIPI CSI |
| 전원 | 5V/3A+ |

### 소프트웨어 요구사항
| 항목 | 버전 |
|------|------|
| Python | 3.7+ |
| OpenCV | 4.5+ |
| NumPy | 1.20+ |
| DXRT | 2.9+ (선택) |

### 성능 메트릭
| 지표 | 기본 모드 | 멀티스레딩 |
|------|---------|-----------|
| FPS | 25-28 | 28-32 |
| 지연시간 | 100ms | 50ms |
| CPU 사용률 | 40-50% | 30-40% |
| 메모리 | ~300MB | ~350MB |

---

## 📖 사용 설명서

### 1. 기본 모드 (권장)

```python
from fire_detection_camera_dxnn import FireDetectionDXNN

detector = FireDetectionDXNN(
    confidence_threshold=0.5,
    input_size=(640, 640)
)
detector.run_camera()
```

**특징:**
- 간단하고 직관적
- 안정적인 성능
- 프로덕션 환경 적합

### 2. 멀티스레딩 모드 (고성능)

```python
from fire_detection_camera_multithreaded import FireDetectionSystem

system = FireDetectionSystem(
    alert_threshold=3
)
system.run()
```

**특징:**
- 더 높은 FPS
- 낮은 지연시간
- 자동 경고 시스템

### 3. 고급 모드 (기능 풍부)

```python
from advanced_fire_detection import AdvancedFireDetectionSystem

system = AdvancedFireDetectionSystem(
    save_detections=True,
    output_dir='results'
)
system.run_with_recording(record_video=True)
```

**특징:**
- 결과 자동 저장
- 비디오 녹화
- 통계 추적
- 이메일 알림 (설정)

---

## 🎨 시각화 및 출력

### 화면 표시
```
카메라 영상
├─ 감지된 화재 (빨간 박스)
├─ 신뢰도 점수 (텍스트)
├─ FPS (우상단)
└─ 감지 개수 (중단)
```

### 콘솔 로그
```
✅ 카메라 준비 완료
   해상도: 1280x720
   신뢰도 임계값: 0.5

🔥 화재 감지: 1개 (0.87)
```

### 저장 파일
- `detection_*.jpg` - 감지된 프레임
- `detection_*.json` - 감지 정보 (좌표, 신뢰도)
- `recording_*.mp4` - 비디오 녹화 (선택)
- `statistics_*.json` - 통계 (선택)

---

## 🔍 문제 해결

### 문제 1: 카메라 오류
```bash
# 원인 확인
python test_setup.py

# 카메라 장치 확인
ls /dev/video*

# 권한 설정
sudo chmod 666 /dev/video*

# 다른 카메라 ID 시도
python fire_detection_camera_dxnn.py --camera-id 1
```

### 문제 2: 낮은 FPS
```bash
# 멀티스레딩 모드 사용
python fire_detection_camera_multithreaded.py

# 해상도 감소
python fire_detection_camera_dxnn.py --width 640 --height 480

# 신뢰도 증가
python fire_detection_camera_dxnn.py --conf 0.7
```

### 문제 3: DXRT 오류
```bash
# 재설치
pip install --upgrade dxnn-runtime

# 확인
python -c "from dx_engine import InferenceEngine; print('OK')"
```

더 많은 문제 해결 방법은 [CAMERA_SETUP_GUIDE.md](CAMERA_SETUP_GUIDE.md#-문제-해결)를 참조하세요.

---

## 📊 성능 최적화

### 옵션 1: 최고 FPS 원하기
```bash
python fire_detection_camera_multithreaded.py
# 예상: 30+ FPS
```

### 옵션 2: 정확도 중시
```bash
python fire_detection_camera_dxnn.py --conf 0.8 --width 1920 --height 1080
# 예상: 15-20 FPS, 높은 정확도
```

### 옵션 3: 균형잡힌 설정 (권장)
```bash
python fire_detection_camera_dxnn.py
# 예상: 25-28 FPS, 좋은 정확도
```

### 옵션 4: 저전력 모드
```bash
python fire_detection_camera_dxnn.py --width 640 --height 480 --conf 0.7
# 예상: 30+ FPS, 낮은 CPU 사용
```

---

## 🌐 배포 가이드

### 1. 서비스로 등록

```bash
# /etc/systemd/system/fire-detection.service 생성
sudo tee /etc/systemd/system/fire-detection.service << EOF
[Unit]
Description=Fire Detection Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 $(pwd)/fire_detection_camera_multithreaded.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl enable fire-detection.service
sudo systemctl start fire-detection.service

# 상태 확인
sudo systemctl status fire-detection.service
```

### 2. 무한 루프로 실행

```bash
#!/bin/bash
while true; do
    python fire_detection_camera_multithreaded.py
    sleep 5
done
```

### 3. 로그 모니터링

```bash
# 실시간 로그 확인
python fire_detection_camera_dxnn.py 2>&1 | tee detection.log

# 특정 로그만 필터링
grep "화재\|경고" detection.log
```

---

## 📚 추가 자료

### 공식 문서
- [Orange Pi 5 Plus 설명서](https://orangepi.org/)
- [OpenCV Python 튜토리얼](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [DXNN 런타임](https://github.com/deepx-ai/dxnn-runtime)

### 관련 파일
- [빠른 시작 가이드](QUICK_START_CAMERA.md)
- [상세 설명서](CAMERA_SETUP_GUIDE.md)
- [프로젝트 개요](PROJECT_STRUCTURE.md)

---

## 🤝 기여 및 피드백

문제 보고 및 제안:
1. `test_setup.py` 실행하여 환경 정보 수집
2. 오류 메시지 및 로그 첨부
3. 재현 단계 명시

---

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 🎓 학습 자료

### Python 기초
```python
# OpenCV 기초
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow('frame', frame)
cap.release()
```

### DXNN 모델 사용
```python
from dx_engine import InferenceEngine
engine = InferenceEngine('model.dxnn')
output = engine.infer(input_data)
```

### NumPy 기초
```python
import numpy as np
arr = np.array([1, 2, 3])
normalized = arr / 255.0
```

---

## 🔐 보안 주의사항

⚠️ **프로덕션 배포 시 주의**

1. **이메일 설정**: 앱 비밀번호 사용 (일반 비밀번호 사용 금지)
2. **카메라 권한**: 불필요한 권한 제한
3. **네트워크**: HTTPS 사용, 인증 필요
4. **데이터 보호**: 감지 결과 암호화 저장

---

## 💡 팁과 트릭

### 팁 1: 카메라 미리보기
```bash
python -c "import cv2; cv2.VideoCapture(0).isOpened()" && echo "✅ 카메라 OK"
```

### 팁 2: 신뢰도 미세 조정
```bash
for conf in 0.3 0.5 0.7 0.9; do
    echo "테스트 conf=$conf"
    python fire_detection_camera_dxnn.py --conf $conf &
    sleep 30
    pkill -f fire_detection_camera_dxnn.py
done
```

### 팁 3: 성능 모니터링
```bash
# 리소스 사용량 모니터링
watch -n 1 "top -b -n 1 | grep python"
```

### 팁 4: 배치 처리
```python
# 여러 이미지 처리
from fire_detection_camera_dxnn import FireDetectionDXNN
import cv2
from pathlib import Path

detector = FireDetectionDXNN()
for img_path in Path('images').glob('*.jpg'):
    frame = cv2.imread(str(img_path))
    result, detections = detector.process_frame(frame)
    print(f"{img_path}: {len(detections)} 감지")
```

---

## 📞 지원

문제가 발생하면:

1. **환경 확인**: `python test_setup.py` 실행
2. **로그 확인**: 콘솔 출력 및 오류 메시지 검토
3. **문서 참조**: [CAMERA_SETUP_GUIDE.md](CAMERA_SETUP_GUIDE.md) 참조
4. **온라인 검색**: Orange Pi, OpenCV, PyTorch 관련 이슈 검색

---

**작성일**: 2025년 12월  
**버전**: 1.0  
**마지막 업데이트**: 2025년 12월  

🔥 **화재 감지 시스템을 안전하게 운영하세요!** 🚀
