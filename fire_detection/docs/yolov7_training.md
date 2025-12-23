# YOLOv7 화염/연기 감지 모델 학습 가이드 (Linux x86 + GPU)

## 1. 시스템 요구사항
- Linux x86_64 (Ubuntu 20.04 이상 권장)
- NVIDIA GPU (CUDA 지원)
- CUDA 12.1 이상
- Python 3.11.x
- 최소 16GB RAM 권장
- 최소 10GB GPU VRAM 권장

## 2. 프로젝트 파일 전송
```bash
# Windows에서 Linux 서버로 fire_detection 폴더 전송 (SCP 사용)
# Windows PowerShell에서 실행:
scp -r C:\Users\lota\Documents\SourceCodes\vscode\NPU_25_2\fire_detection user@server_ip:/home/user/

# 또는 Git을 통해 전송
# Windows에서:
cd C:\Users\lota\Documents\SourceCodes\vscode\NPU_25_2
git add fire_detection/
git commit -m "Add YOLOv7 fire detection training files"
git push

# Linux에서:
git clone https://github.com/lota09/25_1_NPU.git
cd 25_1_NPU/fire_detection
```

## 3. Python 버전 확인 및 설치

### 설치된 Python 버전 확인
```bash
# 방법 1: 시스템에 설치된 모든 Python 버전 확인
ls /usr/bin/python* 

# 방법 2: 사용 가능한 Python 버전 확인
which -a python python3 python3.8 python3.9 python3.10 python3.11 python3.12

# 방법 3: 각 버전 확인
python --version
python3 --version
python3.8 --version 2>/dev/null || echo "Python 3.8 not installed"
python3.9 --version 2>/dev/null || echo "Python 3.9 not installed"
python3.10 --version 2>/dev/null || echo "Python 3.10 not installed"
python3.11 --version 2>/dev/null || echo "Python 3.11 not installed"
python3.12 --version 2>/dev/null || echo "Python 3.12 not installed"
```

### 대체 가능한 Python 버전
**권장 순서: Python 3.11 > 3.10 > 3.9**

- ✅ **Python 3.11.x**: 최적 (테스트 완료)
- ✅ **Python 3.10.x**: 호환 가능 (PyTorch 2.5.1 지원)
- ✅ **Python 3.9.x**: 호환 가능 (PyTorch 2.5.1 지원)
- ⚠️ **Python 3.8.x**: 사용 가능하나 일부 라이브러리 구버전 필요
- ❌ **Python 3.12+**: numpy 호환성 문제 가능
- ❌ **Python 3.7 이하**: PyTorch 2.5.1 미지원

### 옵션 A: Python 3.11 설치 (관리자 권한 필요)
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Python 버전 확인
python3.11 --version
```

### 옵션 B: 기존 Python 버전 사용 (관리자 권한 불필요)
```bash
# Python 3.10 사용 예시
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 옵션 C: Windows .venv 전체 복사 (권장하지 않음)
```bash
# ⚠️ 경고: Windows와 Linux는 바이너리가 다르므로 작동 안 함
# Windows .venv는 .exe, .dll 파일 포함
# Linux .venv는 .so, ELF 바이너리 필요
# 따라서 .venv 복사는 불가능!

# 대신 requirements.txt 사용:
# Windows에서 생성한 requirements.txt를 Linux에서 사용
pip install -r requirements.txt
```

### 옵션 D: Miniconda 사용 (관리자 권한 불필요, **Python 3.12만 있을 때 권장**)
```bash
# 1. Miniconda 다운로드 및 설치 (사용자 홈 디렉토리에 설치)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 2. conda 초기화
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# 3. Anaconda 약관 동의 (필수)
conda config --set channel_priority flexible
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. Python 3.11 환경 생성 (yolov7이라는 이름으로)
conda create -n yolov7 python=3.11 -y

# 5. 환경 활성화
conda activate yolov7

# 6. Python 버전 확인
python --version  # Python 3.11.x 출력

# 7. fire_detection 디렉토리로 이동
cd fire_detection

# 8. PyTorch 및 CUDA 설치
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 9. CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 10. 나머지 의존성 설치
pip install -r requirements.txt

# 11. YOLOv7 저장소 클론
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
cd ..

# 12. 학습 시작
python train_fire_detection.py --epochs 100 --batch 8
```

### Miniconda 환경 관리 명령어
```bash
# 환경 목록 확인
conda env list

# yolov7 환경 활성화
conda activate yolov7

# 환경 비활성화
conda deactivate

# 환경 삭제 (필요시)
conda env remove -n yolov7

# 설치된 패키지 확인
conda list
pip list
```

## 4. CUDA 설치 확인
```bash
# CUDA 버전 확인
nvidia-smi

# CUDA 12.1 이상이 아니면 설치 필요
# CUDA Toolkit 설치 (Ubuntu 예시)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 5. 가상환경 생성 및 활성화

### Python 3.11 사용 (설치되어 있는 경우)
```bash
cd fire_detection

# 가상환경 생성
python3.11 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 활성화 확인
which python
python --version  # Python 3.11.x 출력
```

### Python 3.10 사용 (3.11이 없는 경우)
```bash
cd fire_detection

# 가상환경 생성
python3.10 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 활성화 확인
which python
python --version  # Python 3.10.x 출력
```

### Conda 환경 사용 (관리자 권한 없는 경우 권장)
```bash
cd fire_detection

# Conda 환경 활성화
conda activate yolov7

# 활성화 확인
which python
python --version
```

## 6. PyTorch 및 의존성 설치
```bash
# pip 업그레이드
pip install --upgrade pip

# PyTorch 2.5.1 + CUDA 12.1 설치
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 사용 가능 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 나머지 의존성 설치
pip install -r requirements.txt
```

## 7. YOLOv7 저장소 클론 및 가중치 다운로드
```bash
# YOLOv7 공식 저장소 클론
git clone https://github.com/WongKinYiu/yolov7.git

# 사전학습 가중치 다운로드
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
cd ..
```

## 8. 데이터셋 구조 확인
```bash
# 데이터셋 구조 확인
tree -L 3 assets/homefire/

# 출력 예시:
# assets/homefire/
# ├── train/
# │   ├── images/
# │   └── labels/
# ├── val/
# │   ├── images/
# │   └── labels/
# └── test/
#     ├── images/
#     └── labels/

# 이미지 및 레이블 수 확인
echo "Train images: $(ls assets/homefire/train/images/*.jpg 2>/dev/null | wc -l)"
echo "Train labels: $(ls assets/homefire/train/labels/*.txt 2>/dev/null | wc -l)"
echo "Val images: $(ls assets/homefire/val/images/*.jpg 2>/dev/null | wc -l)"
echo "Val labels: $(ls assets/homefire/val/labels/*.txt 2>/dev/null | wc -l)"

# 레이블 클래스 확인 (0: 불꽃, 1: 연기)
head -5 assets/homefire/train/labels/*.txt
```

## 9. 학습 실행

### 리다이렉션(Redirection) 문법 설명
Linux/Unix에서 출력을 파일로 저장하는 방법:

```bash
# 기본 리다이렉션 기호
>   # 표준 출력(stdout)을 파일로 저장 (덮어쓰기)
>>  # 표준 출력(stdout)을 파일에 추가 (append)
2>  # 표준 에러(stderr)를 파일로 저장
2>> # 표준 에러(stderr)를 파일에 추가
&>  # stdout과 stderr 모두 파일로 저장 (덮어쓰기)
&>> # stdout과 stderr 모두 파일에 추가

# 파일 디스크립터 번호
0 = stdin  (표준 입력)
1 = stdout (표준 출력, 일반 print 문)
2 = stderr (표준 에러, 오류 메시지)
```

**`2>&1` 의미:**
- `2>` : stderr(파일 디스크립터 2)를 리다이렉션
- `&1` : stdout(파일 디스크립터 1)과 같은 곳으로
- **결과**: 에러 메시지도 일반 출력과 함께 같은 파일에 저장

**예시:**
```bash
# 1. stdout만 저장 (에러는 화면에 표시)
python train.py > output.log

# 2. stderr만 저장 (일반 출력은 화면에 표시)
python train.py 2> error.log

# 3. stdout과 stderr를 각각 다른 파일에 저장
python train.py > output.log 2> error.log

# 4. stdout과 stderr를 같은 파일에 저장 (권장)
python train.py > training.log 2>&1

# 5. 간단한 방법 (4번과 동일)
python train.py &> training.log

# 6. 파일에 추가 (이어쓰기)
python train.py >> training.log 2>&1
```

**순서 중요!**
```bash
# ✅ 올바름: stdout을 파일로, stderr도 같은 곳으로
python train.py > training.log 2>&1

# ❌ 틀림: stderr를 stdout으로, 그 다음 stdout을 파일로 (stderr는 여전히 화면 출력)
python train.py 2>&1 > training.log
```

---

### 기본 학습 (터미널 연결 필요)
```bash
# 가상환경 활성화
conda activate yolov7

# 학습 시작 (화면에 출력)
python train_fire_detection.py --epochs 200 --batch 16 --patience 50
```

---

### 백그라운드 학습 (nohup 사용, 권장!)

**nohup이란?**
- **no hangup**: 터미널 종료해도 프로세스 계속 실행
- SSH 연결 끊어져도 학습 계속됨
- 출력을 자동으로 `nohup.out` 파일에 저장

```bash
# 방법 1: 기본 nohup (nohup.out 파일에 저장)
nohup python train_fire_detection.py --epochs 200 --batch 16 &

# 방법 2: 커스텀 로그 파일 지정 (권장!)
nohup python train_fire_detection.py --epochs 200 --batch 16 > training.log 2>&1 &

# 방법 3: 날짜가 포함된 로그 파일명
nohup python train_fire_detection.py --epochs 200 --batch 16 > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 방법 4: conda 환경까지 포함한 완전한 명령어
nohup bash -c "source ~/miniconda3/bin/activate yolov7 && python train_fire_detection.py --epochs 200 --batch 16 --patience 50" > training.log 2>&1 &
```

**명령어 구성 요소 설명:**
```bash
nohup                          # 터미널 종료해도 계속 실행
python train_fire_detection.py # 실행할 프로그램
--epochs 200 --batch 16        # 프로그램 옵션
> training.log                 # stdout을 training.log에 저장 (덮어쓰기)
2>&1                           # stderr도 stdout과 같은 파일(training.log)로
&                              # 백그라운드 실행 (명령 프롬프트 즉시 반환)
```

---

### 학습 모니터링 명령어

```bash
# 실행 중인 프로세스 확인
ps aux | grep train_fire_detection

# 프로세스 ID(PID) 확인
pgrep -f train_fire_detection

# 로그 실시간 확인 (Ctrl+C로 종료)
tail -f training.log

# 로그 마지막 50줄 확인
tail -50 training.log

# 로그에서 epoch 진행상황 검색
grep "Epoch" training.log

# GPU 사용률 실시간 모니터링
watch -n 1 nvidia-smi

# 디스크 용량 확인
df -h
```

---

### 학습 중단 및 재개

```bash
# 프로세스 찾기
ps aux | grep train_fire_detection

# 정상 종료 (SIGTERM)
kill $(pgrep -f train_fire_detection)

# 강제 종료 (SIGKILL, 비권장)
kill -9 $(pgrep -f train_fire_detection)

# 또는 PID 직접 지정
kill 12345  # 12345는 ps aux에서 확인한 PID
```

---

### 완전한 학습 실행 예제

```bash
# 1. 작업 디렉토리로 이동
cd ~/fire_detection

# 2. 캐시 삭제 (처음 실행 시)
rm -f assets/homefire/train/labels.cache
rm -f assets/homefire/val/labels.cache
rm -f assets/homefire/test/labels.cache

# 3. conda 환경 활성화
conda activate yolov7

# 4. 백그라운드 학습 시작
nohup python train_fire_detection.py --epochs 200 --batch 16 > training.log 2>&1 &

# 5. 백그라운드 작업 번호 확인 (예: [1] 12345)
# [1]은 작업 번호, 12345는 PID

# 6. 로그 확인
tail -f training.log

# 7. 터미널 종료해도 계속 실행됨 (exit 또는 Ctrl+D)
exit
```

---

### 재접속 후 확인

```bash
# SSH 재접속 후

# 1. 프로세스 실행 중인지 확인
ps aux | grep train_fire_detection

# 2. 로그 확인
cd ~/fire_detection
tail -f training.log

# 3. 학습 진행 상황 요약
grep "Epoch" training.log | tail -20

# 4. 최신 mAP 확인
grep "mAP" training.log | tail -5
```

### GPU 메모리가 충분한 경우 배치 사이즈 증가
```bash
# 배치 사이즈 16 (더 빠른 학습)
python train_fire_detection.py --epochs 100 --batch 16

# 배치 사이즈 32 (가장 빠른 학습, 20GB+ VRAM 필요)
python train_fire_detection.py --epochs 100 --batch 32
```

### 다중 GPU 사용
```bash
# GPU 2개 사용 (device 0,1)
python yolov7/train.py \
  --workers 8 \
  --device 0,1 \
  --batch-size 16 \
  --epochs 100 \
  --data $(pwd)/fire_data.yaml \
  --img 640 \
  --cfg yolov7/cfg/training/yolov7.yaml \
  --weights yolov7/yolov7.pt \
  --name fire_model \
  --hyp yolov7/data/hyp.scratch.p5.yaml \
  --project $(pwd)/runs/train
```

## 10. 학습 모니터링
```bash
# TensorBoard 실행
tensorboard --logdir runs/train --host 0.0.0.0 --port 6006

# 브라우저에서 접속:
# http://server_ip:6006

# 학습 진행 상황 확인
watch -n 10 "tail -20 training.log"

# GPU 사용률 모니터링
watch -n 1 nvidia-smi
```

## 11. 학습 결과 확인
```bash
# 학습 완료 후 결과 확인
ls -lh runs/train/fire_model*/weights/

# 최적 모델: runs/train/fire_model/weights/best.pt
# 마지막 모델: runs/train/fire_model/weights/last.pt

# 학습 결과 그래프
ls runs/train/fire_model*/*.png

# mAP, precision, recall 등 확인
cat runs/train/fire_model*/results.txt
```

## 12. 학습된 모델 Windows로 전송
```bash
# Linux에서 Windows로 전송 (SCP)
scp -r runs/train/fire_model* user@windows_ip:/path/to/destination/

# 또는 Git을 통해 전송 (모델 파일이 크므로 Git LFS 사용 권장)
git lfs install
git lfs track "*.pt"
git add runs/train/fire_model*/weights/best.pt
git commit -m "Add trained YOLOv7 fire detection model"
git push
```

## 13. 모델 ONNX 변환 (선택사항)
```bash
# Linux에서 ONNX 변환
python convert_yolov7.py runs/train/fire_model/weights/best.pt

# 변환된 ONNX 파일 확인
ls -lh runs/train/fire_model/weights/best.onnx
```

## 14. 학습 중단 및 재개
```bash
# 학습 중단
# Ctrl+C 또는
kill -9 $(ps aux | grep train_fire_detection | grep -v grep | awk '{print $2}')

# 학습 재개 (마지막 체크포인트부터)
python yolov7/train.py \
  --workers 8 \
  --device 0 \
  --batch-size 8 \
  --epochs 100 \
  --data $(pwd)/fire_data.yaml \
  --img 640 \
  --cfg yolov7/cfg/training/yolov7.yaml \
  --weights runs/train/fire_model/weights/last.pt \
  --name fire_model_resume \
  --hyp yolov7/data/hyp.scratch.p5.yaml \
  --project $(pwd)/runs/train \
  --resume
```

## 15. 문제 해결

### CUDA Out of Memory 오류
```bash
# 배치 사이즈 감소
python train_fire_detection.py --epochs 100 --batch 4

# 또는 이미지 크기 감소
# train_fire_detection.py 수정: '--img', '640' -> '--img', '512'
```

### GPU 인식 안됨
```bash
# CUDA 설치 확인
nvcc --version

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# NVIDIA 드라이버 재설치
sudo apt install --reinstall nvidia-driver-535
sudo reboot
```

### 의존성 오류
```bash
# 가상환경 재생성
deactivate
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 16. 예상 학습 시간
- RTX 3080 (10GB): 약 3-4시간 (batch=8, 100 epochs)
- RTX 3090 (24GB): 약 2-3시간 (batch=16, 100 epochs)
- A100 (40GB): 약 1-2시간 (batch=32, 100 epochs)

## 17. 클래스 정보
- **클래스 0**: 불꽃 (flame)
- **클래스 1**: 연기 (smoke)
- 총 클래스 수: 2개 (nc=2)
- 두 클래스 모두 감지 시 사용자에게 알림 전송

## 18. 다음 단계
학습 완료 후:
1. `best.pt` 모델을 Windows로 전송
2. `detect_fire_yolov7.py` 실시간 감지 스크립트 실행
3. 카메라로 불꽃/연기 감지 테스트
4. NPU 배포를 위한 ONNX 변환 및 최적화
