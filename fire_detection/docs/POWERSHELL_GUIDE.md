# PowerShell 스크립트 실행 방법

## ⚠️ PowerShell 실행 정책 오류 해결

PowerShell 스크립트(.ps1) 실행 시 다음과 같은 오류가 발생할 수 있습니다:

```
이 시스템에서 스크립트를 실행할 수 없으므로...
```

### 해결 방법 1: 실행 정책 변경 (권장)

PowerShell을 **관리자 권한**으로 실행 후:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

확인 메시지에서 **Y** 입력

### 해결 방법 2: 일회성 실행

매번 실행 시:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_fire_detection.ps1
```

### 해결 방법 3: 배치 파일 사용

PowerShell 대신 .bat 파일 사용 (단, 한글이 깨질 수 있음):

```cmd
.\run_fire_detection.bat
```

---

## 🚀 실행 방법

### PowerShell 스크립트 (.ps1) - 권장 ✅

**장점:**
- ✅ 한글 인코딩 문제 없음
- ✅ 컬러 출력 지원
- ✅ 더 나은 에러 처리

**사용법:**
```powershell
# 전체 프로세스 실행
.\run_fire_detection.ps1

# 훈련만
.\train_fire.ps1

# 동영상 처리만
.\process_videos.ps1
```

### 배치 파일 (.bat) - CMD에서만

**장점:**
- ✅ 실행 정책 제약 없음
- ✅ CMD에서 바로 실행 가능

**단점:**
- ⚠️ PowerShell에서 한글 깨짐

**사용법 (CMD에서):**
```cmd
# 전체 프로세스 실행
.\run_fire_detection.bat

# 훈련만
.\train_fire.bat

# 동영상 처리만
.\process_videos.bat
```

---

## 📋 선택 가이드

| 상황 | 권장 방법 |
|------|----------|
| PowerShell 사용 중 | `.ps1` 스크립트 ✅ |
| CMD 사용 중 | `.bat` 파일 ✅ |
| 실행 정책 오류 | 해결 방법 1 또는 Python 직접 실행 |
| 한글 깨짐 | `.ps1` 스크립트 또는 Python 직접 실행 |

---

## 🐍 Python 직접 실행 (항상 작동)

실행 정책이나 인코딩 문제가 있다면 Python으로 직접 실행:

```powershell
# 모델 훈련
python train_fire_detection.py

# 동영상 처리
python process_fire_videos.py --model fire_detection_runs/fire_model/weights/best.pt --video-dir assets
```

이 방법은 스크립트 실행 정책과 무관하게 항상 작동합니다.
