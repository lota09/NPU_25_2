## 🚨 현황 정리: Orange Pi NPU 배포 막힌 상태

### 📌 핵심 문제
로컬 **dx_com v2.0.0**으로 컴파일한 DXNN 모델이 **Orange Pi DXRT v2.9.5**에서 로드되지 않음

### ✅ 완료된 작업
1. **YOLOv7 모델 학습 및 검증**
   - 2 클래스 화재 감지 모델
   - PyTorch 추론 정상 작동 ✅
   - 출력: [1, 25200, 7]

2. **ONNX 포맷 변환**
   - NCHW [1,3,640,640] FLOAT32 ✅
   - NHWC [1,640,640,3] FLOAT32 ✅
   - NHWC [1,640,640,3] UINT8 ✅
   - 모두 ONNX 체커 통과

3. **로컬 dx_com 컴파일**
   - INT8 Quantized ✅
   - FP32 (Non-quantized) ✅
   - NHWC UINT8 ✅
   - 모두 컴파일 성공

4. **Orange Pi 연결**
   - SSH 키 기반 연결 ✅
   - 모델 파일 전송 ✅

### ❌ 막힌 문제들

| 모델 타입 | 상태 | 에러 |
|----------|------|------|
| INT8 (로컬 컴파일) | ❌ | "Unwanted Data Type is inserted in GetDataSize.NONE_TYPE" |
| FP32 (로컬 컴파일) | ❌ | 타임아웃 (10초+) |
| NHWC UINT8 (로컬 컴파일) | ❌ | 타임아웃 (15초+) |
| 공식 YOLOv7 ONNX (로컬 컴파일) | ❌ | 타임아웃 (20초+) |

### ✅ 성공 사례
Orange Pi 기본 제공 YOLOv7 DXNN 모델들
- `/home/orangepi/Downloads/yolov7.dxnn` → ✅ 로드 성공 (1.40초)
- `/home/orangepi/dx-all-suite/.../YoloV7.dxnn` → ✅ 로드 성공 (1.38초)

**차이점:**
- 예제 모델은 UINT8 NHWC 입력으로 그래프 구성
- 우리 모델은 NCHW 입력으로 구성
- 컴파일러 버전 차이 가능성

### 🔍 근본 원인 추정

**버전 미스매치:**
- 로컬 dx_com: v2.0.0 (최신)
- Orange Pi DXRT: v2.9.5 (상대적으로 구버전)
- 생성되는 DXNN 포맷이 호환되지 않을 가능성

**그래프 컴파일 이슈:**
- "No graphinfo (1)" 경고
- 텐서 타입 메타데이터 손실 가능성
- DXRT이 필요한 정보 누락

### 🎯 다음 시도 방안 (순서대로)

#### 1. **급소: DeepX 버전 호환성 확인** (가장 가능성 높음)
- dx_com v2.0.0이 생성하는 DXNN이 DXRT v2.9.5와 호환되는지 확인
- 호환되는 다른 버전 확인
- DeepX 지원팀 연락

#### 2. **로컬 dx_com 버전 다운그레이드**
- DXRT v2.9.5에 맞는 구 버전 dx_com 다운로드 시도
- 호환성 테스트

#### 3. **Orange Pi DXRT 업그레이드**
- DXRT를 최신 버전으로 업그레이드
- 로컬 dx_com과 동일 버전 계열 맞추기

#### 4. **임시 해결책: CPU 추론**
```python
# Orange Pi에서 PyTorch로 직접 실행
import torch
model = torch.load('best.pt')
output = model(input_tensor)  # CPU 추론 (느리지만 작동)
```

### 📊 현황 요약

```
┌─────────────────────────────────────┐
│ 로컬 환경                           │
│ - YOLOv7 모델: ✅ 작동             │
│ - ONNX 변환: ✅ 성공               │
│ - dx_com v2.0.0: ✅ 컴파일 성공    │
└─────────────────────────────────────┘
                ↓ (DXNN 파일 전송)
┌─────────────────────────────────────┐
│ Orange Pi DXRT v2.9.5               │
│ - 로컬 DXNN: ❌ 로드 불가          │
│ - 예제 DXNN: ✅ 정상 작동          │
│ → 호환성 문제 (버전? 포맷?)        │
└─────────────────────────────────────┘
```

### 📋 즉시 필요한 정보

1. **DeepX 지원 연락**
   - 문제 상세: `support_inquiry.md` 참조
   - 버전 호환성 확인 요청
   - 다운로드 가능한 호환 버전 목록

2. **로컬 환경에서 할 수 있는 것**
   - 다른 버전의 dx_com 다운로드 및 테스트
   - 예제 ONNX로 재컴파일
   - 컴파일 로그 상세 분석

3. **Orange Pi에서 할 수 있는 것**
   - PyTorch 모델 직접 로드 (CPU 추론)
   - DXRT 업그레이드 (시간 소모)
   - 예제 모델로 NPU 작동 확인

### 🎯 권장 다음 액션

**우선순위:**
1. ⚡ **긴급**: DeepX 지원팀에 `support_inquiry.md` 내용 문의
2. 🔄 **높음**: 호환 dx_com 버전 찾아서 재테스트
3. 📦 **중간**: 임시로 CPU 추론 구현 (NAS/서버에서 실행)
4. 🚀 **낮음**: DXRT 업그레이드 계획

### 📝 생성된 파일 목록

- `support_inquiry.md` - DeepX 지원팀 문의 자료
- `strategy_next.md` - 다음 전략 분석
- `analysis_error.md` - 에러 분석
- 다양한 ONNX/DXNN 컴파일 아티팩트

모든 파일이 `/users/lota7574/fire_detection/` 에 저장됨
