## 📝 디버깅 기록: 3가지 주요 문제 해결

### 🔴 문제 1: 모델 출력 형식 문제

#### 1️⃣ 어떤 문제가 있었나
- YOLOv7 모델의 기본 export 설정에서 3개 분리된 출력 생성
  - Output 1: [1, 3, 80, 80, 7] (scale 1)
  - Output 2: [1, 3, 40, 40, 7] (scale 2)
  - Output 3: [1, 3, 20, 20, 7] (scale 3)
- 이 다중 스케일 출력 형식이 DXNN 컴파일의 SURGERY 단계에서 호환되지 않음

#### 2️⃣ 어떤 시도를 해보았는가
- export_force.py 스크립트를 통한 직접 ONNX 변환
- `model.model[-1].export = False` 설정으로 concatenated 출력 시도
  - 결과: ✅ 단일 출력 [1, 25200, 7] 형식으로 변환 성공

#### 3️⃣ 어느 순간 깨달았는가
- export_force.py 수정 중, `model.model[-1].export = False` 설정으로 출력 형식이 [1,25200,7]로 변환되는 것을 확인
- 원인: YOLOv7의 기본 export는 FPN 3개 스케일별 분리 출력
- DXNN SURGERY 단계가 이 다중 출력을 처리하지 못하고 실패

#### 4️⃣ 어떻게 해결했는가
- `export_force.py`에서 `model.model[-1].export = False` 설정 적용
- ONNX 재export로 단일 concatenated 출력 [1, 25200, 7] 생성
  - 결과: ✅ 단일 출력 형식으로 변환 성공, DXNN 컴파일 준비 완료

---

### 🔴 문제 2: "Unwanted Data Type is inserted in GetDataSize.NONE_TYPE" 컴파일 에러

#### 1️⃣ 어떤 문제가 있었나
- 모델 출력 형식 문제를 해결한 후, DXNN 컴파일 CALIBRATION 단계에서 실패
- 에러: "Unwanted Data Type is inserted in GetDataSize.NONE_TYPE"
- INT8, FP32, NHWC, UINT8 모든 포맷과 quantization 설정에서 동일하게 발생

#### 2️⃣ 어떤 시도를 해보았는가
- SiLU → LeakyReLU 활성화 함수 교체 시도 (NPU가 SiLU 미지원으로 오해)
  - 결과: ❌ 여전히 같은 에러
- NHWC 포맷 변환 + Transpose 노드 추가
  - 결과: ❌ CALIBRATION 단계에서 실패
- Quantization 설정 변경 (INT8 → FP32)
  - 결과: ❌ 여전히 컴파일 불가
- 공식 YOLOv7 ONNX 모델(Orange Pi ModelZoo)로 재테스트
  - 결과: ❌ 역시 같은 에러 발생
- NHWC UINT8 입력 형식으로 직접 export
  - 결과: ❌ 모든 시도가 CALIBRATION 단계에서 실패

#### 3️⃣ 어느 순간 깨달았는가
- Orange Pi에 기본 제공되는 YOLOv7 DXNN 모델들은 정상 작동 ✅
- 예제 모델 로드 정보에서 발견: "No graphinfo (1)" 경고 메시지
- 버전 정보 확인:
  - 로컬 dx_com: v2.0.0 (최신)
  - Orange Pi DXRT: v2.9.5 (상대적 구 버전)
- **핵심 힌트**: DXRT 최소 Compiler 요구사항 v1.18.1 기록
  - v2.0.0은 충분하지만, 생성되는 DXNN 파일 포맷이 v2.9.5와 호환되지 않을 가능성 발견
- 사용자 제시: `~/dx_com_M1_v1.60.1` 대체 컴파일러 존재 확인

#### 4️⃣ 어떻게 해결했는가
- dx_com v1.60.1로 컴파일 재시도
- 컴파일 성공:
  - FP32: 2분 33초 (v2.0.0: 12분 40초)
  - INT8: 2분 38초
  - 파일 크기: 71MB (동일)

**근본 원인:**
- dx_com v2.0.0이 생성하는 DXNN 파일 포맷이 DXRT v2.9.5와 호환되지 않음
- **결론**: 최신 버전이 항상 좋은 것은 아님, 대상 런타임과의 호환성 확인 필수

---

### 🔴 문제 3: NPU 포팅 실패 (DXNN 로드 불가)

#### 1️⃣ 어떤 문제가 있었나
- 로컬 dx_com v2.0.0으로 컴파일한 DXNN이 Orange Pi에서 로드 불가
- INT8: "Unwanted Data Type is inserted in GetDataSize.NONE_TYPE" 에러 즉시 발생
- FP32/UINT8: 타임아웃 (10-15초 응답 없음)
- 같은 입력 형식의 공식 예제 모델은 정상 작동

#### 2️⃣ 어떤 시도를 해보았는가
**초기 시도:**
- 여러 ONNX 포맷 변환 (NCHW, NHWC, UINT8)
  - 결과: 모두 로드 불가 또는 타임아웃
- calibration 데이터 생성 및 설정
  - 결과: 여전히 로드 불가
- Orange Pi에서 직접 dx_com 컴파일 시도 검토
  - 판단: ARM에서는 비효율적 (포기)

**분석 단계:**
- 예제 ONNX 원본 찾아서 로컬에서 재컴파일
  - 결과: 역시 Orange Pi에서 타임아웃
- DXNN 파일 버전/포맷 메타데이터 확인
  - 발견: "No graphinfo (1)" 경고

#### 3️⃣ 어느 순간 깨달았는가
**단서 발견 과정:**
1. **비교 분석**: 
   - 예제 YOLOv7 DXNN: ✅ 로드 성공 (1.3초)
   - 우리 모델 DXNN: ❌ 로드 실패
   - 차이점: 컴파일 도구의 버전?

2. **버전 메타데이터 확인**:
   - dx_com v2.0.0 내부 정보
   - dx_codegen: v2.33.1 (매우 최신)
   - DXRT v2.9.5 요구사항 vs 실제 컴파일러 버전

3. **대안 발견**:
   ```bash
   사용자 제시: ~/dx_com_M1_v1.60.1 존재
   → 버전 1.60.1은 DXRT 요구사항 v1.18.1과 더 가까움
   → 호환성 가능성 높음
   ```

#### 4️⃣ 어떻게 해결했는가
**해결 방법:**
- dx_com v1.60.1로 모델 재컴파일

**결과 - FP32 모델:**
```
✅ 모델 로드 성공 (0.26초)
   Task[0]: CPU 전처리
   Task[1]: NPU 추론 (275MB 메모리)
   Task[2]: CPU 후처리
✅ 추론 성공 (63ms/frame)
```

**결과 - INT8 모델:**
```
✅ 모델 로드 성공 (0.26초)
✅ 추론 성공 (51ms/frame) - FP32보다 19% 빠름 ⭐
```

**성능 비교:**
| 항목 | FP32 | INT8 |
|------|------|------|
| 로드 시간 | 0.26s | 0.26s |
| 추론 시간 | 63ms | **51ms** |
| NPU 활용 | ✅ | ✅ |
| 메모리 | 275MB | 275MB |

**근본 원인:**
- dx_com v2.0.0 (최신) 사용 → 호환 불가
- dx_com v1.60.1 (중간 버전) 사용 → 호환 성공
- **결론**: 버전 호환성이 NPU 포팅의 핵심

---

## 🎯 핵심 교훈

### ❌ 실수한 부분
1. **최신 버전 = 최고 호환성** 이라는 오해
2. **SiLU 활성화 함수가 문제**라고 잘못 진단
3. **입력 형식만 바꾸면 된다**고 가정

### ✅ 올바른 접근
1. **대상 런타임의 요구사항 먼저 확인**
   - DXRT 버전 및 Compiler 최소/권장 사항
2. **이미 작동하는 예제 분석**
   - 예제 모델과의 비교로 차이점 발견
3. **버전 호환성을 우선 의심**
   - 포맷/기능 오류보다 버전 불일치 가능성 높음

### 📊 최종 결과
```
문제 1 (모델 출력 형식): 해결 ✅
  원인: 3개 분리 출력 → DXNN SURGERY 호환 불가
  해결: export=False로 단일 출력 [1,25200,7]로 변환

문제 2 (CALIBRATION 에러): 해결 ✅
  원인: dx_com v2.0.0 → DXNN 포맷 호환성 이슈
  해결: v1.60.1 사용

문제 3 (NPU 포팅 실패): 해결 ✅
  원인: 컴파일 도구 버전 불일치
  해결: v1.60.1 사용 + NPU 추론 성공
  
전체: 성공 🎉
  INT8 모델: 51ms/frame (권장)
  FP32 모델: 63ms/frame (검증용)
```
