# 🔥 YOLOv5S vs YOLOv7/Fire Detection 문제 분석

## 📊 테스트 결과 (2025-12-16)

```
YOLOv5S (작동함):
  RAW Objectness:  [-0.0000, 0.3821]
  After Sigmoid:   [0.5000, 0.5944]
  Valid (>0.5):    24995/25200 (99.2%)

YOLOv7 (작동 안함):
  RAW Objectness:  [-0.0000, 0.8426]
  After Sigmoid:   [0.5000, 0.6990]
  Valid (>0.5):    20398/25200 (81%)

Fire Detection (작동 안함) - best_npu_v1601_fp32.dxnn:
  RAW Confidence:  [-0.0000, 0.4488]
  After Sigmoid:   [0.5000, 0.6099]
  Fire Prob:       [0.0002, 1.0000] ✅ (이미 Sigmoid 적용됨!)
  Output Format:   4개 (multi-scale + concatenated)
```

## 🔍 핵심 발견

### 1. **세 모델 모두 Confidence가 비정상적으로 낮음**
```
일반적인 detection:  logit 3~5 범위 → Sigmoid → 0.95~0.99 신뢰도
현재 모델들:         logit -0.x~0.x 범위 → Sigmoid → 0.5~0.6 신뢰도
```

### 2. **Fire Detection의 특별한 점**
- Fire Probability는 이미 **Sigmoid가 적용된 상태** (0-1 범위, max=1.0)
- Confidence는 여전히 raw logit
- **다른 post-processing 방식을 사용하는 모델**

### 3. **YOLOv5S vs YOLOv7 비교**
| 항목 | YOLOv5S | YOLOv7 |
|------|---------|--------|
| Confidence RAW | 0.3821 | 0.8426 |
| Sigmoid 후 | 0.5944 | 0.6990 |
| Valid (>0.5) | 99.2% | 81% |
| 결과 | 작동함 | 작동 안함 |

→ **Raw 범위가 클수록 false positive가 많음**

## 🎯 다음 확인할 사항

### YOLOv7 테스트 결과 대기
- Objectness 범위는?
- Sigmoid 후 범위는?
- Valid detection 수는?

### Fire Detection 테스트 결과 대기
- Multi-scale 출력 처리 상태
- Objectness 범위는?

## 📋 근본 원인 가설

### **가설 1: CPU Post-processing 누락**
- YOLOv5S/YOLOv7: 3개 multi-scale output (80x80, 40x40, 20x20) → CPU가 concatenate
- 이 과정에서 **Sigmoid 정규화가 안될 수도 있음**
- Raw logit 그대로 concatenate되고 있을 가능성

### **가설 2: 모델 훈련 설정 문제**
- YOLOv7/Fire 모델: 특정 scaling factor가 적용되어 logit이 다른 범위
- DXNN 변환 과정에서 normalization layer 누락

### **가설 3: DXRT 런타임 버그**
- 특정 모델 조합에서 post-processing이 제대로 작동하지 않음

## 🎯 다음 확인할 사항

### **원인 가설 (업데이트됨)**

#### **가설 1: CPU Post-processing에서 Sigmoid 누락** ✅ 확인됨
- YOLOv5S/YOLOv7/Fire: 3개 multi-scale output → CPU concatenate
- CPU에서 **Sigmoid를 적용하지 않음** (Fire Detection은 일부 채널만 적용)
- 결과: raw logit 그대로 출력

#### **가설 2: 모델 학습 설정 차이**
- YOLOv5S: max logit 0.38
- YOLOv7: max logit 0.84 (약 2배 높음)
- Fire: max logit 0.45
→ **각 모델이 다른 scaling factor로 훈련됨**

#### **가설 3: DXNN 변환 최적화 문제**
- CPU post-processing이 완전하지 않음
- Sigmoid layer가 모두 NPU에만 들어가 있음
- CPU decoding 단계에서 미싱된 Sigmoid

## ✅ 해결책

## ✅ 해결책

### **현재 발견 (Critical)**

**YOLOv7 모델의 Class logit이 완전히 비정상:**
```
Objectness logit:      max = 36
Person class logit:    max = 44  ← 이상!
```

심지어 **사람이 없는 상태에서도** person class logit이 44에 도달하며, Sigmoid(44) = 1.0

**결론: YOLOv7 모델 자체가 훼손되었거나 잘못 변환되었음**

### **즉시 적용 가능한 방법**

#### **Option 1: YOLOv5S 사용** (권장)
- YOLOv5S가 더 안정적 (logit 범위가 작음)
- person_detection_monitor.py에서 모델 경로만 변경

#### **Option 2: 모델 재훈련/재변환**
- DXNN 컴파일러 버전 확인 (현재 v1.60.1)
- PyTorch 모델에서 weight 확인
- ONNX export 옵션 검토

#### **Option 3: 탐지 로직 개선**
- Objectness와 class logit의 scale 차이 고려
- 더 엄격한 필터링 (예: `valid_mask = (objectness > T1) & (class_conf > T2) where T2 >> T1`)
- NMS 후처리 추가

### **추천 다음 단계**
1. **YOLOv5S로 전환** - 즉시 문제 해결 가능
2. **불 탐지 모델 테스트** - Fire detection도 같은 문제인지 확인
3. **필요시 모델 재변환** - DXNN 옵션 조정

---

**기억할 것:**
- YOLOv5S도 완전히 정상은 아님 (거의 모든 anchor가 threshold 통과)
- 근본 원인은 **모델 출력 자체의 이상함**
- 삽질하지 말고 이 문서에서 출발하기!
