# NPU 포팅 문제 해결 및 작업 가이드

## 1. 디렉터리 구조 요약

```
fire_detection/
├── calibration_dataset/   # ONNX/DXNN 변환용 calibration 데이터
├── docs/                  # 문서 및 가이드
│   └── npu_porting/       # NPU 포팅 관련 요약/문제/해결 문서
├── export_force.py        # ONNX 변환 스크립트
├── fire_detection_monitor.py # NPU 추론/테스트 메인 스크립트
├── v7_merged_100epoch_16batch.pt # 변환 대상 PyTorch 모델
├── yolov7_fire_int8_v1601.json   # DXNN 변환 config 예시
├── yolov7_fire_nchw_fp32.json    # DXNN 변환 config 예시
```

> 대용량 데이터/임시/실험/불필요 파일은 .gitignore로 관리되어 팀원 clone 시 제외됨

## 2. 문제 상황 및 요약
- 목표: fire_detection/v7_merged_100epoch_16batch.pt → ONNX 변환 → DXNN 컴파일 → NPU에서 화재 감지 정상 동작
- **DXNN surgery error 등 컴파일 문제는 현재 해결됨.**
- **핵심 문제는 모델 변환 및 출력값의 비정상적 분포**
- ONNX/DXNN 변환 후 negative(불 없음) 이미지에서도 fire box가 confidence 1.0로 검출되는 오탐 발생
- threshold/후처리만으로 오탐 완전 제거 불가, 모델 변환/후처리 과정의 한계
- 문제 및 해결 과정은 docs/DEBUGGING_LOG.md, docs/npu_porting/DEBUG_SUMMARY.md, SUMMARY.md 참고

## 3. 시도해본 것들
- 다양한 ONNX export 옵션(NCHW/NHWC, LeakyReLU, opset 등) 실험: 변환은 성공했으나, 오탐 문제는 해결되지 않음
- dev/compare_pt_onnx.py로 PyTorch/ONNX 결과 직접 비교 및 threshold sweep: ONNX 변환 후 일부 negative 이미지에서 오탐이 사라지기도 했으나, 실제 NPU 추론에서는 여전히 confidence 1.0 fire box가 검출됨
- 후처리 방식(and, objectness만, class만, max 등) grid search: threshold/후처리 조합을 바꿔도 오탐 완전 제거 불가
- room2.jpg 등 완전 negative 이미지로도 테스트: negative에서도 fire box 검출됨
- DXNN 변환 후 orangepi에서 실제 NPU 추론: 오탐 현상 동일하게 발생
- 임계값 0.3~0.99 sweep, 후처리 조합, config 파일 등 반복 실험: threshold를 아무리 높여도 오탐이 완전히 사라지지 않음
- DXNN, 소스, 이미지 자동 scp/ssh 연동 및 실험 자동화: 실험 환경은 정상적으로 관리됨

시도 해본 내용들에 대한 자세한 내용은 다음 문서 참조할것 :
[DEBUG_SUMMARY.md](DEBUG_SUMMARY.md), [PROBLEM_ANALYSIS.md](PROBLEM_ANALYSIS.md)

## 4. 팀원 작업 가이드
1. v7_merged_100epoch_16batch.pt → onnx 변환: export_force.py 사용
   - 예시 명령:
     ```bash
     python export_force.py --weights v7_merged_100epoch_16batch.pt --output models/v7_merged_100epoch_16batch.onnx
     ```
2. onnx → dxnn 컴파일: dx_com v1.60.1 필요 (변환이 아닌 "컴파일"임)
   - 컴파일러는 리눅스 환경에서만 동작
   - 공용서버(lota7574@snuserver) 접속 후 사용 권장
   - 컴파일러 위치: ~/dx_com_M1_v1.60.1
   > dx_com 컴파일러가 있는 서버 아이피/계정 정보는 별도 공지 예정
   - 예시 명령:
     ```bash
     ~/dx_com_M1_v1.60.1/dx_com \
       -m models/v7_merged_100epoch_16batch.onnx \
       -c yolov7_fire_nchw_fp32.json \
       -o models/best_npu_nhwc.dxnn \
       --calib calibration_dataset/
     ```
3. dxnn 파일과 소스코드 파일을 NPU로 옮길 때 scp 명령 예시:
   ```bash
   scp models/best_npu_nhwc.dxnn fire_detection_monitor.py orangepi@NPU_IP:/home/orangepi/fire_detection/
   ```
4. dxnn 파일로 NPU에서 fire_detection_monitor.py 실행 예시:
   ```bash
   python fire_detection_monitor.py --model best_npu_nhwc.dxnn --conf 0.5 --source 0
   ```
   - 정상: 불 있을 때만 감지, 없으면 감지X
   - threshold/후처리만으로 오탐이 완전히 사라지지 않으면 컴파일/후처리 옵션을 재검토
   - 컴파일 과정에서 surgery error 등 문제가 발생할 경우, 반드시 docs/compile/SUCCESS_FINAL.md의 내용을 참고할 것

- 원격 개발(VNC): mobaxterm 등 VNC 클라이언트로 NPU 화면 접속 추천
    - VNC 서버 구축법은 docs/VNC_SETUP_GUIDE.md 참고

## 5. 참고 문서
- docs/DEBUGGING_LOG.md : 문제 발생/해결 상세 기록
- docs/npu_porting/ : SUMMARY.md, DEBUG_SUMMARY.md, PROBLEM_ANALYSIS.md
- docs/VNC_SETUP_GUIDE.md : VNC 서버/클라이언트 활용법

---