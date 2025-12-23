# Fire Detection ONNX/DXNN 문제 해결 요약 (2025-12-18)

## 문제 인식
- PyTorch 모델은 fire/no-fire(negative) 이미지를 구별 가능
- ONNX/DXNN 변환 후, negative(예: thief.jpg, room2.jpg)에서도 fire가 오탐지됨
- threshold(임계값) 조정, 후처리 변경만으로 오탐이 완전히 사라지지 않음
- DXNN 변환 및 NPU 추론에서도 negative에서 confidence 1.0 fire box 검출

## 시도한 방법
- 다양한 ONNX export 옵션(NCHW/NHWC, LeakyReLU, opset 등) 실험
- dev/compare_pt_onnx.py로 PyTorch/ONNX 결과 직접 비교 및 threshold sweep
- 후처리 방식(and, objectness만, class만, max 등) grid search
- room2.jpg 등 완전 negative 이미지로도 테스트
- DXNN 변환 후 orangepi에서 실제 NPU 추론
- 임계값 0.3~0.99 sweep, 후처리 조합, config 파일 등 반복 실험
- DXNN, 소스, 이미지 자동 scp/ssh 연동 및 실험 자동화

## 시도를 통해 알아낸 것
- ONNX sweep(numpy)에서는 일부 negative에서 오탐이 없었으나, 실제 NPU 추론에서는 confidence 1.0 fire box가 검출됨
- threshold를 아무리 높여도 negative에서 오탐이 완전히 사라지지 않음(모델 한계)
- 후처리 방식만으로는 해결 불가, 모델 자체가 negative에서 fire로 강하게 오탐함
- config 파일, 경로, DXNN 컴파일러 환경 등은 모두 정상적으로 관리됨

## 향후 계획
- 모델 재학습(negative/edge case 데이터 보강, 하드마이닝 등) 또는 아키텍처 개선 필요
- 후처리 추가 실험(예: fire+smoke 동시 조건, NMS, 박스 필터링 등)
- ONNX/DXNN 변환 후 결과 이미지(박스 시각화) 자동 저장 기능 추가
- 실험/환경/이슈는 ENVIRONMENT_LOG.md, SUMMARY.md 등으로 계속 기록 및 관리
- 사용자 요청 시 threshold, 후처리, 변환 옵션 등 추가 실험 즉시 진행

---

> 이 문서는 fire_detection 프로젝트의 ONNX/DXNN 문제 해결 과정을 요약한 최신 상태입니다.
> 추가 요청/실험/이슈 발생 시 계속 갱신합니다.
