# [3] Model Methods & Result

## 3.1. 화재 감지,방범

### **3.1.1. AI Model Architecture**

본 프로젝트는 화재 및 침입을 실시간으로 감지하기 위해 YOLOv7 기반의 객체 인식 모델을 설계하였다. 1인 가구의 안전을 위해 골든타임 내 신속한 대응이 필수적이므로, DeepX NPU를 활용한 온디바이스 추론 아키텍처를 채택하였다.

- Model Selection: Backbone 모델로 화재 감지를 위해 'flame(불꽃)'과 'smoke(연기)' 두 가지 클래스를 구분하는 **YOLOv7** 모델을 선정하였다.
    - **Backbone Selection**: YOLOv7은 ELAN(Efficient Layer Aggregation Network) 구조를 사용하여 파라미터 수 대비 높은 정확도와 추론 속도를 제공하므로, 실시간 화재 감지 백본으로 최적이라 판단하였다.
    - **침입 감지**: 실시간성을 극대화하기 위해 경량화된 **YOLOv8n (COCO)** 모델을 사용하여 'person' 클래스를 고속으로 검출한다.
- Dataset & Training: 가정 내 화재 상황을 반영한 커스텀 데이터셋(Home Fire Dataset)과 공개 화재 데이터셋을 병합(Merged Dataset)하여 다양한 실내/실외, 주간/야간 환경에서의 불꽃과 연기 패턴을 학습하였다. 100 epoch, batch size 16으로 학습한 **v7_merged_100epoch_16batch** 모델이 최종 배포 모델로 선정되었다.
- Hardware Optimization (DeepX NPU Dedicated): 클라우드 서버 통신은 네트워크 지연으로 인한 골든타임 손실과 개인 영상 유출 위험이 있다. 이를 해결하기 위해 DeepX NPU를 전용 가속 장치로 선정하고, **ONNX 포맷으로 변환 후 DXNN 형태로 포팅**을 수행하여 NPU에서 실행 가능하도록 하였다.

### **3.1.2. Custom Algorithm Flow**

정확한 화재 및 침입 감지를 위해 다음과 같은 순차적 알고리즘 흐름을 개발하였다.

- 화재 감지 및 다단계 대응 로직
    - 단일 프레임의 높은 신뢰도만으로는 오탐 가능성이 있으므로, **시간 기반 평균 신뢰도(Time-Averaged Confidence)** 메커니즘을 도입하였다. 최근 일정 시간 창(Time Window) 내 모든 프레임에서 감지된 신뢰도의 평균값을 계산하여, 이 평균값이 임계값을 초과하는 경우에만 실제 위협으로 판단한다. 인접 프레임 간 신뢰도 차이가 크지 않다는 특성을 활용하여, 지속적으로 나타나는 화재는 높은 평균 신뢰도를 유지하고, 순간적으로 오인식된 사물(콘센트, 전구 등)은 낮은 평균 신뢰도로 자동 필터링된다.
    - **평균 신뢰도에 따라 차등적인 알림 등급을 적용하여 (`config.json` 설정 기준)**:
        - **0.20 미만**: 무시 (Monitoring - Noise)
        - **0.20 이상 ~ 0.35 미만 (LOW)**: 주의 (Caution) - 조용한 시각적 알림
        - **0.35 이상 ~ 0.60 미만 (MEDIUM)**: 경고 (Warning) - 사용자 확인 유도
        - **0.60 이상 (HIGH)**: 위험 (Danger) - 긴급 알림 및 대피 권고
    
    [fire2.mp4](attachment:242e3a30-c902-4008-9382-bdc562d9a88d:fire2.mp4)
    
- 침입 판별 로직 (**Dual Engine** + **Robust Presence Scan**)
    
    ![test_image_intrusion_detected.jpg](attachment:ab98a271-6867-4a71-8704-8a42d1d784a2:test_image_intrusion_detected.jpg)
    

- **독립된 YOLOv8 엔진**: 화재 감지와 독립적으로 실행되는 YOLOv8 Nano 모델이 'person' 객체를 검출한다.
- **재실 여부 교차 검증**: 사람이 검출되면 즉시 `user_presence.py`의 **3단계 Robust Scan** (ARP 테이블 → UDP Knocking → ICMP Ping)이 동작하여 거주자의 재실 여부를 정밀 확인한다. (사람 검출 O + 재실 X = 침입)
    - **평균 신뢰도 ≥ 0.60**: 침입 경보 발령 (Alert Threshold)
- 이러한 다중 검증 로직을 통해 문 앞 배달원이나 창밖 지나가는 행인은 오탐 필터링되며, 실제 침입 상황만 정확히 포착한다.

[intrusion2.mp4](attachment:d292362d-23ab-4347-8473-4ec172a318c8:intrusion2.mp4)

### **3.1.3. System Implementation**

최종 시스템 파이프라인은 [NPU 기반 객체 검출 → 시간 기반 평균 신뢰도 계산 → 다단계 알림 등급 판정 → 침입/화재 최종 판별 → 즉각 대응]으로 구성되었다.

- Real-Time NPU Inference: DeepX NPU 가속을 통해 FHD급 홈캠 영상에서도 프레임 저하 없이 실시간으로 불꽃/연기/사람 객체를 추적하였다. 특히 **ThreadPoolExecutor**를 활용한 병렬 추론 파이프라인을 구축하여, 화재(YOLOv7)와 침입(YOLOv8) 두 개의 독립된 NPU 모델을 동시에 실시간으로 구동하면서도 평균 추론 시간 15ms 이하를 달성하였다.
- High Accuracy Detection: 최종 배포 모델(v7_merged_100epoch_16batch)은 검증 데이터셋에서 **mAP@0.5 78.24%**를 달성하였으나, 더 중요한 것은 **학습되지 않은 실제 환경에서 오탐 없이 안정적인 화재 감지 성능**을 보였다는 점이다. Validation 지표가 높더라도 실제 배포 환경에서의 일반화 성능은 떨어질 수 있음을 실증하였으며, 병합 데이터셋 학습과 실전 검증을 통해 다양한 조명 조건, 촬영 각도, 화재 단계에서 신뢰할 수 있는 감지율과 낮은 오탐률을 확보하였다.
- Privacy-Preserving Architecture: 평상시에는 영상이 외부로 전송되지 않으며, **오직 화재나 침입과 같은 실제 위협이 감지되었을 때만** 해당 시점의 이미지를 **HTTPS(TLS) 암호화 채널**을 통해 안전하게 전송하도록 설계하였다. 이를 통해 24시간 감시의 보안 효과는 누리면서도, 개인의 사생활이 담긴 영상이 불필요하게 서버에 저장되거나 유출될 위험을 **최소화**하여 **사용자 편의성과 개인정보 보호의 균형**을 극대화하였다.

## 3.2. 낙상 및 쓰러짐 탐지

### 3.2.1. AI Model Architecture

 낙상 및 쓰러짐 탐지를 수행하기 위해 머리의 하강 속도와 자세를 분석하는 방식을 채택하였으며, 이를 구현하고자 backbone 모델로 **YOLOv5-Pose** 를 사용하였다.  

- Algorithm: 단순히 넘어진 자세만을 판단하는 정적 모델은 취침 상태와 낙상을 구분하기 어렵다는 한계가 있다. 이를 극복하기 위해 **1단계**에서 **YOLOv5-Pose**(NPU)를 통해 인체 관절 좌표를 실시간으로 추출하고, **2단계**(CPU)에서 머리 좌표의 수직 가속도와 자세를 계산 및 분석하였다.
- Hardware Optimization (DeepX NPU Dedicated): 기존의 클라우드 서버 통신 방식은 네트워크 지연으로 인해 **골든타임을 놓칠 위험**이 있으며, 외부로 영상을 전송하는 과정에서 **사생활 유출 우려**가 있다는 명확한 한계가 있다. 이러한 문제를 해결하기 위해 본 시스템은 **DeepX NPU**를 메인 연산 장치로 채택하였다. 특히 **DXNN 형태로 포팅된 모델 파일**을 활용하여 NPU의 고속 병렬 연산을 수행하여, 외부 통신 없이 기기 자체(엣지)에서 실시간 낙상 판별이 가능하도록 구현하였다.

### 3.2.2. Custom Algorithm Flow

정확한 낙상 판별과 오작동 최소화를 위해 다음과 같은 순차적 알고리즘 흐름을 개발하였다.

- Step 1: 객체 검증
    - YOLOv5-Pose가 추출한 객체 중 ‘person’ class 에 대한 Confidence Score(신뢰도)가 0.5 이상인 객체만을 추적 대상으로 한정하였다.
- Step 2: 머리 추적 및 가속도 분석
    - 민감도를 높이기 위해 머리(Head) 위치를 최우선으로 추적한다. 코(Nose), 눈(Eyes), 귀(Ears) 순서의 계층적 우선순위 로직을 적용하여 머리 좌표를 특정하고, 해당 좌표의 Y축 변화 속도(Velocity)가 임계값(Threshold, 1200px/s)을 초과하며 급격히 하강할 때를 낙상 의심 구간으로 1차 판별한다.
- Step 3: 기하학적 자세 분석
    - 가속도만으로는 감지하기 어려운 '스르르 쓰러지는' 상황을 보완하기 위해 신체 비율을 분석한다. 서 있을 때의 Aspect Ratio(머리-허리 길이 대 어깨 너비 비율)가 1.5 이상에서 0.65 미만으로 급격히 역전되거나, 가로 면적(Horizontal Extent)이 비정상적으로 확장(Sprawl)되는 경우를 낙상으로 확정하여 오탐률을 낮췄다.

### 3.2.3. System Implementation

최종 시스템 파이프라인은 [NPU 기반 포즈 추출 → 하이브리드(가속도+자세) 추론 → 즉각적 대응] 이다. 

- Real-Time NPU Inference: DeepX NPU 가속을 통해 고해상도 영상에서도 프레임 저하 없이 다중 객체의 관절과 움직임을 실시간으로 추적했으며, CPU 처리 대비 압도적인 반응 속도를 확보하여 즉각적인 낙상 사고 인지가 가능했다.
- Active Alert System: 낙상 판별 즉시 애플리케이션의 화면에 경보를 울려 초기 대응을 유도함과 동시에 낙상 사진을 첨부하여 오탐지를 빠르게 파악할 수 있게 하여 신뢰도를 높였다.

### 3.2.4. 실행 결과

- **NPU 모니터 내에서의 실행 결과**
    1. 누워 있는 경우 : 낙상을 인식하지 않음. 

![스크린샷 2025-12-01 184927.png](attachment:04d7615f-cbdf-4039-93b0-81ce4548e6f0:스크린샷_2025-12-01_184927.png)

1. 낙상하는 경우 : 낙상으로 성공적으로 인식함. 

![스크린샷 2025-12-01 185115.png](attachment:7c719dac-9e0b-4462-a219-3995754ca500:스크린샷_2025-12-01_185115.png)

![스크린샷 2025-12-01 185153.png](attachment:ba72ab63-fdbb-4cdd-9ab6-a5a9fc6eb175:스크린샷_2025-12-01_185153.png)

[fall_detection (online-video-cutter.com).mp4](attachment:802b964d-d8b1-4ade-a0b0-8d549a12d325:fall_detection_(online-video-cutter.com).mp4)

- **애플리케이션 실행 결과**
    - 낙상 탐지가 수행될 때 (1) 모델 로그(VS code), (2) NPU 에서 실시간으로 처리하는 영상, (3) 휴대폰, (4) 웹사이트 에서 확인할 수 있는 바는 아래 영상과 같다.
        
        낙상 탐지 모델은 실시간으로 관절의 좌표 및 낙상 여부를 판단하며, NPU 영상은 이를 시각적으로 보여준다. 
        
        사용자 및 낙상 위험 1인가구의 보호자가 실제로 사용하는 애플리케이션에서는 평상시에는 경보가 없고 Home monitoring 이 수행되고 있음을 보여주며, 낙상 사고 발생 시 탐지 사진과 함께 경보를 제공한다. 모든 영상을 제공하지 않고, 사진 만을 전송하여 경보에 대한 신뢰성은 높이되, 개인정보 유출은 최소화하도록 하였다. 
        
        낙상 사고 사진과 발생 시각을 메인 화면 및 로그 확인 화면에서 확인가능하도록 설정하여 알림을 놓치더라도 재확인이 가능하다.
        

[falling3.mp4](attachment:02d21e18-a31f-4727-b288-0d22f1f83f10:falling3.mp4)

## 3.3. 수면 관리

### 3.3.1 AI Model Architecture

본 프로젝트는 1인 가구의 수면 상태를 실시간으로 모니터링하기 위해 YOLO-v5-Pose 모델을 선정하고, DEEPX SDK를 적극 활용하여 시스템을 구축하였다.

- Model Selection : 객체 탐지(Object Detection)와 자세 추정(Pose Estimation)을 동시에 수행하는 Single Stage 모델인 YOLOv5-Pose를 채택하였다. 이는 DX-M1 NPU 아키텍처에 맞춰 양자화 및 최적화가 완료된 바이너리이며, 엣지 디바이스 환경에서도 높은 FPS와 정확도를 보장하는 효율적인 모델이다.

### 3.3.2 Custom Algorithm Flow

NPU가 출력한 17개 관절 좌표(Keypoints)를 유의미한 수면 정보로 변환하기 위해 다음과 같은 자체 개발 알고리즘을 적용하였다.

- Data Parsing : NPU의 Raw Tensor 출력에서 사람 객체별 17개 관절의 (x, y, confidence) 좌표를 파싱하였다.
- Posture Classification (자세 판별) : 상단 카메라 뷰의 특성을 고려하여, X축 너비가 아닌 ‘양 어깨의 수직(Y축) 높이 차이’를 계산하고, 이를 임계값(Threshold)와 비교하여 UPRIGHT(정자세)/SIDE(측면)/PRONE(엎드림)을 정밀하게 판별하였다.
- Movement Detection (뒤척임 감지) : 관절의 미세한 떨림을 배제하기 위해 4개 핵심 관절(어깨, 엉덩이)의 평균점인 ‘몸통 중심점’을 계산하고, 이전 프레임과의 이동 거리가 임계값(30px)을 초과할 경우에만 유효한 뒤척임으로 카운트하였다.
    
    ![스크린샷 2025-12-03 001620.png](attachment:e4cfb01d-a910-4cc6-9e4a-b1379b0e5707:스크린샷_2025-12-03_001620.png)
    

### 3.3.3 System Implementation

수면관리 시스템은 [감지 → 분석 → 전송 → 제어]로 구성되며, 4단계 파이프라인을 성공적으로 수행하였다.

- Latency : 사전 최적화된 .dxnn 모델을 사용하여 NPU 추론 속도를 극대화하였으며, 로컬 네트워크 통신을 통해 전체 시스템 지연 시간을 최소화하였다.
- Connectivity : Home Assistant가 MQTT 브로커를 통해 수면 데이터를 수신하고, 설정된 자동화 규칙(빈도 기반 온도 조절)이 정확한 타이밍에 트리거됨을 확인하였다.
    
    ![스크린샷 2025-12-03 001824.png](attachment:369b4295-02de-4370-99f3-52909814c617:스크린샷_2025-12-03_001824.png)
    

---

### 3.3.4 실행 결과

- 애플리케이션 실행 결과

[sleep.mp4](attachment:46fbba51-6c7b-4e13-b456-32c53b93c8ff:sleep.mp4)

본 시연 영상은 NPU 엣지 컴퓨팅과 Home Assistant 연동을 통한 지능형 수면 관리 시스템의 동작 과정을 보여준다.

1. Demo Configuration
    
    실제 환경에서는 입면 후 5분 뒤 수면 모드, 1시간 뒤 숙면 모드로 진입하나, 본 영상에서는 기능 검증을 위해 각 단계를 **10초 간격**으로 빠르게 전환되도록 설정하였다. (시간 설정은 사용자 수면 패턴에 맞춰 커스터마이징 가능)
    
2. 동작 시나리오 및 온도 제어 알고리즘
    
    본 시스템은 **정밀 수면 분석**을 통해 사용자가 ‘숙면(DEEP_SLEEP)’ 상태일 때의 움직임만을 유의미한 데이터로 수집하며, 이에 맞춰 단계별 온도 제어를 수행한다.
    
    **초기 상태 (21°C)**: 사용자가 침대에 누우면 쾌적한 입면을 돕기 위해 21**°**C로 시작
    
    **숙면 모드 전환 (23°C)**: 사용자가 깊은 잠(DEEP_SLEEP)에 들면 체온 유지를 위해 시스템이 자동으로 23**°**C로 온도를 높임
    
    **수면 케어 동작 (22°C)**: 숙면 중 **뒤척임이 3회 이상 감지**되면, 수면 환경이 덥다고 판단하여 온도를 1**°**C 낮춰 뒤척임을 완화하고 수면 질 개선
    
3. 시스템 아키텍처 및 Data flow
    1. **분석 및 전송 (NPU → HA)**
        - NPU가 카메라 영상을 통해 사용자의 수면 자세와 상태를 실시간으로 분석
        - 분석된 메타데이터는 MQTT 프로토콜을 통해 Home Assistant 서버로 전송
    2. **자동화 트리거 (Home Assistant)**
        - 수신된 데이터를 바탕으로 HA의 자동화 엔진이 작동
        - 예: “숙면 상태에서 뒤척임 빈도가 높음” 조건을 감지하여 에어컨 온도를 조절하는 액션 수행
    3. **피드백 및 동기화 (HA → NPU → App)**
        - HA가 처리한 결과(온도, 수면 모드, 상태 메시지 등)는 다시 MQTT를 통해 NPU로 회신
        - NPU는 이 데이터를 받아 app 화면에 즉시 동기화하여 사용자에게 현재 상황을 시각화

## 3.4. Integrated System Architecture

본 프로젝트는 개별 기능(화재/방범, 낙상, 수면)을 단일 시스템으로 통합하고, 유지보수성과 확장성을 확보하기 위해 전체 코드를 모듈화하였다. `integrated_monitor_2.py`에서 시작된 통합 로직은 최종적으로 5개의 독립 모듈(`main.py`, `firedetector.py`, `intruderdetector.py`, `sleepmonitor.py`, `falldetector.py`)로 재구성되어 유기적으로 동작한다.

### 3.4.1. System Flow (Main Logic)

시스템의 전체 흐름은 **User Presence(재실 여부)**에 따라 **HOME Mode**와 **AWAY Mode**로 자동 전환되며, 각 모드에 최적화된 NPU 모델을 동적으로 로드하여 실행한다.

1. **Initialization & Configuration**:
    - `main.py`가 실행되면 `config.json`을 로드하여 임계값 및 모델 경로를 설정한다.
    - `SystemManager` 클래스가 각 서브 모듈(`FireDetector`, `SleepMonitor` 등)을 초기화하고, `DataPusher`와 `RobustPresenceDetector`를 구동한다.

2. **Mode Switching & Model Loading**:
    - **Robust Presence Scan**: 백그라운드 스레드가 ARP/UDP/ICMP 3단계 스캔을 수행하여 사용자의 재실 여부를 실시간 확인한다.
    - **Dynamic Model Loading**:
        - **HOME Mode**: 사용자가 집에 있을 때. `YOLOv5-Pose` 모델을 로드하여 낙상 및 수면 상태를 분석한다.
        - **AWAY Mode**: 사용자가 외출했을 때. `YOLOv8n` (침입 감지)과 `YOLOv7` (화재 감지) 모델을 로드하여 보안 감시를 수행한다.

3. **Inference & Parallel Processing**:
    - FHD급 영상을 `640x640`으로 전처리하여 NPU에 전달한다.
    - **Parallel Pipeline**: AWAY 모드에서는 화재와 침입 감지라는 두 가지 무거운 작업을 동시에 수행해야 한다. 이를 위해 Python의 `ThreadPoolExecutor`를 사용하여 멀티스레드 환경에서 두 NPU 엔진에 대한 추론을 **동시에 요청(Concurrency)**함으로써, 단일 스레드 방식의 순차적 대기(Blocking) 시간을 제거하여 전체 파이프라인의 처리 효율을 극대화하였다. (로그 상 Overlap Ratio로 병렬 효율 확인 가능)

4. **Action & Visualization**:
    - 감지된 결과(화재, 침입, 낙상 등)는 각 Detector 모듈의 `process()` 메서드를 통해 분석된다.
    - 위협 발생 시 `MQTT`를 통해 Home Assistant로 이벤트를 발행(Publish)하고, `HTTPS`로 연결된 Web Backend로 데이터를 전송하여 대시보드에 즉각 반영한다.

### 3.4.2. Modularization Strategy

기존의 단일 파일(`integrated_monitor_2.py`)이 가진 복잡성을 해소하기 위해 다음과 같이 기능을 분리하였다.

- **`main.py`**: 시스템의 진입점(Entry Point). 전체 수명 주기 관리, 모드 전환, NPU 엔진 로딩 및 스레드 풀 관리를 담당한다.
- **`firedetector.py`**: 화재/연기 감지 전용 클래스. 시간 기반 평균 신뢰도 계산 및 알림 등급(LOW/MEDIUM/HIGH) 판정 로직을 포함한다.
- **`intruderdetector.py`**: 침입자 감지 전용 클래스. 시간 기반 필터링 및 쿨다운 로직을 처리한다.
- **`sleepmonitor.py`**: 수면 자세(Upright/Side/Prone) 판별 및 뒤척임 카운트, 일일 리셋 로직을 담당한다.
- **`falldetector.py`**: 머리 하강 속도 및 관절 비율(Aspect Ratio) 분석을 통한 낙상 감지 알고리즘을 수행한다.

이러한 모듈형 아키텍처는 특정 기능(예: 화재 감지 임계값 조정) 수정 시 다른 기능에 영향을 주지 않아 유지보수가 용이하며, 향후 새로운 AI 기능을 추가하기에도 유연한 구조를 갖추고 있다.