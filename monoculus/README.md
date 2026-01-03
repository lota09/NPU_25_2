# Monoculus: NPU Integrated Security System

이 문서는 **Monoculus** 시스템의 실행 방법, 구조, 기능 및 설정 방법을 팀원들에게 안내하기 위해 작성되었습니다.

### 사전 준비 (최초 1회)

시스템 실행 전, 필요한 파이썬 패키지를 설치해야 합니다.

```bash
cd ~/monoculus
pip install -r requirements.txt
```

**⚠ 트러블슈팅: 윈도우에서 복사한 파일 줄바꿈 문제 해결**
윈도우에서 파일을 수정 후 오렌지 파이로 옮겼을 때, 실행 스크립트(`run_monoculus.sh`)에서 `\r` 관련 에러가 발생할 수 있습니다. 아래 명령어로 해결하세요.

```bash
# 스크립트의 윈도우식 줄바꿈(\r\n)을 유닉스식(\n)으로 변환
sed -i 's/\r$//' run_monoculus.sh
chmod +x run_monoculus.sh
```

### 실행 방법 (`run_monoculus.sh`)

Monoculus 시스템은 `monoculus` 디렉토리 전체를 오렌지 파이(Orange Pi)로 전송하여 실행합니다.

```bash
# 로컬 터미널에서 실행 (팀원 공통: 사용자명 orangepi)

# 1. SSH 세션 별칭(Alias)을 설정하지 않은 경우 (IP 직접 입력)
scp -r monoculus orangepi@123.123.123.123:~/

# 2. SSH 세션 별칭을 설정한 경우 (예: 'orangepi'로 설정 시)
scp -r monoculus orangepi:~/
```

### 실행 방법 (`run_monoculus.sh`)

파일 전송 후, 오렌지 파이에 접속하여 스크립트를 실행합니다. 이 스크립트는 백엔드(`app.py`)와 모니터링 코어(`integrated_monitor_2.py`)를 자동으로 관리합니다.

```bash
cd ~/monoculus

# GUI 없이 실행 (추천: SSH 원격 환경 또는 성능 최적화 시)
./run_monoculus.sh --nogui

# GUI 모드로 실행 (개발/디버깅 시, VNC 등 필요)
./run_monoculus.sh
```

*   **종료 방법**: 터미널에서 `Ctrl+C`를 누르면 두 프로세스가 안전하게 종료됩니다.
*   **강제 종료**: 만약 프로세스가 남았다면 `pkill -f python3` 등을 사용하세요.

---

## 2. 디렉터리 구조

*   **`monoculus/`**
    *   `run_monoculus.sh`: 전체 시스템 실행 스크립트.
    *   `integrated_monitor_2.py`: **시스템의 핵심**. NPU를 사헤 감지(화재, 침입자, 낙상) 및 데이터를 처리합니다.
    *   `user_presence.py`: 네트워크 스캔(ARP)을 통한 재실 감지 모듈.
    *   `app.py`: 웹 UI를 위한 백엔드 서버(Flask). 데이터 릴레이 및 이미지 서빙 역할을 합니다.
    *   `config.json`: 시스템 설정 파일 (임계값, MAC 주소 등).
    *   **`static/`**: 웹 프론트엔드 리소스.
        *   `index.html`: 통합 대시보드 UI (Alpine.js 기반).
        *   `images/`: 실시간 감지된 이미지가 저장되는 곳.

---

## 3. 통합 코드 및 시스템 로직

### NPU 통합 감지 (`integrated_monitor_2.py`)
*   **병렬 처리 아키텍처**: `ThreadPoolExecutor`를 사용하여 **화재/연기 감지 모델**과 **사람/포즈 감지 모델**을 병렬로 처리합니다. 테스트 결과, 이를 통해 두 모델을 동시에 구동하면서도 안정적인 멀티태스킹이 가능함을 확인하였습니다.
*   **DataPusher**: 감지된 데이터를 1초 단위(알림 발생 시) 또는 주기적으로 `app.py`로 전송합니다.
*   **Sticky Logic**: 화재 감지 시 일시적으로 불꽃이 가려져도 5초간 알림 상태를 유지하여 UI 깜빡임을 방지합니다.

### 재실 감지 모듈 (`user_presence.py`)
사용자의 스마트폰이 Wi-Fi에 연결되어 있는지를 통해 재실 여부를 판단합니다. 단순히 Ping만 보내는 것이 아니라, 3단계 **Robust Scan** 로직을 수행합니다.

1.  **Passive Check**: 시스템의 ARP 테이블(`ip neighbor show`)을 조회하여 최근 통신 기록이 있는지 확인합니다.
2.  **UDP Knocking**: ARP 테이블에 정보가 없거나 오래된(STALE) 경우, UDP 패킷을 브로드캐스트하여 기기를 '깨우고' 응답을 유도합니다.
3.  **Active Probe**: 마지막으로 등록된 IP에 Ping을 보내 실제 연결 상태를 최종 검증합니다.

### 데이터 릴레이 및 통신 (`app.py` & Frontend)
프론트엔드와 백엔드는 **HTTP 기반**으로 통신하며, 다음과 같은 흐름을 가집니다:

1.  **Push (Monitor -> Backend)**:
    *   `integrated_monitor_2.py`가 `/api/update`로 **POST** 요청을 보냅니다.
    *   **Payload 예시**:
        ```json
        {
          "fire": {"detected": true, "type": "Fire", "level": "MEDIUM", "conf": 0.45},
          "intruder": {"detected": false},
          "is_home": true
        }
        ```
2.  **Polling (Frontend -> Backend)**:
    *   `index.html` (Alpine.js)이 1초마다 `/api/status`로 **GET** 요청을 보냅니다.
    *   백엔드는 최신 시스템 상태를 JSON으로 반환하며, 프론트엔드는 이를 받아 UI를 렌더링합니다.
    *   **Heartbeat**: 프론트엔드는 백엔드 응답의 `last_heartbeat` 시간을 체크하여, 일정 시간(15초) 동안 NPU 데이터가 없으면 'DISCONNECTED' 상태로 전환합니다.

---

## 4. 웹 UI 기능

사용자가 요청한 주요 기능들이 구현되어 있습니다:

*   **상태별 동적 디자인**:
    *   **재실 중 (HOME)**: 🟢 초록색 테마 + 🏠 집 아이콘. "User detected..."
    *   **부재 중 (AWAY)**: 🔵 파란색 테마 + 🛡️ 방패 아이콘. "User away..."
    *   **연결 끊김**: "DISCONNECTED" 전용 오버레이 화면.
*   **위급 상황 알림 (ALERT)**:
    *   **화재**: 심각도에 따라 **CAUTION(노랑) / WARNING(주황) / DANGER(빨강)**으로 구분됨.
    *   **침입자**: 🌑 검정색(Dark) 테마. "Person detected in your monitoring zone..."
    *   **낙상**: 🔴 빨간색 테마 + "PRIVACY MASKING" 적용된 이미지.
*   **실시간 카메라 피드**:
    *   알림 발생 시 중앙에 실제 현장 이미지가 **1초 간격**으로 갱신되며 표시됩니다. (Cache-busting 적용) (평소에는 숨겨져 있음)
    *   우측 상단 **' LIVE FEED'** 배지는 항상 **빨간색**으로 고정되어 위급함을 알립니다.

---

## 5. 설정 가이드 (`config.json`)

**⚠ 중요: 재실 판단(HOME/AWAY)이 정확히 작동하려면 네트워크 설정이 필수입니다.**

`config.json` 파일을 열어 다음 항목을 수정하세요. 실제 파일 구조와 동일하게 작성해야 합니다:

```json
{
  "system": {
    "arp_interface": "wlP2p33s0",
    "arp_interval": 10,
    "subnet_prefix": "192.168.50", <-- 네트워크 설정에 따라 변경
    "force_mode": "AUTO",
    "away_timeout": 300
  },
  "trusted_devices": [
    "C2:77:BE:99:90:A1", <-- 스마트폰 MAC 주소 등록
    "1A:95:E8:88:9E:BA",
    "11:22:33:44:55:66"
  ],
  "models": {
    "home_pose": "/home/orangepi/deepx_sdk/dx_app/assets/models/YOLOV5Pose640_1.dxnn",
    "away_detect": "/home/orangepi/deepx_sdk/dx_app/assets/models/YoloV8N.dxnn",
    "fire_detect": "/home/orangepi/deepx_sdk/dx_app/assets/models/v7_opset12_proper.dxnn"
  },
  "mqtt": {
    "host": "192.168.219.101", <-- MQTT broker IP
    "port": 1883, <-- MQTT broker port
    "topic": "home/security",
    "ha_topic": "home/ha_status"
  },
```

*   **네트워크 인터페이스 확인법**: 터미널에 `ip addr` 또는 `ifconfig`를 입력하여 본인의 무선 인터페이스 이름(예: `wlan0`, `wlP2p33s0` 등)을 확인하고 `arp_interface` 값을 수정하세요. (기기마다 다를 수 있음)
*   **MAC 주소 확인법**: 스마트폰 설정 -> 휴대전화 정보 -> 상태 -> Wi-Fi MAC 주소.
*   **설정 적용**: 파일을 저장하고 `run_monoculus.sh`를 재시작하면 적용됩니다.

---

## 6. 향후 계획 (To-Do)

1.  **웹 UI에서 사용자 정보 전달**: 웹 UI 접속 시 사용자의 IP와 MAC 주소를 감지하여 백엔드로 전달, 자동 등록 기능 검토.
2.  **백엔드 코드 모듈화**: `app.py`의 라우팅 및 상태 관리 로직을 분리하여 유지보수성 향상.
3.  **재실 시 웹 UI 동작 검증**: 실제 '재실(HOME)' 상태에서의 다양한 시나리오(화재 감지 등)에 대한 UI 반응성 정밀 테스트 필요.
