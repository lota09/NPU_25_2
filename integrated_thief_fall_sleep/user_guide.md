# 통합 보안 및 수면 모니터링 시스템 사용자 가이드

이 문서는 NPU(Orange Pi), Home Assistant(HA), 그리고 사용자 앱을 연동하여 시스템을 구축하고 운영하는 방법을 설명합니다.

## 1. 시스템 개요 (Architecture)
시스템은 크게 3가지 요소로 구성됩니다.
1.  **NPU (Orange Pi)**: 카메라로 사람을 감지(YOLO/Pose)하고, 데이터를 수집하여 MQTT로 방송합니다. (데이터 생성자)
2.  **Home Assistant (HA)**: NPU의 데이터를 받아 에어컨 등 가전제품을 제어하고, 제어 결과를 다시 NPU로 알립니다. (제어자)
3.  **사용자 앱 (App)**: NPU가 방송하는 데이터를 구독하여 실시간 상태를 보여줍니다. (뷰어)

## 2. NPU 설정 및 실행 (Python)

### 필수 요구사항
*   **H/W**: Orange Pi 5 (또는 DeepX NPU 탑재 기기), USB 웹캠
*   **S/W**: Python 3.8+, `deepx_sdk`, `opencv-python`, `numpy`, `paho-mqtt`

### 실행 방법
터미널에서 아래 명령어로 메인 프로그램을 실행합니다.
```bash
sudo python integrated_monitor_2.py
```
> `sudo`는 `arp-scan` (재실 감지) 기능을 위해 필요합니다.

### 설정 파일 (`config.json`)
`config.json` 파일을 통해 시스템 동작을 변경할 수 있습니다.

| 항목 | 설명 | 예시 값 |
| :--- | :--- | :--- |
| `mqtt.host` | MQTT 브로커(노트북/서버)의 IP 주소 | `"192.168.0.101"` |
| `trusted_devices` | 재실 감지할 가족들의 스마트폰 MAC 주소 | `["AA:BB:CC:..."]` |
| `models` | AI 모델 파일(.dxnn)의 절대 경로 | Check file |

> [!IMPORTANT]
> 스마트폰의 **"랜덤 MAC(비공개 주소)"** 기능을 꺼야 재실 감지가 정확히 작동합니다.

## 3. Home Assistant (HA) 연동

HA가 NPU와 통신하려면 아래 설정이 필요합니다.

### 1) Configuration.yaml
NPU가 보내는 데이터를 센서로 등록합니다.
```yaml
mqtt:
  sensor:
    - name: "수면 상태"
      state_topic: "home/security"
      value_template: "{{ value_json.status }}"
    - name: "뒤척임 횟수"
      state_topic: "home/security"
      value_template: "{{ value_json.moves }}"
input_boolean:
  deep_sleep_mode:
    name: "Deep Sleep Mode"
```

### 2) Automations.yaml (자동화)
*   **입면 감지 (1단계)**: 자세가 `SIDE`/`PRONE`/`UPRIGHT`면 10초 후 `Deep Sleep Mode` 켬.
*   **스마트 온도 조절 (2단계)**: `Deep Sleep Mode`가 켜져 있고 뒤척임이 많으면 온도 조절.
*   **기상 감지 (0단계)**: 침대에서 벗어나면 `Deep Sleep Mode` 끔.
*   **피드백**: 모든 액션은 `home/ha_status` 토픽으로 NPU에 결과를 전송해야 합니다.

## 4. 사용자 앱 (App) 개발 가이드

앱은 복잡한 로직 없이 **MQTT 구독(Subscribe)**만 하면 됩니다.

### 연결 정보
*   **Protocol**: MQTT over TCP
*   **IP/Port**: `config.json`의 `mqtt.host`와 동일 / `1883`
*   **Topic**: `home/security` (이것만 구독하면 됩니다!)

### 수신 데이터 형식 (JSON)
NPU는 아래와 같은 JSON을 1초마다, 또는 이벤트 발생 시 전송합니다.

```json
{
  "mode": "HOME",                  // 현재 모드 (HOME / AWAY)
  "status": "SIDE",                // 수면 자세 (SIDE, PRONE, UPRIGHT, N/A)
  "moves": 12,                     // 현재 뒤척임 횟수 (누적)
  "fall": false,                   // 낙상 감지 여부 (비상!)
  "intruder": false,               // 침입자 감지 여부 (비상!)
  "ha": {                          // [HA 피드백 데이터] (없으면 null)
    "msg": "Temp Adjusted (+1C)",  // 알림 메시지
    "temp": 24,                    // 현재 에어컨 온도
    "status": "DEEP_SLEEP",        // 수면 단계 (SLEEP_ENTRY / DEEP_SLEEP / WAKE_UP)
    "time": "2026-01-02 23:55:00", // 이벤트 발생 시간
    "moves": 12                    // 해당 시점의 뒤척임 횟수
  }
}
```

### 앱 개발 팁
*   `ha` 필드가 `null`이 아닐 때만 하단 로그창이나 토스트 메시지를 띄우세요.
*   `ha.moves`와 `ha.temp`, `ha.time`을 저장하면 **"시간대별 수면 온도 및 뒤척임 그래프"**를 그릴 수 있습니다.
