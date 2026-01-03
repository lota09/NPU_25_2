# Monoculus AI 홈 모니터링 시스템 사용 설명서

이 문서는 **Monoculus 시스템**을 실행하는 방법과, 다른 환경(다른 PC 또는 NPU 보드)으로 옮길 때 변경해야 할 설정을 안내합니다.

---

## 🚀 1. 시스템 실행 방법

이 프로젝트는 편의를 위해 **통합 실행 스크립트**를 제공합니다.

### 실행 명령어

터미널(Terminal)을 열고 프로젝트 폴더로 이동한 뒤 아래 명령어를 입력하세요.

```bash
# 기본 실행 (모든 시스템 시작)
./run_monoculus.sh

# 백그라운드 모드 (GUI 창 없이 실행)
./run_monoculus.sh --nogui
```

### 실행 확인
1. **웹 대시보드 접속**: [http://localhost:5000](http://localhost:5000)
   - 모바일 등 외부 기기 접속 시: `http://<NPU_IP>:5000` (예: `http://192.168.219.101:5000`)
2. **로그 확인**: `monoculus.log` 파일에 실행 로그가 저장됩니다.

---

## 🛠️ 2. 다른 기기(NPU)로 이전 시 변경해야 할 것들

시스템을 다른 하드웨어나 네트워크 환경으로 옮길 경우, **`config.json`** 파일을 반드시 수정해야 합니다.

### 📁 설정 파일: `config.json`

아래 항목들을 새 환경에 맞게 변경해주세요.

#### 1) 모델 경로 변경 (`models`)
NPU마다 파일 시스템 구조가 다를 수 있습니다. `.dxnn` 모델 파일이 저장된 **절대 경로**를 지정해야 합니다.

```json
"models": {
    "home_pose": "/새로운/경로/YOLOV5Pose640_1.dxnn",
    "away_detect": "/새로운/경로/YoloV8N.dxnn",
    "fire_detect": "/새로운/경로/v7_opset12_proper.dxnn"
}
```

#### 2) 네트워크 인터페이스 (`system`)
재실 감지(ARP 스캔)를 위해 **와이파이 인터페이스 이름**이 정확해야 합니다.
- **확인 방법**: 터미널에 `ifconfig` 또는 `ipconfig` 입력 후 사용 중인 인터페이스 이름 확인 (예: `wlan0`, `eth0`, `wlP2p33s0` 등).

```json
"system": {
    "arp_interface": "wlan0",  <-- 여기를 수정하세요
    ...
}
```

#### 3) MQTT 브로커 주소 (`mqtt`)
Home Assistant가 설치된 서버(또는 PC)의 IP 주소로 변경해야 합니다.

```json
"mqtt": {
    "host": "192.168.0.200",   <-- Home Assistant IP로 변경
    ...
}
```

#### 4) 사용자 휴대폰 MAC 주소 (`trusted_devices`)
자동으로 외출/재실 모드를 전환하기 위해 사용자의 휴대폰 MAC 주소를 등록해야 합니다.

```json
"trusted_devices": [
    "AA:BB:CC:DD:EE:FF"        <-- 사용자 폰 MAC 주소 추가
]
```

---

## 📦 3. 필수 설치 환경 (Dependencies)

새로운 기기에는 다음 소프트웨어와 라이브러리가 설치되어 있어야 합니다.

1. **Python 3.8 이상**
2. **필수 라이브러리**:
   ```bash
   pip install flask opencv-python paho-mqtt numpy
   ```
3. **NPU 런타임 (DX/Hailo 등)**:
   - 사용 하드웨어에 맞는 NPU 드라이버와 SDK(`dx-runtime` 등)가 시스템 경로에 설정되어 있어야 합니다.
   - ⚠️ **주의**: 모델 파일(`.dxnn`)은 해당 하드웨어용 컴파일러로 변환된 버전이어야 합니다.

---

## 🏠 4. Home Assistant 연동 (선택 사항)

HA와 연동하려면 `automations.yaml` 파일의 내용을 HA 설정에 추가해야 합니다.
- **파일 위치**: `./HA_yaml/automations.yaml`
- **적용 방법**: 내용을 복사하여 HA의 `automations.yaml`에 붙여넣거나, 파일을 그대로 덮어쓰기 후 HA 재시작.
- **실제 사용법**: 현재는 automations.yaml 코드 안의 수면 모드 전환 및 숙면 모드 전환이 시연용으로 10초로 되어있음. 실제 사용 시 수면 모드 전환 5분, 숙면 모드 전환 1시간으로 수정하여 사용
---
궁금한 점이 있다면 언제든 문의해주세요!
