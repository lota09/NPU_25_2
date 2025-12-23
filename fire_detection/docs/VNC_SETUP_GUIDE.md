# Orange Pi 5 Plus VNC 원격 접속 가이드

## 📋 개요
Orange Pi 5 Plus에 VNC 서버를 설치하여 원격 데스크톱 접속을 가능하게 하는 완벽 가이드입니다.

**사양:**
- **대상 기기**: Orange Pi 5 Plus (RK3588)
- **OS**: Ubuntu 22.04 LTS (ARM64)
- **VNC 서버**: TigerVNC (또는 TightVNC)
- **접속 방식**: SSH 터널링 + VNC (보안 권장)

---

## 1️⃣ Orange Pi 서버 설정

### 1.1 SSH로 Orange Pi 접속

```bash
ssh -p 34283 orangepi@221.151.167.152
# 비밀번호: orangepi
```

### 1.2 VNC 서버 설치

#### 옵션 A: TigerVNC 설치 (추천)

```bash
# 패키지 업데이트
sudo apt-get update
sudo apt-get upgrade -y

# TigerVNC 서버 설치
sudo apt-get install -y tigervnc-server tigervnc-common

# 데스크톱 환경 설치 (필요한 경우)
sudo apt-get install -y xfce4 xfce4-goodies
# 또는
sudo apt-get install -y gnome-core
# 또는
sudo apt-get install -y kde-plasma-desktop
```

#### 옵션 B: TightVNC 설치

```bash
sudo apt-get install -y tightvncserver
```

### 1.3 VNC 서버 초기 설정

```bash
# VNC 비밀번호 설정 (첫 실행)
vncserver

# 프롬프트에서:
# - VNC 접속 비밀번호: 예) orangepi2025
# - 읽기 전용 비밀번호: n (아니오 선택)
```

### 1.4 VNC 서버 설정 파일 수정

```bash
# VNC 서버 중지
vncserver -kill :1

# 설정 파일 수정
nano ~/.vnc/xstartup
```

다음 내용으로 수정:

```bash
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
/etc/X11/Xsession

# 또는 XFCE 사용 시:
# startxfce4 &

# 또는 GNOME 사용 시:
# gnome-session &
```

파일 권한 설정:
```bash
chmod +x ~/.vnc/xstartup
```

### 1.5 VNC 서버 자동 시작 설정 (systemd)

```bash
# VNC 서버 유닛 파일 생성
sudo nano /etc/systemd/system/vncserver@.service
```

다음 내용 추가:

```ini
[Unit]
Description=TigerVNC server on %i
After=syslog.target network-online.target remote-fs.target nss-lookup.target
Wants=network-online.target

[Service]
Type=forking
User=orangepi
Group=orangepi
WorkingDirectory=/home/orangepi

ExecStartPre=-/usr/bin/vncserver -kill :%i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver -depth 24 -geometry 1920x1080 :%i
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
```

저장 후:
```bash
# systemd 재로드
sudo systemctl daemon-reload

# VNC 서버 활성화
sudo systemctl enable vncserver@:1.service

# VNC 서버 시작
sudo systemctl start vncserver@:1.service

# 상태 확인
sudo systemctl status vncserver@:1.service
```

### 1.6 VNC 포트 확인

```bash
# VNC 포트 확인 (기본: 5900 + display number)
netstat -tlnp | grep vnc
# 또는
ss -tlnp | grep Xvnc

# 출력 예:
# tcp    0  0 127.0.0.1:5901  0.0.0.0:*  LISTEN  12345/Xvnc
# 포트: 5901 (display :1 = 5900 + 1)
```

### 1.7 방화벽 설정 (선택)

```bash
# VNC 포트 개방
sudo ufw allow 5901
sudo ufw allow 5902
sudo ufw allow 5903

# 상태 확인
sudo ufw status
```

---

## 2️⃣ 클라이언트 설정 (로컬 PC)

### 2.1 VNC 클라이언트 설치

#### Windows
1. VNC Viewer 다운로드: https://www.realvnc.com/en/connect/download/viewer/
2. 설치 후 실행

#### macOS
```bash
brew install vnc-viewer
# 또는 App Store에서 "VNC Viewer" 검색
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install -y vncviewer
# 또는
sudo apt-get install -y tigervnc-viewer
```

---

## 3️⃣ VNC 접속 방법

### 방법 1: SSH 터널링을 통한 보안 접속 (권장)

#### Windows PowerShell / macOS / Linux:

```bash
# SSH 터널 생성 (로컬에서)
ssh -p 34283 -L 5901:127.0.0.1:5901 orangepi@221.151.167.152 -N

# 다른 터미널에서 VNC 클라이언트 시작:
# - Host: localhost:5901
# - Password: vnc서버에서설정한비밀번호
```

#### Windows (PuTTY 사용):
1. PuTTY 열기
2. Session:
   - Host Name: 221.151.167.152
   - Port: 34283
3. SSH → Tunnels:
   - Source port: 5901
   - Destination: 127.0.0.1:5901
   - "Add" 클릭
4. "Open" 클릭 (SSH 연결 유지)
5. VNC Viewer에서 `localhost:5901` 입력

### 방법 2: 직접 접속 (방화벽 개방 필요)

**주의: 보안 위험! SSH 터널링 권장**

```
VNC Viewer에서:
- Host: 221.151.167.152:5901
- Password: vnc서버에서설정한비밀번호
```

### 방법 3: vncviewer 명령어 (Linux/macOS)

```bash
# SSH 터널 생성 (백그라운드)
ssh -p 34283 -L 5901:127.0.0.1:5901 orangepi@221.151.167.152 -N &

# VNC 연결
vncviewer localhost:5901
```

---

## 4️⃣ 자동화 스크립트

### 4.1 Orange Pi 자동 설정 스크립트

`setup_vnc_server.sh` 생성:

```bash
#!/bin/bash

echo "=== VNC Server Setup for Orange Pi ==="

# 패키지 업데이트
echo "Updating packages..."
sudo apt-get update
sudo apt-get upgrade -y

# TigerVNC 설치
echo "Installing TigerVNC..."
sudo apt-get install -y tigervnc-server tigervnc-common

# 데스크톱 환경 설치
echo "Installing XFCE4 desktop..."
sudo apt-get install -y xfce4 xfce4-goodies

# VNC 디렉토리 생성
mkdir -p ~/.vnc

# VNC 비밀번호 자동 설정 (예)
echo "Setting VNC password..."
# 자동 설정 (예시, 실제로는 대화형 입력 필요)
# echo "orangepi2025" | vncpasswd -f > ~/.vnc/passwd

# xstartup 파일 생성
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
/etc/X11/Xsession
EOF

chmod +x ~/.vnc/xstartup

# systemd 유닛 파일 생성
sudo tee /etc/systemd/system/vncserver@.service > /dev/null << 'EOF'
[Unit]
Description=TigerVNC server on %i
After=syslog.target network-online.target remote-fs.target nss-lookup.target
Wants=network-online.target

[Service]
Type=forking
User=$USER
Group=$USER
WorkingDirectory=/home/$USER

ExecStartPre=-/usr/bin/vncserver -kill :%i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver -depth 24 -geometry 1920x1080 :%i
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
EOF

# systemd 재로드 및 활성화
sudo systemctl daemon-reload
sudo systemctl enable vncserver@:1.service
sudo systemctl start vncserver@:1.service

echo "=== VNC Server Setup Complete ==="
echo "VNC Server running on port 5901"
echo "Please set VNC password manually: vncpasswd"
```

실행:
```bash
chmod +x setup_vnc_server.sh
./setup_vnc_server.sh
```

### 4.2 로컬 PC 자동 SSH 터널 스크립트

#### Bash/zsh (macOS/Linux):

`vnc_connect.sh`:
```bash
#!/bin/bash

REMOTE_HOST="221.151.167.152"
REMOTE_PORT="34283"
REMOTE_USER="orangepi"
LOCAL_PORT="5901"

echo "Starting SSH tunnel to $REMOTE_HOST:$REMOTE_PORT..."
ssh -p $REMOTE_PORT -L $LOCAL_PORT:127.0.0.1:5901 $REMOTE_USER@$REMOTE_HOST -N
```

실행:
```bash
chmod +x vnc_connect.sh
./vnc_connect.sh

# 다른 터미널에서 VNC 클라이언트 시작
vncviewer localhost:5901
```

#### Windows PowerShell:

`vnc_connect.ps1`:
```powershell
$REMOTE_HOST = "221.151.167.152"
$REMOTE_PORT = "34283"
$REMOTE_USER = "orangepi"
$LOCAL_PORT = "5901"

Write-Host "Starting SSH tunnel to $REMOTE_HOST`:$REMOTE_PORT..."
ssh -p $REMOTE_PORT -L "${LOCAL_PORT}:127.0.0.1:5901" "$REMOTE_USER@$REMOTE_HOST" -N
```

실행:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\vnc_connect.ps1

# 다른 PowerShell 창에서 VNC Viewer 실행
```

---

## 5️⃣ 문제 해결

### VNC 서버가 시작되지 않는 경우

```bash
# 로그 확인
sudo journalctl -u vncserver@:1.service -n 50

# 수동 시작 테스트
vncserver -depth 24 -geometry 1920x1080 :1

# 디버그 모드
vncserver -verbose :1
```

### 포트 충돌

```bash
# 사용 중인 포트 확인
sudo lsof -i :5901

# 프로세스 종료
sudo kill -9 <PID>

# VNC 강제 종료
vncserver -kill :1
vncserver -kill :2
```

### SSH 터널 문제

```bash
# SSH 연결 테스트
ssh -p 34283 orangepi@221.151.167.152 "echo OK"

# 터널 연결 테스트
ssh -p 34283 -L 5901:127.0.0.1:5901 orangepi@221.151.167.152 -N -v

# 로컬 포트 바인딩 확인
netstat -tlnp | grep 5901
```

### 데스크톱 환경 문제

```bash
# XFCE 설정 재설정
rm -rf ~/.config/xfce4
xfce4-panel --restart

# 또는 직접 xterm으로 테스트
vncserver -kill :1
nano ~/.vnc/xstartup
# xterm & 추가
vncserver :1
```

---

## 6️⃣ 보안 강화

### 6.1 VNC 비밀번호 변경

```bash
vncpasswd

# 또는 읽기 전용 비밀번호도 설정
vncpasswd -o
```

### 6.2 SSH 키 기반 인증 설정

```bash
# 이미 SSH 키로 접속 중인 경우 추가 설정 불필요
# SSH 키만 사용하고 비밀번호는 비활성화
ssh-keygen -t ed25519
```

### 6.3 VNC 바인딩을 localhost만으로 제한

`/etc/systemd/system/vncserver@.service` 수정:
```ini
ExecStart=/usr/bin/vncserver -localhost -depth 24 -geometry 1920x1080 :%i
```

그 후:
```bash
sudo systemctl restart vncserver@:1.service
```

---

## 7️⃣ 빠른 접속 요약

### Orange Pi 설정 (일회성):
```bash
ssh -p 34283 orangepi@221.151.167.152
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y tigervnc-server xfce4 xfce4-goodies
vncserver  # 비밀번호 설정
vncserver -kill :1
```

### 매번 접속 (로컬 PC):
```bash
# 터미널 1: SSH 터널
ssh -p 34283 -L 5901:127.0.0.1:5901 orangepi@221.151.167.152 -N

# 터미널 2: VNC 클라이언트 실행
vncviewer localhost:5901
# 또는 VNC Viewer GUI에서 localhost:5901 입력
```

---

## 📝 참고 사항

- **기본 VNC 포트**: 5900 + display 숫자 (display :1 = 5901)
- **권장 해상도**: 1920x1080, 1280x720
- **권장 색상 깊이**: 24비트
- **성능**: SSH 터널은 약간의 지연 가능 (보안 > 속도)
- **다중 접속**: 여러 display (:1, :2, :3...) 생성 가능

---

## 🔗 유용한 링크

- [TigerVNC 공식 문서](https://tigervnc.org/)
- [RealVNC 공식](https://www.realvnc.com/)
- [Ubuntu VNC 가이드](https://ubuntu.com/blog/ubuntu-remote-desktop-using-vnc)

