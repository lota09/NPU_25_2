"""
ARP Scanner Module
네트워크 상의 장치를 스캔하고 특정 MAC 주소의 장치가 존재하는지 확인하는 모듈
"""

import subprocess
import re
import platform
import time
from typing import List, Dict, Optional


class ARPScanner:
    """ARP 프로토콜을 사용하여 네트워크 장치를 스캔하는 클래스"""
    
    def __init__(self, network_range: str = "192.168.0.0/24", timeout: int = 2):
        """
        Args:
            network_range: 스캔할 네트워크 범위 (CIDR 표기법)
            timeout: 응답 대기 시간 (초)
        """
        self.network_range = network_range
        self.timeout = timeout
        
    def scan(self) -> List[Dict[str, str]]:
        """
        네트워크를 스캔하여 활성 장치 목록을 반환
        
        Returns:
            장치 정보 딕셔너리 리스트 [{"ip": "192.168.0.1", "mac": "AA:BB:CC:DD:EE:FF"}, ...]
        """
        print(f"[ARP Scanner] 네트워크 스캔 시작: {self.network_range}")
        
        # Windows의 arp 명령어를 사용하는 방법
        if platform.system() == "Windows":
            return self._scan_windows()
        else:
            return self._scan_linux()
    
    def _scan_windows(self) -> List[Dict[str, str]]:
        """Windows에서 arp 테이블과 ping을 사용한 스캔"""
        devices = []
        
        # 먼저 네트워크 범위에서 IP 추출
        # 예: "192.168.50.0/24" -> "192.168.50"
        network_base = '.'.join(self.network_range.split('.')[:3])
        
        # ARP 캐시를 채우기 위해 간단한 ping
        print(f"[ARP Scanner] 네트워크 장치 탐색 중... (게이트웨이 ping)")
        try:
            subprocess.run(
                ["ping", "-n", "1", "-w", "100", f"{network_base}.1"],
                capture_output=True,
                timeout=2
            )
        except:
            pass
        
        # ARP 테이블 조회
        try:
            result = subprocess.run(
                ["arp", "-a"],
                capture_output=True,
                text=True,
                encoding='cp949',  # Windows 한글 인코딩
                timeout=5
            )
            
            # 줄 단위로 처리하되 연속된 줄을 합쳐서 처리
            lines = result.stdout.replace('\r\n', '\n').split('\n')
            
            current_ip = None
            for line in lines:
                line = line.strip()
                
                # IP 주소 찾기
                ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
                if ip_match:
                    potential_ip = ip_match.group(1)
                    # 네트워크 범위 확인
                    if potential_ip.startswith(network_base):
                        current_ip = potential_ip
                
                # MAC 주소 찾기 (여러 형식 지원)
                # 형식: 08-bf-b8-66-d5-10 또는 08:bf:b8:66:d5:10
                mac_match = re.search(r'([0-9a-fA-F]{2}[-:]){5}[0-9a-fA-F]{2}', line)
                if mac_match and current_ip:
                    mac = mac_match.group(0).replace('-', ':').upper()
                    
                    # 멀티캐스트/브로드캐스트 주소 제외
                    if not mac.startswith('01:00:5E') and mac != 'FF:FF:FF:FF:FF:FF':
                        # 중복 확인
                        if not any(d['ip'] == current_ip for d in devices):
                            devices.append({
                                "ip": current_ip,
                                "mac": mac
                            })
                    current_ip = None
            
            print(f"[ARP Scanner] 스캔 완료: {len(devices)}개 장치 발견")
            
            # 디버깅: 발견된 장치 출력
            if devices:
                print(f"[ARP Scanner] 발견된 장치:")
                for d in devices[:5]:  # 처음 5개만
                    print(f"  - {d['ip']}: {d['mac']}")
                if len(devices) > 5:
                    print(f"  ... 외 {len(devices)-5}개")
            
        except Exception as e:
            print(f"[ARP Scanner] ARP 테이블 조회 실패: {e}")
            import traceback
            traceback.print_exc()
        
        return devices
    
    def _scan_linux(self) -> List[Dict[str, str]]:
        """Linux/Mac에서 arp-scan 또는 arp 명령어 사용"""
        devices = []
        
        try:
            # arp-scan 시도 (설치되어 있다면)
            result = subprocess.run(
                ["arp-scan", "-l", "-I", "eth0"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # arp-scan 출력 파싱
            for line in result.stdout.split('\n'):
                parts = line.split()
                if len(parts) >= 2 and re.match(r'\d+\.\d+\.\d+\.\d+', parts[0]):
                    devices.append({
                        "ip": parts[0],
                        "mac": parts[1].upper()
                    })
                    
        except FileNotFoundError:
            print("[ARP Scanner] arp-scan을 찾을 수 없습니다. arp 명령어를 사용합니다.")
            
            # arp 명령어로 대체
            try:
                result = subprocess.run(
                    ["arp", "-n"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                for line in result.stdout.split('\n'):
                    parts = line.split()
                    if len(parts) >= 3 and re.match(r'\d+\.\d+\.\d+\.\d+', parts[0]):
                        devices.append({
                            "ip": parts[0],
                            "mac": parts[2].upper()
                        })
                        
            except Exception as e:
                print(f"[ARP Scanner] arp 명령어 실패: {e}")
        
        except Exception as e:
            print(f"[ARP Scanner] 스캔 실패: {e}")
        
        return devices
    
    def is_device_present(self, mac_address: str) -> bool:
        """
        특정 MAC 주소의 장치가 네트워크에 존재하는지 확인
        
        Args:
            mac_address: 확인할 MAC 주소 (형식: "AA:BB:CC:DD:EE:FF")
            
        Returns:
            장치 존재 여부 (True/False)
        """
        devices = self.scan()
        mac_address = mac_address.upper()
        
        for device in devices:
            if device["mac"] == mac_address:
                print(f"[ARP Scanner] 장치 발견: {mac_address} ({device['ip']})")
                return True
        
        print(f"[ARP Scanner] 장치 미발견: {mac_address}")
        return False
    
    def are_trusted_devices_present(self, trusted_macs: List[str]) -> bool:
        """
        신뢰할 수 있는 장치 중 하나라도 네트워크에 존재하는지 확인
        
        Args:
            trusted_macs: 신뢰할 수 있는 MAC 주소 리스트
            
        Returns:
            신뢰 장치 존재 여부 (True/False)
        """
        devices = self.scan()
        device_macs = {device["mac"] for device in devices}
        trusted_macs_upper = {mac.upper() for mac in trusted_macs}
        
        present_devices = device_macs & trusted_macs_upper
        
        if present_devices:
            print(f"[ARP Scanner] 신뢰 장치 발견: {', '.join(present_devices)}")
            return True
        else:
            print(f"[ARP Scanner] 신뢰 장치 없음 (집에 아무도 없는 상태)")
            return False
    
    def display_devices(self):
        """현재 네트워크의 모든 장치를 출력 (디버깅/설정 용도)"""
        devices = self.scan()
        
        print("\n" + "="*60)
        print("네트워크 장치 목록")
        print("="*60)
        print(f"{'IP 주소':<20} {'MAC 주소':<20}")
        print("-"*60)
        
        for device in devices:
            print(f"{device['ip']:<20} {device['mac']:<20}")
        
        print("="*60)
        print(f"총 {len(devices)}개 장치\n")


def test_scanner():
    """스캐너 테스트 함수"""
    print("ARP Scanner 테스트 시작\n")
    
    # 스캐너 생성 (네트워크 범위는 환경에 맞게 조정)
    scanner = ARPScanner(network_range="192.168.50.0/24")
    
    # 모든 장치 표시
    scanner.display_devices()
    
    # 특정 MAC 주소 테스트 (실제 MAC 주소로 변경 필요)
    test_mac = "AA:BB:CC:DD:EE:FF"
    print(f"\n테스트: {test_mac} 장치 존재 여부 확인")
    scanner.is_device_present(test_mac)


if __name__ == "__main__":
    test_scanner()
