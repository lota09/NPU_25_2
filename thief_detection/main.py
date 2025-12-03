"""
Thief Detection System - Main Module
ARP 스캔과 YOLOv8 객체 인식을 통합하여 강도 침입을 감지하는 메인 시스템
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, Optional

# Enhanced ARP Scanner 사용 (더 많은 장치 감지)
try:
    from enhanced_arp_scanner import EnhancedARPScanner as ARPScanner
    print("[System] Enhanced ARP Scanner 사용 (Scapy 기반)")
except ImportError:
    from arp_scanner import ARPScanner
    print("[System] 기본 ARP Scanner 사용")

from person_detector import PersonDetector


class ThiefDetectionSystem:
    """강도 침입 감지 통합 시스템"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self.load_config(config_path)
        
        # 모듈 초기화
        self.arp_scanner = ARPScanner(
            network_range=self.config.get("network_range", "192.168.50.0/24"),
            timeout=self.config.get("arp_timeout", 2)
        )
        
        self.person_detector = PersonDetector(
            model_name=self.config.get("yolo_model", "yolov8n.pt"),
            confidence_threshold=self.config.get("detection_threshold", 0.5)
        )
        
        self.trusted_devices = self.config.get("trusted_devices", [])
        self.alert_cooldown = self.config.get("alert_cooldown", 30)  # 초
        self.last_alert_time = 0
        
        print("[Thief Detection] 시스템 초기화 완료")
        print(f"[Thief Detection] 신뢰 장치 수: {len(self.trusted_devices)}")
        
    def load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        if not os.path.exists(config_path):
            print(f"[Thief Detection] 경고: 설정 파일이 없습니다. 기본 설정을 사용합니다.")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"[Thief Detection] 설정 파일 로드 완료: {config_path}")
            return config
        except Exception as e:
            print(f"[Thief Detection] 설정 파일 로드 실패: {e}")
            return {}
    
    def check_intrusion(self) -> tuple[bool, str]:
        """
        침입 여부를 확인
        
        Returns:
            (침입 여부, 상태 메시지)
        """
        print("\n" + "="*70)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 침입 감지 체크 시작")
        print("="*70)
        
        # 1단계: 사용자 기기 존재 확인 (ARP 스캔)
        print("\n[1단계] ARP 스캔으로 사용자 기기 확인 중...")
        user_present = self.arp_scanner.are_trusted_devices_present(self.trusted_devices)
        
        if user_present:
            status = "안전: 사용자 기기가 네트워크에 존재합니다."
            print(f"[침입 감지] {status}")
            return False, status
        
        # 2단계: 사람 감지 (YOLOv8)
        print("\n[2단계] 카메라로 사람 감지 중...")
        person_detected = self.person_detector.detect_from_webcam(
            camera_index=self.config.get("camera_index", 0),
            show_window=self.config.get("show_detection_window", True)
        )
        
        if not person_detected:
            status = "안전: 사용자 부재 중이지만 사람이 감지되지 않았습니다."
            print(f"[침입 감지] {status}")
            return False, status
        
        # 침입 감지!
        status = "⚠️ 경고: 침입자 감지! 사용자 부재 중 사람이 감지되었습니다!"
        print(f"\n{'*'*70}")
        print(f"[침입 감지] {status}")
        print(f"{'*'*70}\n")
        
        return True, status
    
    def trigger_alert(self, message: str):
        """
        경보 발생
        
        Args:
            message: 경보 메시지
        """
        current_time = time.time()
        
        # 쿨다운 체크
        if current_time - self.last_alert_time < self.alert_cooldown:
            remaining = int(self.alert_cooldown - (current_time - self.last_alert_time))
            print(f"[경보] 쿨다운 중... {remaining}초 후 다시 경보 가능")
            return
        
        self.last_alert_time = current_time
        
        print("\n" + "🚨"*30)
        print(f"🚨 침입 경보 🚨")
        print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"메시지: {message}")
        print("🚨"*30 + "\n")
        
        # 여기에 추가 알림 기능 구현 가능
        # - 소리 재생
        # - 모바일 푸시 알림
        # - 이메일 발송
        # - 녹화 시작
        
        # 로그 파일에 기록
        self.log_intrusion(message)
    
    def log_intrusion(self, message: str):
        """침입 로그 기록"""
        log_file = self.config.get("log_file", "intrusion_log.txt")
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{timestamp}] {message}\n")
            print(f"[침입 감지] 로그 기록 완료: {log_file}")
        except Exception as e:
            print(f"[침입 감지] 로그 기록 실패: {e}")
    
    def run_continuous_monitoring(self, interval: int = 30):
        """
        연속 모니터링 모드
        
        Args:
            interval: 체크 간격 (초)
        """
        print("\n" + "="*70)
        print("연속 모니터링 모드 시작")
        print(f"체크 간격: {interval}초")
        print("종료하려면 Ctrl+C를 누르세요")
        print("="*70 + "\n")
        
        try:
            check_count = 0
            intrusion_count = 0
            
            while True:
                check_count += 1
                print(f"\n--- 체크 #{check_count} ---")
                
                intrusion_detected, message = self.check_intrusion()
                
                if intrusion_detected:
                    intrusion_count += 1
                    self.trigger_alert(message)
                
                print(f"\n다음 체크까지 {interval}초 대기 중...")
                print(f"(총 체크: {check_count}회, 침입 감지: {intrusion_count}회)")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n[침입 감지] 모니터링 종료")
            print(f"총 {check_count}회 체크, {intrusion_count}회 침입 감지")
    
    def run_single_check(self):
        """단일 체크 모드"""
        intrusion_detected, message = self.check_intrusion()
        
        if intrusion_detected:
            self.trigger_alert(message)
        
        return intrusion_detected


def main():
    """메인 함수"""
    print("="*70)
    print("강도 침입 감지 시스템")
    print("="*70 + "\n")
    
    # 설정 파일 확인
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"경고: {config_path} 파일이 없습니다.")
        print("기본 설정으로 진행합니다. 신뢰 장치를 설정하려면 config.json을 생성하세요.\n")
    
    # 시스템 초기화
    system = ThiefDetectionSystem(config_path)
    
    # 모드 선택
    print("\n모드를 선택하세요:")
    print("1. 단일 체크 모드 (1회만 확인)")
    print("2. 연속 모니터링 모드 (주기적으로 확인)")
    print("3. ARP 스캔만 실행 (네트워크 장치 확인)")
    print("4. 사람 감지만 실행 (카메라 테스트)")
    
    try:
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == "1":
            print("\n[단일 체크 모드]")
            system.run_single_check()
            
        elif choice == "2":
            interval = input("체크 간격(초, 기본값 30): ").strip()
            interval = int(interval) if interval.isdigit() else 30
            system.run_continuous_monitoring(interval)
            
        elif choice == "3":
            print("\n[ARP 스캔 모드]")
            system.arp_scanner.display_devices()
            print("\n신뢰 장치 확인:")
            system.arp_scanner.are_trusted_devices_present(system.trusted_devices)
            
        elif choice == "4":
            print("\n[사람 감지 모드]")
            print("10초 동안 감지를 실행합니다. (q를 눌러 중지)")
            person_present, frames = system.person_detector.detect_continuous(duration=10)
            print(f"\n결과: {'사람 감지됨' if person_present else '사람 없음'} ({frames}프레임)")
            
        else:
            print("잘못된 선택입니다.")
            
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
