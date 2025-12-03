"""
Person Detector Module
YOLOv8을 사용하여 웹캠에서 실시간으로 사람을 감지하는 모듈
"""

import cv2
import torch
from ultralytics import YOLO
from typing import Tuple, List
import time


class PersonDetector:
    """YOLOv8을 사용한 실시간 사람 감지 클래스"""
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Args:
            model_name: YOLO 모델 이름 (yolov8n.pt, yolov8s.pt 등)
            confidence_threshold: 감지 신뢰도 임계값 (0.0 ~ 1.0)
        """
        print(f"[Person Detector] YOLOv8 모델 로딩 중: {model_name}")
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Person Detector] 사용 디바이스: {self.device}")
        
        # COCO 데이터셋에서 사람 클래스는 ID 0
        self.person_class_id = 0
        
    def detect_from_webcam(self, camera_index: int = 0, show_window: bool = True) -> bool:
        """
        웹캠에서 사람을 감지
        
        Args:
            camera_index: 카메라 인덱스 (기본값: 0)
            show_window: 감지 결과를 화면에 표시할지 여부
            
        Returns:
            사람 감지 여부 (True/False)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"[Person Detector] 오류: 카메라 {camera_index}를 열 수 없습니다.")
            return False
        
        print(f"[Person Detector] 카메라 {camera_index} 활성화")
        person_detected = False
        
        try:
            ret, frame = cap.read()
            if not ret:
                print("[Person Detector] 오류: 프레임을 읽을 수 없습니다.")
                return False
            
            # YOLOv8 추론
            results = self.model(frame, device=self.device, verbose=False)
            
            # 사람 감지 확인
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                        person_detected = True
                        print(f"[Person Detector] 사람 감지! (신뢰도: {confidence:.2f})")
                        
                        if show_window:
                            # 바운딩 박스 그리기
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if show_window:
                status_text = "PERSON DETECTED!" if person_detected else "No person"
                color = (0, 0, 255) if person_detected else (0, 255, 0)
                cv2.putText(frame, status_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Person Detection", frame)
                cv2.waitKey(1)
        
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
        
        return person_detected
    
    def detect_continuous(self, camera_index: int = 0, duration: int = 10) -> Tuple[bool, int]:
        """
        지정된 시간 동안 연속으로 사람 감지
        
        Args:
            camera_index: 카메라 인덱스
            duration: 감지 지속 시간 (초)
            
        Returns:
            (사람 감지 여부, 감지된 프레임 수)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"[Person Detector] 오류: 카메라 {camera_index}를 열 수 없습니다.")
            return False, 0
        
        print(f"[Person Detector] {duration}초 동안 연속 감지 시작")
        start_time = time.time()
        person_detected_frames = 0
        total_frames = 0
        
        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                
                total_frames += 1
                
                # YOLOv8 추론
                results = self.model(frame, device=self.device, verbose=False)
                
                # 사람 감지 확인
                frame_has_person = False
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == self.person_class_id and confidence >= self.confidence_threshold:
                            frame_has_person = True
                            person_detected_frames += 1
                            
                            # 바운딩 박스 그리기
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            break
                    
                    if frame_has_person:
                        break
                
                # 화면 표시
                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                status_text = f"Detecting... {remaining}s remaining"
                detection_text = "PERSON DETECTED!" if frame_has_person else "No person"
                
                cv2.putText(frame, status_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, detection_text, (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 0, 255) if frame_has_person else (0, 255, 0), 2)
                
                cv2.imshow("Person Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Person Detector] 사용자에 의해 중지됨")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        detection_rate = (person_detected_frames / total_frames * 100) if total_frames > 0 else 0
        print(f"[Person Detector] 감지 완료: {person_detected_frames}/{total_frames} 프레임 ({detection_rate:.1f}%)")
        
        # 감지율이 10% 이상이면 사람이 있다고 판단
        person_present = detection_rate >= 10
        return person_present, person_detected_frames


def test_detector():
    """감지기 테스트 함수"""
    print("Person Detector 테스트 시작\n")
    
    # 감지기 생성
    detector = PersonDetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    
    print("\n1회 감지 테스트:")
    result = detector.detect_from_webcam(show_window=True)
    print(f"결과: {'사람 감지됨' if result else '사람 없음'}\n")
    
    print("\n10초 연속 감지 테스트 (q를 눌러 중지):")
    person_present, frames = detector.detect_continuous(duration=10)
    print(f"결과: {'사람 존재' if person_present else '사람 없음'} ({frames}프레임 감지)")


if __name__ == "__main__":
    test_detector()
