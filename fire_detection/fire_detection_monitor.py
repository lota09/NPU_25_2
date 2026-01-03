#!/usr/bin/env python3
"""
Fire Detection Real-time Monitoring Script
카메라 또는 비디오 입력에서 화재를 감지하고 시간 기반 평균 신뢰도로 다단계 알림을 실행합니다.
Orange Pi DXNN 모델 사용
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from typing import Optional, Tuple

# OpenCV 백엔드 설정 (디스플레이가 있을 때는 xcb 사용)
if os.environ.get('DISPLAY') is None:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
from scipy.special import expit  # sigmoid 함수

try:
    from dx_engine import InferenceEngine
except ImportError:
    print("Warning: dx_engine not available. Running in CPU mode.")
    InferenceEngine = None


# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FireDetectionMonitor:
        """시간 기반 평균 신뢰도를 사용하는 화재 감지 모니터"""
        
        # 알림 등급 정의 (Sigmoid 정규화 후 [0, 1] 범위)
        # Raw logit을 Sigmoid로 정규화한 신뢰도 기준
        ALERT_LEVEL = {
            'MONITORING': (0.00, 0.35, '✅ 정상'),
            'LOW': (0.35, 0.50, '🟡 주의'),
            'MEDIUM': (0.50, 0.65, '🟠 경고'),
            'HIGH': (0.65, 1.00, '🔴 긴급 대피')
        }
        
        def __init__(
            self,
            model_path: str,
            video_source: str = '/dev/video0',
            time_window: float = 3.0,
            input_size: Tuple[int, int] = (640, 640),
            conf_threshold: float = 0.5,
            use_dxnn: bool = True
        ):
            """
            Args:
                model_path: DXNN 모델 파일 경로 (.dxnn)
                video_source: 비디오 소스 (카메라 또는 비디오 파일)
                time_window: 평균 신뢰도 계산 시간 윈도우 (초)
                input_size: 모델 입력 크기
                conf_threshold: 감지 신뢰도 임계값
                use_dxnn: DXNN 사용 여부
            """
            self.model_path = model_path
            self.video_source = video_source
            self.time_window = time_window
            self.input_size = input_size
            self.conf_threshold = conf_threshold
            self.use_dxnn = use_dxnn and InferenceEngine is not None
            
            # 모델 로드
            self.engine = None
            if self.use_dxnn:
                self._load_dxnn_model()
            
            # 비디오 캡처 초기화
            self.cap = None
            self.fps = 30
            self._init_video_capture()
            
            # 신뢰도 이력 (최대 시간 윈도우에 해당하는 프레임 수)
            self.confidence_history = deque(maxlen=int(self.fps * self.time_window))
            self.timestamp_history = deque(maxlen=int(self.fps * self.time_window))
            
            # 상태 추적
            self.current_alert_level = 'MONITORING'
            self.last_alert_time = 0
            self.alert_duration = 2.0  # 알림 지속 시간 (초)
            
            logger.info(f"Fire Detection Monitor initialized")
            logger.info(f"  - Model: {model_path}")
            logger.info(f"  - Video source: {video_source}")
            logger.info(f"  - Time window: {time_window}s")
            logger.info(f"  - Use DXNN: {self.use_dxnn}")
        
        def _load_dxnn_model(self):
            """DXNN 모델 로드"""
            try:
                logger.info(f"Loading DXNN model: {self.model_path}")
                start_time = time.time()
                self.engine = InferenceEngine(self.model_path)
                load_time = time.time() - start_time
                logger.info(f"✅ Model loaded in {load_time:.3f}s")
            except Exception as e:
                logger.error(f"Failed to load DXNN model: {e}")
                self.use_dxnn = False
        
        def _init_video_capture(self):
            """비디오 캡처 초기화"""
            try:
                # 카메라 또는 비디오 파일 열기
                if self.video_source.isdigit() or self.video_source.startswith('/dev/video'):
                    # 카메라 (숫자 또는 /dev/video0 형식)
                    device_index = int(self.video_source) if self.video_source.isdigit() else self.video_source
                    self.cap = cv2.VideoCapture(device_index if isinstance(device_index, int) else self.video_source)
                    logger.info(f"✅ Camera opened: {self.video_source}")
                    is_camera = True
                else:
                    # 비디오 파일
                    self.cap = cv2.VideoCapture(self.video_source)
                    logger.info(f"✅ Video file opened: {self.video_source}")
                    is_camera = False
                
                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError(f"Cannot open video source: {self.video_source}")
                
                # FPS 정보 획득
                self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
                logger.info(f"  - FPS: {self.fps}")
                
                # 해상도 정보 획득
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"  - Resolution: {width}x{height}")
                
                # 카메라인 경우 추가 정보
                if is_camera:
                    logger.info(f"  - Camera Type: {self.cap.get(cv2.CAP_PROP_BACKEND)}")
                    logger.info(f"  - Codec: {self.cap.get(cv2.CAP_PROP_FOURCC)}")
                
            except Exception as e:
                logger.error(f"Failed to initialize video capture: {e}")
                sys.exit(1)
        
        def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
            """
            프레임 전처리 (YOLO 형식)
            - 리사이징
            - Float32 정규화
            - NCHW 포맷 변환
            """
            # 입력 프레임 검증
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return np.zeros((1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
            
            # 리사이징 (아스펙트 비율 유지하며 패딩)
            h, w = frame.shape[:2]
            scale = min(self.input_size[0] / w, self.input_size[1] / h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 검은색 배경에 패딩
            padded = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            pad_y = (self.input_size[1] - new_h) // 2
            pad_x = (self.input_size[0] - new_w) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            # BGR -> RGB
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            
            # NCHW 포맷으로 변환 (Float32로 정규화)
            img_array = np.asarray(rgb, dtype=np.float32)
            img_array = img_array.transpose(2, 0, 1) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # 배치 추가
            
            logger.debug(f"Preprocessed - shape: {img_array.shape}, dtype: {img_array.dtype}, "
                        f"min: {img_array.min():.4f}, max: {img_array.max():.4f}")
            
            return img_array
        
        def infer(self, frame: np.ndarray) -> np.ndarray:
            """
            프레임에서 화재 감지 추론 수행
            
            Returns:
                predictions: 모델 출력 [1, 25200, 7] (YOLO 형식)
                            각 열: [x, y, w, h, confidence, class_0, class_1]
            """
            if self.engine is None:
                logger.warning("Model not available, skipping inference")
                return np.zeros((1, 25200, 7), dtype=np.float32)
            
            try:
                # 전처리
                preprocessed = self.preprocess_frame(frame)
                
                # 추론 (DXNN)
                predictions = self.engine.run([preprocessed])
                
                if isinstance(predictions, list):
                    predictions = predictions[0]
                
                # 디버그: 예측 형태 확인
                if predictions.size > 0:
                    logger.debug(f"Raw predictions shape: {predictions.shape}")
                    if len(predictions.shape) == 3:
                        logger.debug(f"  Max confidence in raw: {np.max(predictions[:, :, 4]):.4f}")
                
                return predictions
            
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return np.zeros((1, 25200, 7), dtype=np.float32)
        
        def extract_max_confidence(self, predictions: np.ndarray) -> float:
            """
            예측에서 최대 화재 신뢰도 추출
            
            **중요**: Sigmoid 정규화를 먼저 적용한 후,
            정규화된 [0, 1] 범위에서 threshold를 적용합니다.
            """
            if predictions.size == 0 or len(predictions.shape) < 3:
                return 0.0
            
            try:
                # Shape 확인
                logger.debug(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
                
                # UINT8 데이터인 경우 float32로 변환 및 정규화 (0-255 → 0-1)
                if predictions.dtype == np.uint8:
                    predictions = predictions.astype(np.float32) / 255.0
                    logger.debug(f"Converted UINT8 to float32 and normalized to [0, 1]")
                
                num_channels = predictions.shape[-1]
                
                if num_channels == 7:
                    # 화재 감지 모델 (2 클래스)
                    objectness = predictions[0, :, 4]
                    fire_confidences = predictions[0, :, 5]
                
                elif num_channels == 85:
                    # COCO 모델 (80 클래스) 또는 일반 YOLOv7
                    objectness = predictions[0, :, 4]
                    class_confidences = predictions[0, :, 5:]
                    fire_confidences = np.max(class_confidences, axis=1)
                
                else:
                    logger.warning(f"Unknown output format with {num_channels} channels")
                    return 0.0
                
                # 원본 raw logit 값 분석
                logger.debug(f"Objectness (RAW logit) - min: {objectness.min():.2f}, max: {objectness.max():.2f}, "
                           f"mean: {objectness.mean():.2f}")
                logger.debug(f"Fire confidence (RAW logit) - min: {fire_confidences.min():.2f}, max: {fire_confidences.max():.2f}, "
                           f"mean: {fire_confidences.mean():.2f}")
                
                # ⚠️ 테스트: 아까와 동일하게 RAW LOGIT에 threshold 적용 (Sigmoid 없이)
                valid_mask = (objectness > self.conf_threshold) & (fire_confidences > self.conf_threshold)
                
                num_valid = np.sum(valid_mask)
                logger.debug(f"Valid detections (RAW logit - objectness > {self.conf_threshold} AND fire_conf > {self.conf_threshold}): {num_valid}")
                
                if not np.any(valid_mask):
                    logger.debug(f"No valid detections (RAW logit mode)")
                    return 0.0
                
                # 유효한 detection의 최대 fire confidence (Sigmoid 후)
                fire_sigmoid = expit(fire_confidences.astype(np.float64))
                max_confidence = float(np.max(fire_sigmoid[valid_mask]))
                logger.debug(f"Max fire confidence (after Sigmoid): {max_confidence:.4f}")
                
                return max_confidence
            
            except Exception as e:
                logger.warning(f"Error extracting confidence: {e}")
                return 0.0
        
        def get_time_averaged_confidence(self) -> float:
            """시간 기반 평균 신뢰도 계산"""
            if not self.confidence_history:
                return 0.0
            
            return float(np.mean(list(self.confidence_history)))
        
        def determine_alert_level(self, avg_confidence: float) -> str:
            """평균 신뢰도에 따른 알림 등급 결정"""
            for level, (min_conf, max_conf, msg) in self.ALERT_LEVEL.items():
                if min_conf <= avg_confidence < max_conf:
                    return level
            return 'HIGH'  # >= 0.65
        
        def log_alert(self, avg_confidence: float, alert_level: str):
            """알림 로그 출력"""
            current_time = time.time()
            
            # 알림 레벨이 변경되었거나 충분한 시간이 경과했을 때만 로그
            if (alert_level != self.current_alert_level or 
                current_time - self.last_alert_time > self.alert_duration):
                
                self.current_alert_level = alert_level
                self.last_alert_time = current_time
                
                min_conf, max_conf, msg = self.ALERT_LEVEL[alert_level]
                logger.info(
                    f"{msg} | "
                    f"Avg Confidence: {avg_confidence:.4f} | "
                    f"Detections: {len(self.confidence_history)}"
                )
        
        def draw_info_on_frame(
            self,
            frame: np.ndarray,
            current_confidence: float,
            avg_confidence: float,
            alert_level: str
        ) -> np.ndarray:
            """프레임에 정보 및 알림 표시"""
            frame_display = frame.copy()
            h, w = frame_display.shape[:2]
            
            # 배경 (반투명)
            overlay = frame_display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame_display, 0.7, 0, frame_display)
            
            # 텍스트 색상 (알림 등급별)
            alert_colors = {
                'MONITORING': (200, 200, 200),  # 회색
                'LOW': (0, 165, 255),            # 주황색
                'MEDIUM': (0, 255, 255),         # 노랑
                'HIGH': (0, 0, 255)              # 빨강
            }
            color = alert_colors.get(alert_level, (200, 200, 200))
            
            # 정보 표시
            y_offset = 30
            cv2.putText(
                frame_display,
                f"Current Conf: {current_confidence:.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            y_offset += 30
            cv2.putText(
                frame_display,
                f"Avg Conf ({self.time_window}s): {avg_confidence:.4f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            y_offset += 30
            min_conf, max_conf, alert_msg = self.ALERT_LEVEL[alert_level]
            alert_msg_short = alert_msg.split('|')[0].strip()
            cv2.putText(
                frame_display,
                f"Alert: {alert_msg_short}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # 알림 등급별 테두리
            thickness = 3
            cv2.rectangle(frame_display, (0, 0), (w-1, h-1), color, thickness)
            
            return frame_display
        
        def run(self, display: bool = True, output_video: Optional[str] = None):
            """
            실시간 화재 감지 모니터링 실행
            
            Args:
                display: 화면에 표시할지 여부
                output_video: 결과를 저장할 비디오 파일 경로 (None이면 저장 안 함)
            """
            logger.info("🎬 Starting fire detection monitoring...")
            
            # 비디오 저장 설정
            writer = None
            if output_video:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    output_video,
                    fourcc,
                    self.fps,
                    (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                )
                logger.info(f"📹 Output video will be saved to: {output_video}")
            
            frame_count = 0
            inference_times = deque(maxlen=30)
            
            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.info("End of video or camera disconnected")
                        break
                    
                    frame_count += 1
                    current_time = time.time()
                    
                    # 추론
                    infer_start = time.time()
                    predictions = self.infer(frame)
                    infer_time = time.time() - infer_start
                    inference_times.append(infer_time)
                    
                    # 신뢰도 추출
                    current_confidence = self.extract_max_confidence(predictions)
                    
                    # 이력에 저장
                    self.confidence_history.append(current_confidence)
                    self.timestamp_history.append(current_time)
                    
                    # 평균 신뢰도 계산 및 알림 결정
                    avg_confidence = self.get_time_averaged_confidence()
                    alert_level = self.determine_alert_level(avg_confidence)
                    
                    # 알림 로그
                    self.log_alert(avg_confidence, alert_level)
                    
                    # 프레임에 정보 표시
                    frame_with_info = self.draw_info_on_frame(
                        frame,
                        current_confidence,
                        avg_confidence,
                        alert_level
                    )
                    
                    # 화면 표시
                    if display:
                        cv2.imshow('Fire Detection Monitor', frame_with_info)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("User requested exit")
                            break
                    
                    # 비디오 저장
                    if writer:
                        writer.write(frame_with_info)
                    
                    # 주기적 통계 출력
                    if frame_count % (self.fps * 5) == 0:  # 5초마다
                        avg_infer_time = np.mean(list(inference_times))
                        logger.info(
                            f"[Frame {frame_count}] "
                            f"Avg Infer: {avg_infer_time*1000:.2f}ms | "
                            f"FPS: {1/avg_infer_time:.1f}"
                        )
            
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            
            finally:
                logger.info("🛑 Shutting down...")
                self.cap.release()
                if writer:
                    writer.release()
                if display:
                    cv2.destroyAllWindows()
                logger.info(f"Total frames processed: {frame_count}")


def main():
        parser = argparse.ArgumentParser(
            description="Fire Detection Real-time Monitoring"
        )
        parser.add_argument(
            '--model',
            type=str,
            required=True,
            help='Path to DXNN model file (.dxnn)'
        )
        parser.add_argument(
            '--video',
            type=str,
            default='/dev/video0',
            help='Video source (camera: /dev/video0, or video file path)'
        )
        parser.add_argument(
            '--time-window',
            type=float,
            default=3.0,
            help='Time window for averaging confidence (seconds)'
        )
        parser.add_argument(
            '--conf-threshold',
            type=float,
            default=0.5,
            help='Confidence threshold for detection'
        )
        parser.add_argument(
            '--output',
            type=str,
            default=None,
            help='Output video file path (optional)'
        )
        parser.add_argument(
            '--no-display',
            action='store_true',
            help='Disable display window'
        )
        
        args = parser.parse_args()
        
        # 모니터 실행
        monitor = FireDetectionMonitor(
            model_path=args.model,
            video_source=args.video,
            time_window=args.time_window,
            conf_threshold=args.conf_threshold
        )
        
        monitor.run(
            display=not args.no_display,
            output_video=args.output
        )


if __name__ == '__main__':
    main()
