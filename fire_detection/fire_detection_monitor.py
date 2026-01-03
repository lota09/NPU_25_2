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
from typing import Optional, Tuple, List, Union

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
        
        # LeakyReLU Density Filter Config
        DENSITY_THRESHOLD = 800  # Fire > 1800, Room < 500. Margin > 500. Safe 800.
        
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
            - 리사이즈
            - Float32 정규화
            - NCHW 포맷 변환
            """
            # 입력 프레임 검증
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return np.zeros((1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
            
            # 리사이즈 (아스펙트 비율 유지하며 패딩)
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
            
            # NCHW 포맷으로 변환 (Float32로 정규화) -> UINT8로 변경
            # NPU 입력이 UINT8 Quantized를 기대할 수 있음. 하지만 verify script에서는 flat uint8을 썼음.
            # 하지만 여기서 dx_engine의 example을 보면 uint8 raw buffer를 넘김.
            # verify_silu_model.py에서는: input_data = img.flatten().astype(np.uint8)
            # 여기서는 preprocess_frame이 NCHW float를 리턴하고 있음.
            # 이 monitor script는 기존에 float입력을 가정했던 것 같음.
            # 일단 기존 로직을 유지하되 flatten된 uint8을 넘기는게 안전할 수 있음.
            # 하지만 verify script에서 성공한 방식(Uint8 flatten)을 따르는게 좋음.
            
            # img_array = np.asarray(rgb, dtype=np.float32)
            # img_array = img_array.transpose(2, 0, 1) / 255.0
            # img_array = np.expand_dims(img_array, axis=0)
            
            # DXNN verify script style:
            return rgb.flatten().astype(np.uint8)

        
        def infer(self, frame: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
            """
            프레임에서 화재 감지 추론 수행
            """
            if self.engine is None:
                logger.warning("Model not available, skipping inference")
                return []
            
            try:
                # 전처리
                preprocessed = self.preprocess_frame(frame)
                
                # 추론 (DXNN)
                # run() takes list of inputs
                predictions = self.engine.run([preprocessed])
                
                # 3개의 출력을 리스트로 반환 (Output 0, 1, 2)
                return predictions
            
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return []
        
        def compute_nms(self, boxes, scores, iou_threshold=0.45):
            """Simple NMS implementation"""
            if len(boxes) == 0:
                return []
            
            # Convert to x1, y1, x2, y2
            # Boxes are already xyxy
            
            # OpenCV NMS
            # cv2.dnn.NMSBoxes expects (x, y, w, h)
            # boxes are [x1, y1, x2, y2]
            
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            
            # Create list of [x, y, w, h]
            boxes_xywh = []
            for i in range(len(boxes)):
                boxes_xywh.append([int(x[i]), int(y[i]), int(w[i]), int(h[i])])
                
            indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self.conf_threshold, iou_threshold)
            
            if len(indices) == 0:
                return []
                
            return indices.flatten()

        def postprocess(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> Tuple[List[dict], float]:
            """
            Multi-scale Output Decoding + NMS
            Returns: (detections, max_confidence, positive_count)
            """
            if not predictions:
                return [], 0.0, 0
            
            if not isinstance(predictions, list):
                predictions = [predictions]
                
            all_boxes = []
            all_scores = []
            max_conf_global = 0.0
            total_positive_count = 0
            
            # Anchors for YOLOv7 (P3, P4, P5)
            # Strides: 8, 16, 32
            anchors = {
                8:  [[12,16], [19,36], [40,28]],
                16: [[36,75], [76,55], [72,146]],
                32: [[142,110], [192,243], [459,401]]
            }
            
            input_h, input_w = self.input_size
            
            for output in predictions:
                if output.ndim != 5:
                    continue
                    
                # output shape: [1, 3, H, W, 7]
                bs, na, h, w, c = output.shape
                stride = input_h // h
                
                if stride not in anchors:
                    continue
                    
                curr_anchors = np.array(anchors[stride])
                
                # Grid coordinates
                grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                grid = np.stack((grid_x, grid_y), axis=2).reshape(1, 1, h, w, 2)
                
                # Decode
                # 0-1: x, y (sigmoid * 2 - 0.5 + grid) * stride
                # 2-3: w, h ((sigmoid * 2) ** 2 * anchor)
                # 4: obj (sigmoid)
                # 5: cls (sigmoid)
                
                # Apply sigmoid to all
                out_sigmoid = expit(output)

                # Density Count Logic (Pre-filtering)
                # Count anchors with Objectness * Class0 > 0.5
                # Since model saturates to 1.0, this counts 'active' regions
                raw_scores_map = out_sigmoid[..., 4] * out_sigmoid[..., 5]
                total_positive_count += np.sum(raw_scores_map > 0.5)
                
                # Box coordinates
                xy = (out_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
                wh = (out_sigmoid[..., 2:4] * 2.0) ** 2 * curr_anchors.reshape(1, 3, 1, 1, 2)
                
                # Center to TopLeft conversion
                xy_tl = xy - wh / 2.0
                xy_br = xy + wh / 2.0
                
                boxes = np.concatenate((xy_tl, xy_br), axis=-1) # x1, y1, x2, y2
                
                # Confidence
                obj_conf = out_sigmoid[..., 4]
                cls_conf = out_sigmoid[..., 5]
                scores = obj_conf * cls_conf
                
                # Filter low scores
                mask = scores > self.conf_threshold
                
                if np.any(mask):
                    filtered_boxes = boxes[mask]
                    filtered_scores = scores[mask]
                    
                    all_boxes.append(filtered_boxes)
                    all_scores.append(filtered_scores)
                    
                    current_max = np.max(filtered_scores)
                    if current_max > max_conf_global:
                        max_conf_global = current_max

            if not all_boxes:
                return [], max_conf_global, total_positive_count
                
            # Concatenate all scales
            all_boxes = np.concatenate(all_boxes, axis=0)
            all_scores = np.concatenate(all_scores, axis=0)
            
            # NMS
            indices = self.compute_nms(all_boxes, all_scores)
            
            detections = []
            for idx in indices:
                box = all_boxes[idx]
                score = all_scores[idx]
                detections.append({
                    'x1': int(box[0]), 'y1': int(box[1]),
                    'x2': int(box[2]), 'y2': int(box[3]),
                    'score': float(score)
                })
                
            return detections, max_conf_global, total_positive_count

        def extract_max_confidence(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> float:
            # This is now handled inside postprocess, wrapper for compatibility if needed
            _, max_conf = self.postprocess(predictions)
            return max_conf
        
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
            alert_level: str,
            detections: List[dict] = []
        ) -> np.ndarray:
            """프레임에 정보 및 알림 표시"""
            frame_display = frame.copy()
            h, w = frame_display.shape[:2]
            
            # --- Draw Detections ---
            for det in detections:
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                score = det['score']
                
                # Scale coordinates if frame size differs from input size (640x640)
                # But here we assume frame is the original camera frame?
                # The preprocessing resized it.
                # Actually, the boxes are in 640x640 coords.
                # We need to scale them back to frame_display size.
                
                scale_x = w / self.input_size[1]
                scale_y = h / self.input_size[0]
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Color based on score
                color = (0, 0, 255) if score > 0.5 else (0, 255, 255)
                
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_display, f"Fire: {score:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Draw Status Info ---
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
                f"Alert: {alert_msg_short} (Boxes: {len(detections)})",
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
                    
                    # 후처리 (Decoding + NMS)
                    detections, max_conf_global, pos_count = self.postprocess(predictions)
                    
                    # Standard Confidence Logic (No Density Override)
                    effective_conf = max_conf_global

                    # 이력에 저장
                    self.confidence_history.append(effective_conf)
                    self.timestamp_history.append(current_time)
                    
                    # 평균 신뢰도 계산 및 알림 결정
                    avg_confidence = self.get_time_averaged_confidence()
                    alert_level = self.determine_alert_level(avg_confidence)
                    
                    # 알림 로그


                    self.log_alert(avg_confidence, alert_level)
                    
                    # 프레임에 정보 표시 (박스 포함)
                    frame_with_info = self.draw_info_on_frame(
                        frame,
                        effective_conf,
                        avg_confidence,
                        alert_level,
                        detections
                    )
                    
                    # Draw Density Count Debug Info
                    cv2.putText(frame_with_info, f"Density Count: {pos_count}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
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
