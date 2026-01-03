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
        
        # Class Definitions
        CLASS_NAMES = {
            0: 'Fire',
            1: 'Smoke'
        }
        CLASS_COLORS = {
            0: (0, 0, 255),    # Red for Fire
            1: (200, 200, 200) # Grey for Smoke
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
            
            # 신뢰도 이력 (Class별 분리)
            self.conf_history = {
                0: deque(maxlen=int(self.fps * self.time_window)), # Fire
                1: deque(maxlen=int(self.fps * self.time_window))  # Smoke
            }
            
            # 상태 추적 (Class별 분리)
            self.alert_status = {
                0: {'level': 'MONITORING', 'last_time': 0},
                1: {'level': 'MONITORING', 'last_time': 0}
            }
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
            Multi-scale Output Decoding + NMS (Updated for Multi-Class)
            Returns: (detections, max_conf_per_class, positive_count)
            """
            if not predictions:
                return [], {0: 0.0, 1: 0.0}, 0
            
            if not isinstance(predictions, list):
                predictions = [predictions]
                
            all_boxes = []
            all_scores = []
            all_classes = []
            max_conf_map = {0: 0.0, 1: 0.0}
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
                # 0-1: x, y
                # 2-3: w, h
                # 4: obj
                # 5: Class 0 (Fire)
                # 6: Class 1 (Smoke) - IF AVAILABLE
                
                # Apply sigmoid
                out_sigmoid = expit(output)

                # Box coordinates
                xy = (out_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
                wh = (out_sigmoid[..., 2:4] * 2.0) ** 2 * curr_anchors.reshape(1, 3, 1, 1, 2)
                xy_tl = xy - wh / 2.0
                xy_br = xy + wh / 2.0
                boxes = np.concatenate((xy_tl, xy_br), axis=-1)
                
                # Confidence & Classes
                obj_conf = out_sigmoid[..., 4]
                
                # Iterate over classes (0: Fire, 1: Smoke)
                # Ensure channel count supports it
                num_classes = c - 5
                
                for cls_id in range(num_classes):
                    if cls_id > 1: break # Only care about Fire(0) and Smoke(1)
                    
                    cls_conf = out_sigmoid[..., 5 + cls_id]
                    scores = obj_conf * cls_conf
                    
                    # Update Max Conf (Raw)
                    current_max = np.max(scores)
                    if current_max > max_conf_map.get(cls_id, 0.0):
                        max_conf_map[cls_id] = current_max
                    
                    # Filter
                    mask = scores > self.conf_threshold
                    if np.any(mask):
                        all_boxes.append(boxes[mask])
                        all_scores.append(scores[mask])
                        all_classes.append(np.full_like(scores[mask], cls_id, dtype=np.int32))

            if not all_boxes:
                return [], max_conf_map, total_positive_count
                
            # Concatenate
            all_boxes = np.concatenate(all_boxes, axis=0)
            all_scores = np.concatenate(all_scores, axis=0)
            all_classes = np.concatenate(all_classes, axis=0)
            
            # Simple NMS (Class-agnostic or per-class? Let's do Class-agnostic for safety or per-class?)
            # Usually per-class is better.
            
            detections = []
            
            # Process NMS per class
            unique_classes = np.unique(all_classes)
            for cls_id in unique_classes:
                cls_mask = all_classes == cls_id
                c_boxes = all_boxes[cls_mask]
                c_scores = all_scores[cls_mask]
                
                indices = self.compute_nms(c_boxes, c_scores)
                
                for idx in indices:
                    box = c_boxes[idx]
                    score = c_scores[idx]
                    detections.append({
                        'x1': int(box[0]), 'y1': int(box[1]),
                        'x2': int(box[2]), 'y2': int(box[3]),
                        'score': float(score),
                        'class_id': int(cls_id),
                        'class_name': self.CLASS_NAMES.get(int(cls_id), 'Unknown')
                    })
                
            return detections, max_conf_map, total_positive_count

        def extract_max_confidence(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> float:
            # This is now handled inside postprocess, wrapper for compatibility if needed
            _, max_conf = self.postprocess(predictions)
            return max_conf
        
        def get_time_averaged_confidence(self, cls_id) -> float:
            """시간 기반 평균 신뢰도 계산 (Class별)"""
            hist = self.conf_history.get(cls_id)
            if not hist:
                return 0.0
            return float(np.mean(list(hist)))
        
        def determine_alert_level(self, avg_confidence: float) -> str:
            """평균 신뢰도에 따른 알림 등급 결정"""
            for level, (min_conf, max_conf, msg) in self.ALERT_LEVEL.items():
                if min_conf <= avg_confidence < max_conf:
                    return level
            return 'HIGH'  # >= 0.65
        
        def log_alert(self, avg_conf_map, alert_status_map):
            """알림 로그 출력 (Class별)"""
            current_time = time.time()
            
            for cls_id, status in alert_status_map.items():
                level = status['level']
                last = status['last_time']
                avg = avg_conf_map.get(cls_id, 0.0)
                
                # Check change or timeout
                if (level != self.alert_status[cls_id]['level'] or 
                    current_time - self.alert_status[cls_id]['last_time'] > self.alert_duration):
                    
                    self.alert_status[cls_id]['level'] = level
                    self.alert_status[cls_id]['last_time'] = current_time
                    
                    _, _, msg = self.ALERT_LEVEL[level]
                    name = self.CLASS_NAMES[cls_id]
                    
                    # Only log if relevant (skip boring monitoring logs unless debug)
                    if level != 'MONITORING':
                        logger.info(f"[{name}] {msg} | Conf: {avg:.4f}")
        
        def draw_info_on_frame(
            self,
            frame: np.ndarray,
            max_conf_map: dict,
            avg_conf_map: dict,
            alert_status_map: dict,
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
                
                # Color based on class
                cls_id = det.get('class_id', 0)
                name = det.get('class_name', 'Fire')
                
                base_color = self.CLASS_COLORS.get(cls_id, (0, 255, 0))
                
                # Highlight logic
                color = base_color
                
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_display, f"{name}: {score:.2f}", (x1, y1-10), 
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
            # Remove legacy 'color = ...' line as it is defined inside the loop below
            
            # 정보 표시 Loop
            y_offset = 30
            
            for cls_id in [0, 1]:
                name = self.CLASS_NAMES[cls_id]
                avg = avg_conf_map.get(cls_id, 0.0)
                level = alert_status_map[cls_id]['level']
                _, _, alert_msg = self.ALERT_LEVEL[level]
                
                color = self.CLASS_COLORS[cls_id]
                if level == 'HIGH': color = (0, 0, 255) # High alert is always Red
                
                text = f"[{name}] {level} ({avg:.2f}) {alert_msg.split('|')[0]}"
                
                cv2.putText(frame_display, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
            
            # 알림 등급별 테두리 (Fire Priority)
            fire_level = alert_status_map[0]['level']
            smoke_level = alert_status_map[1]['level']
            
            border_color = (0, 0, 0)
            if fire_level == 'HIGH': border_color = (0, 0, 255)
            elif fire_level == 'MEDIUM': border_color = (0, 165, 255)
            elif smoke_level == 'HIGH': border_color = (200, 200, 200) # Smoke High -> Grey Border
            
            if border_color != (0, 0, 0):
                thickness = 5
                cv2.rectangle(frame_display, (0, 0), (w-1, h-1), border_color, thickness)
            
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
                    detections, max_conf_map, _ = self.postprocess(predictions)
                    
                    # 각 클래스별 통계 업데이트
                    avg_conf_map = {}
                    alert_status_map = {}
                    
                    for cls_id in [0, 1]:
                        # 이력 저장
                        curr = max_conf_map.get(cls_id, 0.0)
                        self.conf_history[cls_id].append(curr)
                        
                        # 평균 계산
                        avg = self.get_time_averaged_confidence(cls_id)
                        avg_conf_map[cls_id] = avg
                        
                        # 레벨 결정
                        lvl = self.determine_alert_level(avg)
                        alert_status_map[cls_id] = {'level': lvl, 'last_time': 0} # time not used here strictly
                    
                    self.log_alert(avg_conf_map, alert_status_map)
                    
                    # 프레임에 정보 표시 (박스 포함)
                    frame_with_info = self.draw_info_on_frame(
                        frame,
                        max_conf_map,
                        avg_conf_map,
                        alert_status_map,
                        detections
                    )
                    
                    # Draw Density Count Debug Info - REMOVED (Legacy)
                    # cv2.putText(frame_with_info, f"Density Count: {pos_count}", (10, 70), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
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
