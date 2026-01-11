
import cv2
import numpy as np
import time
from collections import deque
from scipy.special import expit

class FireDetector:
    CLASS_NAMES = {0: 'Fire', 1: 'Smoke'}
    
    def __init__(self, config):
        self.config = config
        
        # Config Params
        t_config = self.config.get("thresholds", {}).get("fire", {})
        levels = t_config.get("confidence_levels", {"LOW": 0.35, "MEDIUM": 0.50, "HIGH": 0.65})
        
        # Build Level Ranges dynamically
        self.ALERT_LEVEL = {
            'MONITORING': (0.00, levels['LOW']),
            'LOW': (levels['LOW'], levels['MEDIUM']),
            'MEDIUM': (levels['MEDIUM'], levels['HIGH']),
            'HIGH': (levels['HIGH'], 1.01)
        }
        
        time_window = t_config.get("time_window", 3.0)
        self.history_len = int(15 * time_window) # 15 FPS (Alternating)
        
        # History for Fire(0) and Smoke(1)
        self.conf_history = {
            0: deque(maxlen=self.history_len),
            1: deque(maxlen=self.history_len)
        }
        self.alert_status = {
            0: {'level': 'MONITORING', 'last_time': 0, 'first_low_time': 0},
            1: {'level': 'MONITORING', 'last_time': 0, 'first_low_time': 0}
        }
        
        self.alert_cooldown = 2.0
        self.levels = levels # Usage ref
        self.conf_thresh = self.levels["LOW"]
        self.input_size = (640, 640)
        
    def preprocess_frame(self, frame):
        # Letterbox Padding (Keep Aspect Ratio)
        h, w = frame.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        padded = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        pad_y, pad_x = (self.input_size[1] - new_h) // 2, (self.input_size[0] - new_w) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).tobytes()

    def determine_level(self, avg_conf):
        for lvl, (low, high) in self.ALERT_LEVEL.items():
            if low <= avg_conf < high:
                return lvl
        return 'HIGH'

    def process(self, outputs, shape):
        # Post-process (Returns detections and RAW max confidence per class)
        detections, max_conf_map = self._postprocess(outputs, shape)
        
        status_results = {}
        should_alert_any = False
        now = time.time()
        
        for cls_id in [0, 1]:
            # Update History (Use raw max conf, even if 0.0)
            curr_max = max_conf_map.get(cls_id, 0.0)
            self.conf_history[cls_id].append(curr_max)
            
            # Calculate Average
            hist = self.conf_history[cls_id]
            avg_conf = sum(hist) / len(hist) if hist else 0.0
            
            # Determine Level
            new_level = self.determine_level(avg_conf)
            old_level = self.alert_status[cls_id]['level']
            last_time = self.alert_status[cls_id]['last_time']
            
            # Persistent LOW Alert Logic
            if new_level == 'LOW':
                if self.alert_status[cls_id]['first_low_time'] == 0:
                     self.alert_status[cls_id]['first_low_time'] = now
            else:
                self.alert_status[cls_id]['first_low_time'] = 0 # Reset
            
            # Check Alert Condition
            is_alert = False
            
            # 1. Level Changed (Upward or to Monitoring)
            if new_level != old_level:
                self.alert_status[cls_id]['level'] = new_level
                self.alert_status[cls_id]['last_time'] = now
                if new_level != 'MONITORING': is_alert = True
            
            # 2. Frequent Re-alert (High Level)
            elif new_level != 'MONITORING' and new_level != 'LOW' and (now - last_time > self.alert_cooldown):
                self.alert_status[cls_id]['last_time'] = now
                is_alert = True
                
            # 3. Persistent LOW Alert (> 60s)
            elif new_level == 'LOW':
                first_low = self.alert_status[cls_id]['first_low_time']
                if first_low > 0 and (now - first_low > 60.0) and (now - last_time > self.alert_cooldown):
                    self.alert_status[cls_id]['last_time'] = now
                    is_alert = True 
            
            if is_alert: should_alert_any = True
            
            status_results[cls_id] = {
                'level': new_level,
                'avg_conf': avg_conf,
                'is_alert': is_alert,
                'name': self.CLASS_NAMES[cls_id]
            }
            
        return detections, status_results

    def _postprocess(self, predictions, shape):
        # Robust check for empty predictions (List or Array)
        if predictions is None: return [], {0:0.0, 1:0.0}
        if isinstance(predictions, list) and not predictions: return [], {0:0.0, 1:0.0}
        if isinstance(predictions, np.ndarray) and predictions.size == 0: return [], {0:0.0, 1:0.0}
            
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        all_boxes = []
        all_scores = []
        all_classes = []
        max_conf_map = {0: 0.0, 1: 0.0}
        
        anchors = {
            8:  [[12,16], [19,36], [40,28]],
            16: [[36,75], [76,55], [72,146]],
            32: [[142,110], [192,243], [459,401]]
        }
        
        input_h, input_w = 640, 640
        h_img, w_img = shape[:2]
        
        for output in predictions:
            if output.ndim != 5: continue
            bs, na, h, w, c = output.shape
            stride = input_h // h
            if stride not in anchors: continue
            
            curr_anchors = np.array(anchors[stride])
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack((grid_x, grid_y), axis=2).reshape(1, 1, h, w, 2)
            
            out_sigmoid = expit(output)
            
            xy = (out_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
            wh = (out_sigmoid[..., 2:4] * 2.0) ** 2 * curr_anchors.reshape(1, 3, 1, 1, 2)
            
            xy_tl = xy - wh / 2.0
            xy_br = xy + wh / 2.0
            boxes = np.concatenate((xy_tl, xy_br), axis=-1)
            
            obj_conf = out_sigmoid[..., 4]
            
            # Fire(0) and Smoke(1)
            for cls_id in [0, 1]:
                if 5 + cls_id >= c: break
                cls_conf = out_sigmoid[..., 5 + cls_id]
                scores = obj_conf * cls_conf
                
                # Update Max Conf (Before Thresholding)
                curr_max = np.max(scores)
                if curr_max > max_conf_map.get(cls_id, 0.0):
                    max_conf_map[cls_id] = curr_max
                
                mask = scores > self.conf_thresh
                if np.any(mask):
                    f_boxes = boxes[mask]
                    f_scores = scores[mask]
                    
                    # Scale boxes
                    scale_x = w_img / input_w
                    scale_y = h_img / input_h
                    f_boxes[:, 0] *= scale_x
                    f_boxes[:, 2] *= scale_x
                    f_boxes[:, 1] *= scale_y
                    f_boxes[:, 3] *= scale_y
                    
                    all_boxes.append(f_boxes)
                    all_scores.append(f_scores)
                    all_classes.append(np.full_like(f_scores, cls_id, dtype=np.int32))

        if not all_boxes: return [], max_conf_map
        
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)
        
        # NMS
        detections = []
        unique_classes = np.unique(all_classes)
        for cls_id in unique_classes:
            cls_mask = all_classes == cls_id
            c_boxes = all_boxes[cls_mask]
            c_scores = all_scores[cls_mask]
            
            boxes_xywh = []
            for b in c_boxes:
                boxes_xywh.append([int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])])
            
            indices = cv2.dnn.NMSBoxes(boxes_xywh, c_scores.tolist(), self.conf_thresh, 0.45)
            
            if len(indices) > 0:
                for i in indices.flatten():
                     bbox = c_boxes[i]
                     int_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                     detections.append({
                         'bbox': int_bbox, 
                         'score': c_scores[i],
                         'class_id': int(cls_id),
                         'class_name': self.CLASS_NAMES[int(cls_id)]
                     })
                 
        return detections, max_conf_map
