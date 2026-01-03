#!/usr/bin/env python3
"""
DeepX NPU Integrated Security System (Final)
- Logic: Based on verified 'fall_code.py' & '2_sleep.py'
- Config: External 'config.json'
"""

import cv2
import time
import os
import sys
import struct
import numpy as np
import json
import math
import subprocess
import threading
import traceback # Debug
from collections import deque, Counter
from scipy.special import expit # For Fire Detector

# DeepX SDK Check
try:
    from dx_engine import InferenceEngine, InferenceOption
    DEEPX_AVAILABLE = True
except ImportError:
    print("❌ DeepX SDK not found. NPU disabled.")
    sys.exit(1)

# MQTT Check
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# Global Config
CONFIG = {}

# Keypoint Indices (COCO)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12 
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

FULL_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)
]

def load_config(path="config.json"):
    global CONFIG
    if not os.path.exists(path):
        print(f"❌ Config file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        CONFIG = json.load(f)
    print(f"✅ Configuration loaded from {path}")

# ==========================================
# Logic Modules (Directly from verified code)
# ==========================================

class FallDetector:
    """Original Logic from verified code"""
    def __init__(self):
        self.fall_count = 0
        self.last_fall_time = 0
        self.head_tracker = {}
        self.posture_tracker = {}
        self.fall_cooldown = {}
        
        # Load params
        p = CONFIG["fall_params"]
        self.velocity_threshold = p["velocity_threshold"]
        self.min_fall_dist = p["min_fall_dist"]
        self.time_window = p["time_window"]
        self.max_history = p["max_history"]
        
    def process(self, pid, kpts, t):
        if pid in self.fall_cooldown:
            if t - self.fall_cooldown[pid] < 3.0: return False, None

        head_pos = self._get_head_pos(kpts)
        if not head_pos: return False, None
        head_x, head_y = head_pos
        
        torso_metrics = self._get_torso_metrics(kpts, head_pos)

        # Algorithm A: Vertical Fall
        if self._check_vertical_fall(pid, head_y, t):
            self._register_fall(pid, t, "VERTICAL")
            return True, "VERTICAL"

        # Algorithm B: Posture Fall
        if torso_metrics and self._check_posture_fall(pid, torso_metrics, t):
            self._register_fall(pid, t, "POSTURE")
            return True, "POSTURE"
            
        return False, None

    def _get_head_pos(self, kpts):
        if kpts[NOSE][2] > 0.5: return (kpts[NOSE][0], kpts[NOSE][1])
        eyes = [kpts[i] for i in [L_EYE, R_EYE] if kpts[i][2] > 0.5]
        if eyes: return (sum(p[0] for p in eyes)/len(eyes), sum(p[1] for p in eyes)/len(eyes))
        ears = [kpts[i] for i in [L_EAR, R_EAR] if kpts[i][2] > 0.3]
        if ears: return (sum(p[0] for p in ears)/len(ears), sum(p[1] for p in ears)/len(ears))
        return None

    def _get_torso_metrics(self, kpts, head_pos):
        hips = [kpts[i] for i in [L_HIP, R_HIP] if kpts[i][2] > 0.3]
        if not hips: return None
        hip_x = sum(p[0] for p in hips)/len(hips)
        hip_y = sum(p[1] for p in hips)/len(hips)
        head_hip_dist = np.sqrt((head_pos[0]-hip_x)**2 + (head_pos[1]-hip_y)**2)
        
        shoulders = [kpts[i] for i in [L_SHOULDER, R_SHOULDER] if kpts[i][2] > 0.3]
        if len(shoulders) < 2: return None
        sh_width = np.sqrt((shoulders[0][0]-shoulders[1][0])**2 + (shoulders[0][1]-shoulders[1][1])**2)
        horizontal_extent = sh_width
        
        wrists = [kpts[i][0] for i in [L_WRIST, R_WRIST] if kpts[i][2] > 0.3]
        if len(wrists) == 2:
            horizontal_extent = max(horizontal_extent, abs(wrists[1] - wrists[0]))
            
        aspect_ratio = head_hip_dist / horizontal_extent if horizontal_extent > 0 else 0
        return {"aspect_ratio": aspect_ratio, "horizontal_extent": horizontal_extent}

    def _check_vertical_fall(self, pid, head_y, t):
        if pid not in self.head_tracker: self.head_tracker[pid] = deque(maxlen=self.max_history)
        self.head_tracker[pid].append((t, head_y))
        
        if len(self.head_tracker[pid]) < 8: return False
        recent = [d for d in self.head_tracker[pid] if t - d[0] <= self.time_window]
        if len(recent) < 5: return False
        
        dy = recent[-1][1] - recent[0][1]
        dt = recent[-1][0] - recent[0][0]
        
        if dy <= 0 or dy < self.min_fall_dist: return False
        velocity = dy / dt if dt > 0 else 0
        
        if len(recent) >= 8:
            mid = len(recent) // 2
            early_vel = (recent[mid][1] - recent[0][1]) / (recent[mid][0] - recent[0][0] + 1e-6)
            late_vel = (recent[-1][1] - recent[mid][1]) / (recent[-1][0] - recent[mid][0] + 1e-6)
            if early_vel > 0 and late_vel > 0:
                if late_vel / early_vel < 0.5: return False

        return velocity > self.velocity_threshold

    def _check_posture_fall(self, pid, metrics, t):
        if pid not in self.posture_tracker: self.posture_tracker[pid] = deque(maxlen=self.max_history)
        self.posture_tracker[pid].append({"time": t, **metrics})
        
        if len(self.posture_tracker[pid]) < 5: return False
        recent = [d for d in self.posture_tracker[pid] if t - d["time"] <= self.time_window]
        if len(recent) < 2: return False
        
        first, last = recent[0], recent[-1]
        if first["aspect_ratio"] > 1.5 and last["aspect_ratio"] < 0.65: return True
        if first["horizontal_extent"] > 70:
            extent_increase = last["horizontal_extent"] / first["horizontal_extent"]
            if extent_increase > 2.2 and last["aspect_ratio"] < 0.8: return True
        return False

    def _register_fall(self, pid, t, ftype):
        if t - self.last_fall_time > 2.0:
            self.fall_count += 1
            self.last_fall_time = t
            print(f"⚠️ FALL DETECTED ({ftype})! Total: {self.fall_count}")
        self.fall_cooldown[pid] = t

class SleepMonitor:
    def __init__(self):
        self.history = deque(maxlen=CONFIG["thresholds"]["sleep_buffer"])
        self.center_history = deque(maxlen=5)
        self.moves = 0
        self.prev_center = None
        self.move_thresh = CONFIG["thresholds"]["move"]
        self.conf_thresh = CONFIG["thresholds"]["confidence"]
        
    def process(self, det):
        kpts = det['keypoints']
        bbox = det['bbox']
        
        # 1. Posture
        head_conf = max(kpts[NOSE][2], kpts[L_EYE][2], kpts[R_EYE][2])
        body_conf = min(kpts[L_SHOULDER][2], kpts[R_SHOULDER][2])
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        ar = h / w if w > 0 else 0
        
        if body_conf > self.conf_thresh:
            sh_diff = abs(kpts[L_SHOULDER][1] - kpts[R_SHOULDER][1])
            if head_conf > 0.7:
                status = "SIDE" if sh_diff < 25 else "UPRIGHT"
            else:
                status = "PRONE"
        else:
            if ar > 0.9: status = "SIDE"
            else: status = "UPRIGHT" if head_conf > 0.7 else "PRONE"
        
        self.history.append(status)
        final = Counter(self.history).most_common(1)[0][0]
        
        # 2. Movement
        curr_center = self._get_center(kpts, bbox)
        self.center_history.append(curr_center)
        avg_x = sum(c[0] for c in self.center_history) / len(self.center_history)
        avg_y = sum(c[1] for c in self.center_history) / len(self.center_history)
        current_avg = (avg_x, avg_y)
        
        if self.prev_center:
            dist = math.sqrt((self.prev_center[0]-current_avg[0])**2 + (self.prev_center[1]-current_avg[1])**2)
            if dist > self.move_thresh: self.moves += 1
            
        self.prev_center = current_avg
        return final, self.moves

    def _get_center(self, kpts, bbox):
        if kpts[L_HIP][2] > 0.3 and kpts[R_HIP][2] > 0.3:
            return ((kpts[L_HIP][0]+kpts[R_HIP][0])/2, (kpts[L_HIP][1]+kpts[R_HIP][1])/2)
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)


class ThiefDetector:
    def __init__(self):
        self.last_intruder_alert = 0
        self.cool = CONFIG.get("thresholds", {}).get("alert_cooldown", 5)
        
        # Config Params
        t_conf = CONFIG.get("thresholds", {})
        self.conf_thresh = t_conf.get("intruder", {}).get("confidence", 0.75)
        self.time_window = t_conf.get("intruder", {}).get("time_window", 1.0)
        
        # History (Assuming ~15 FPS in alternating mode)
        self.history_len = int(15 * self.time_window)
        self.conf_history = deque(maxlen=self.history_len)

    def process(self, detections, current_time):
        """
        침입자 감지 (시간 평균 신뢰도 적용)
        """
        # 1. Get Max Confidence in current frame
        max_conf = 0.0
        if detections:
            max_conf = max(d['conf'] for d in detections)
            
        # 2. Update History
        self.conf_history.append(max_conf)
        
        # 3. Calculate Average
        if not self.conf_history: return False, False
        avg_conf = sum(self.conf_history) / len(self.conf_history)
        
        is_intruder = False
        should_alert = False

        if avg_conf >= self.conf_thresh:
            is_intruder = True
            if current_time - self.last_intruder_alert > self.cool:
                should_alert = True
                self.last_intruder_alert = current_time
            
        return is_intruder, should_alert, avg_conf
            
        return is_intruder, should_alert

# ==========================================
# Fire Detector Module
# ==========================================
# ==========================================
# Fire Detector Module (Updated for Fire & Smoke)
# ==========================================
class FireDetector:
    CLASS_NAMES = {0: 'Fire', 1: 'Smoke'}
    ALERT_LEVEL = {
        'MONITORING': (0.00, 0.35),
        'LOW': (0.35, 0.50),
        'MEDIUM': (0.50, 0.65),
        'HIGH': (0.65, 1.01)
    }

    def __init__(self):
        # Config Params
        t_config = CONFIG.get("thresholds", {}).get("fire", {})
        # Note: thresholds in config might need update if we want separate controls, 
        # but for now we follow the hardcoded levels in fire_detection_monitor.py or mix them.
        # User requested consistency with fire_detection_monitor.py
        
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
        self.levels = {"LOW": 0.35, "MEDIUM": 0.50, "HIGH": 0.65} # For compatibility with config if needed
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


# ==========================================
# System Manager (Main)
# ==========================================
class SystemManager:
    def __init__(self):
        load_config()
        self.fall_detector = FallDetector()
        self.sleep_monitor = SleepMonitor()
        self.fire_detector = FireDetector() # Fire module
        self.engine = None
        self.engine_fire = None # Separate engine for fire
        self.current_model = None
        self.last_scan = 0
        self.is_home = True
        self.mqtt = None
        self.thief_detector = ThiefDetector()
        self.macs = [m.upper() for m in CONFIG["trusted_devices"]]
        self.known_ips = {} # MAC -> Last Known IP
        self.last_home_time = time.time() # Grace period logic
        self.ha_data = None # Data from Home Assistant
        
        if MQTT_AVAILABLE:
            try:
                # MQTT V2 API 사용 (DeprecationWarning 해결)
                if hasattr(mqtt, "CallbackAPIVersion"):
                     self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                else:
                     self.mqtt = mqtt.Client() # 구버전 호환
                
                self.mqtt.on_message = self._on_mqtt_message
                self.mqtt.connect(CONFIG["mqtt"]["host"], CONFIG["mqtt"]["port"])
                self.mqtt.loop_start()
                
                ha_topic = CONFIG["mqtt"].get("ha_topic", "home/ha_status")
                self.mqtt.subscribe(ha_topic)
                print(f"📡 MQTT Connected & Subscribed to {ha_topic}")
            except: pass

        # Start Background Presence Scan
        self.scan_thread = threading.Thread(target=self._presence_loop, daemon=True)
        self.scan_thread.start()

    def _on_mqtt_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            self.ha_data = json.loads(payload)
            print(f"📩 HA Update: {self.ha_data}")
        except: pass

    def check_presence(self):
        # Main Loop에서는 저장된 상태만 즉시 반환 (Non-blocking)
        return self.is_home

    def _presence_loop(self):
        """백그라운드에서 주기적으로 존재 감지 수행"""
        print("🕵️ Presence Scan Thread Started")
        while True:
            try:
                self._perform_scan()
            except Exception as e:
                print(f"Error in scan loop: {e}")
            
            # 다음 스캔까지 대기
            time.sleep(CONFIG["system"]["arp_interval"])

    def _perform_scan(self):
        mode = CONFIG["system"].get("force_mode", "AUTO")
        if mode != "AUTO": 
            self.is_home = (mode == "HOME")
            return

        try:
            iface = CONFIG["system"]["arp_interface"]
            cmd = ["sudo", "arp-scan", "-l", f"--interface={iface}"]
            if not os.path.exists("/usr/bin/arp-scan"): cmd = ["arp", "-a"]
            
            # [ARP Cache Refresh] arp-scan이 없을 때
            if cmd[0] == "arp":
                try:
                    subprocess.run(["ping", "-c", "1", "-b", "192.168.219.255"], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.2)
                except: pass

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
            found = False
            active_devs = []
            
            # 1. ARP Cache Parsing
            current_ips = {} 
            for line in res.stdout.splitlines():
                upper_line = line.upper()
                for mac in self.macs:
                    if mac in upper_line:
                        parts = line.split()
                        ip = None
                        if '(' in line and ')' in line: # Linux arp format
                            start = line.find('(') + 1
                            end = line.find(')')
                            ip = line[start:end]
                        elif len(parts) >= 2 and parts[0] != '?': # Windows/Other
                            ip = parts[0]
                        if ip:
                            current_ips[mac] = ip
                            self.known_ips[mac] = ip 

            # 2. Active Ping Check
            for mac in self.macs:
                target_ip = current_ips.get(mac)
                if not target_ip: target_ip = self.known_ips.get(mac)
                
                if target_ip:
                    # Ping (타임아웃 1초 -> 이것 때문에 메인 루프가 끊김)
                    param = '-n' if sys.platform.lower()=='windows' else '-c'
                    ping_cmd = ["ping", param, "1", target_ip]
                    if sys.platform.lower() != 'windows': ping_cmd.extend(["-W", "1"])
                    
                    try:
                        pr = subprocess.run(ping_cmd, capture_output=True, text=True, timeout=1)
                        if pr.returncode == 0:
                            found = True
                            active_devs.append(f"{mac}({target_ip})")
                    except: pass

            if found:
                self.last_home_time = time.time()
                if not self.is_home:
                     print(f"📡 MODE CHANGE: HOME (Active: {active_devs})")
                     self.is_home = True
            else:
                # [Simple Logic] Timeout check (Disabled for testing)
                # timeout = CONFIG["system"].get("away_timeout", 300)
                # elapsed = time.time() - self.last_home_time
                # if elapsed < timeout:
                #    print(f"⏳ Grace Period: {int(elapsed)}/{timeout}s (Waiting for signal...)")
                
                # if elapsed >= timeout:
                if True: # Immediate mode
                    if self.is_home:
                        print(f"📡 MODE CHANGE: AWAY (Immediate)")
                        self.is_home = False
        except: pass

    def load_model(self, mode):
        if self.current_model == mode: return
        target = CONFIG["models"]["home_pose"] if mode == 'HOME' else CONFIG["models"]["away_detect"]
        fire_target = CONFIG["models"].get("fire_detect")
        
        # [DEBUG] Check Config
        if mode == 'AWAY':
            print(f"🔍 DEBUG: Loading AWAY Mode. Fire Target from Config: '{fire_target}'")
            if fire_target:
                print(f"🔍 DEBUG: File Exists? {os.path.exists(fire_target)}")
            else:
                print("❌ DEBUG: 'fire_detect' key missing in CONFIG['models']")
        
        if not os.path.exists(target):
            print(f"❌ Model missing: {target}"); return

        # Unload previous
        if self.engine: del self.engine; self.engine = None
        if self.engine_fire: del self.engine_fire; self.engine_fire = None
        time.sleep(0.5)
        
        try:
            # Main Engine
            print(f"⏳ Loading Main Model: {target}...")
            self.engine = InferenceEngine(target, InferenceOption())
            
            # Fire Engine (Only in AWAY)
            if mode == 'AWAY':
                if fire_target and os.path.exists(fire_target):
                    try:
                        print(f"⏳ Attempting to load Fire Model: {fire_target}...")
                        self.engine_fire = InferenceEngine(fire_target, InferenceOption())
                        print(f"🔥 Fire Model Loaded Successfully!")
                    except Exception as e:
                        import traceback
                        print(f"⚠️ Failed to load Fire Model: {e}")
                        traceback.print_exc()
                else:
                    print(f"⚠️ Skipping Fire Model: Path invalid or not set.")
            
            self.current_model = mode
            print(f"✅ Loaded: {mode} (Main)")
        except: sys.exit(1)

    def parse_pose(self, output, shape):
        dets = []
        try:
            if output.size % 256 == 0: output = output.reshape(-1, 256)
            else: return []
            img_h, img_w = shape[:2]
            conf_t = CONFIG["thresholds"]["confidence"]
            
            for i in range(output.shape[0]):
                data = output[i, :].tobytes()
                box = np.frombuffer(data[0:16], dtype=np.float32)
                grid_y, grid_x, _, l_idx = struct.unpack('4B', data[16:20])
                conf = np.frombuffer(data[20:24], dtype=np.float32)[0]
                
                if conf < conf_t: continue
                stride = [8, 16, 32][l_idx] if l_idx < 3 else 8
                sx, sy = img_w / 640.0, img_h / 640.0
                
                kpts_raw = np.frombuffer(data[28:232], dtype=np.float32)
                kpts = []
                vx, vy = [], []
                for k in range(17):
                    mx = (kpts_raw[k*3] * 2.0 - 0.5 + grid_x) * stride
                    my = (kpts_raw[k*3+1] * 2.0 - 0.5 + grid_y) * stride
                    c = 1.0 / (1.0 + np.exp(-kpts_raw[k*3+2]))
                    kpts.append((mx * sx, my * sy, c))
                    if c > 0.3: vx.append(mx*sx); vy.append(my*sy)
                
                if vx and vy:
                    x1, y1 = int(min(vx)-20), int(min(vy)-20)
                    x2, y2 = int(max(vx)+20), int(max(vy)+20)
                    dets.append({'bbox':(x1,y1,x2,y2), 'keypoints':kpts, 'conf':conf, 'area':(x2-x1)*(y2-y1)})
        except: pass
        return dets

    def parse_yolov8(self, output, shape):
        dets = []
        try:
            output = output.reshape(1, 84, 8400)
            output = np.transpose(output, (0, 2, 1))
            data = output[0]
            img_h, img_w = shape[:2]
            xf, yf = img_w / 640.0, img_h / 640.0
            ct = CONFIG["thresholds"]["confidence"]
            boxes, confs = [], []
            
            for row in data:
                score = np.max(row[4:])
                if score >= ct and np.argmax(row[4:]) == 0:
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    boxes.append([int((cx-w/2)*xf), int((cy-h/2)*yf), int(w*xf), int(h*yf)])
                    confs.append(float(score))
            
            indices = cv2.dnn.NMSBoxes(boxes, confs, ct, CONFIG["thresholds"]["iou"])
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    dets.append({'bbox': (x, y, x+w, y+h), 'conf': confs[i]})
        except: pass
        return dets

    def run(self):
        print("📷 Camera Init...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(3, 640); cap.set(4, 480)
        print("🚀 System Started")
        
        last_mqtt_time = 0
        fps_start = time.time()
        fps_cnt = 0
        fps = 0
        self.last_status = "UPRIGHT" # 초기 상태
        
        while True:
            is_home = self.check_presence()
            target = 'HOME' if is_home else 'AWAY'
            self.load_model(target)
            
            ret, frame = cap.read()
            if not ret: time.sleep(1); continue
            
            curr_time = time.time()
            sleep_status = "N/A"
            move_count = 0
            fall_alert = False
            intruder_alert = False
            if self.engine:
                 # Standard Resize (For Intruder/Pose) -- will handle inside FireDetector separately
                 buf = cv2.resize(frame, (640, 640))
                 buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB).tobytes()
                 input_data = [np.frombuffer(buf, dtype=np.uint8)]
                 
                 try:
                     if is_home:
                         # [HOME MODE] Pose Estimation
                         outputs = self.engine.run(input_data)
                         res = self.parse_pose(outputs[0], frame.shape)
                         res.sort(key=lambda x: x['area'], reverse=True)
                         if res:
                             user = res[0]
                             sleep_status, move_count = self.sleep_monitor.process(user)
                             self.last_status = sleep_status
                             
                             fall_alert, ftype = self.fall_detector.process(0, user['keypoints'], curr_time)
                             self._draw_overlay(frame, user['bbox'], f"Status: {sleep_status} | Moves: {move_count}", (0,255,0))
                             self._draw_skeleton(frame, user['keypoints'])
                             if fall_alert:
                                 cv2.putText(frame, f"FALL ({ftype})", (50, 200), 1, 2, (0,0,255), 3)
                     else:
                         # [AWAY MODE] Dual Monitoring (Alternating)
                         
                         # 1. Fire Detection (Odd Frames) -> NOW PRIMARY for Debug
                         # 1. Fire Detection (Odd Frames)
                         if self.engine_fire and (fps_cnt % 2 != 0):
                             # Preprocess for Fire (Letterbox)
                             fire_buf = self.fire_detector.preprocess_frame(frame)
                             fire_input = [np.frombuffer(fire_buf, dtype=np.uint8)]
                             
                             outputs = self.engine_fire.run(fire_input)
                             # PASS FULL OUTPUTS LIST, NOT outputs[0]
                             fdets, f_results = self.fire_detector.process(outputs, frame.shape)
                             
                             # Draw Fire/Smoke Detections
                             for det in fdets:
                                 name = det.get('class_name', 'Fire')
                                 color = (0,0,255) if name == 'Fire' else (200,200,200)
                                 self._draw_overlay(frame, det['bbox'], f"{name} {det['score']:.2f}", color)
                             
                             # Draw Status (Fire & Smoke)
                             y_off = 80
                             for cls_id, res in f_results.items():
                                 name = res['name']
                                 level = res['level']
                                 avg = res['avg_conf']
                                 color = (0,0,255) if name == 'Fire' else (200,200,200)
                                 
                                 cv2.putText(frame, f"{name}: {level} ({avg:.2f})", (10, y_off), 1, 1.5, color, 2)
                                 y_off += 40
                                 
                                 # MQTT Alert
                                 if res['is_alert'] and self.mqtt:
                                     topic_suffix = f"{name.upper()}_DETECTED_{level}"
                                     self.mqtt.publish(CONFIG["mqtt"]["topic"], topic_suffix)
                                     print(f"🔥 SENT {name.upper()} ALERT: {level} (Avg: {avg:.2f})")

                         # 2. Intruder Detection (Even Frames)
                         else: # if not fire or even frame
                             outputs = self.engine.run(input_data)
                             dets = self.parse_yolov8(outputs[0], frame.shape)
                             
                             intruder, should_a, conf = self.thief_detector.process(dets, curr_time)
                             
                             for d in dets:
                                 self._draw_overlay(frame, d['bbox'], f"Person {d['conf']:.2f}", (0,0,255))
                             
                             if intruder:
                                 intruder_alert = True # Set for payload
                                 self._draw_overlay(frame, (10,10,200,60), f"INTRUDER! ({conf:.2f})", (0,0,255))
                                 if should_a and self.mqtt:
                                      self.mqtt.publish(CONFIG["mqtt"]["topic"], "INTRUDER_DETECTED")
                                      print(f"🚨 SENT INTRUDER ALERT (Conf: {conf:.2f})")
                             else:
                                 cv2.putText(frame, f"Monitoring... (Intruder AVG: {conf:.2f})", (10, 50), 1, 1, (255,255,0), 2)
                 except Exception: 
                     print("🚨 RUNTIME ERROR IN LOOP:")
                     traceback.print_exc()
                     pass

            # [복구] 1초 주기 상태 전송
            if self.mqtt and (curr_time - last_mqtt_time > 1.0):
                payload = {
                    "mode": target,
                    "status": sleep_status,
                    "moves": move_count,
                    "fall": fall_alert,
                    "intruder": intruder_alert,
                    "ha": self.ha_data # Forward HA data to App
                }
                self.mqtt.publish(CONFIG["mqtt"]["topic"], json.dumps(payload))
                last_mqtt_time = curr_time

            # [FPS Calculation & Log]
            fps_cnt += 1
            if curr_time - fps_start > 1.0:
                fps = fps_cnt / (curr_time - fps_start)
                fps_cnt = 0
                fps_start = curr_time
                
                # 1초마다 상태 출력
                if is_home:
                    print(f"🏠 [HOME] FPS:{fps:.1f} | Status: {sleep_status} | Moves: {move_count}")
                elif intruder_alert and res:
                    print(f"🚨 [AWAY] INTRUDER DETECTED! Confidence: {res[0]['conf']:.2f}")

            # [Alert Log] - 즉시 출력 (Fall은 비상상황이므로 즉시)
            if is_home and fall_alert:
                 print(f"\n🚨 FALL DETECTED ({ftype})! Sending Alert...")

            cv2.putText(frame, f"MODE: {target}", (10, 30), 1, 1.5, (0,255,0) if is_home else (0,255,255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (640-150, 30), 1, 1.5, (255,255,255), 2)
            cv2.imshow("Security System", frame)
            if cv2.waitKey(1) == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()

    def _draw_overlay(self, frame, b, t, c):
        cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), c, 2)
        cv2.putText(frame, t, (b[0], b[1]-10), 1, 0.7, c, 2)

    def _draw_skeleton(self, frame, kpts):
        for p in FULL_SKELETON:
            k1, k2 = kpts[p[0]], kpts[p[1]]
            if k1[2]>0.3 and k2[2]>0.3:
                cv2.line(frame, (int(k1[0]),int(k1[1])), (int(k2[0]),int(k2[1])), (0,255,0), 2)
        for k in kpts:
            if k[2]>0.3: cv2.circle(frame, (int(k[0]),int(k[1])), 4, (0,0,255), -1)

if __name__ == "__main__":
    SystemManager().run()