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
from collections import deque, Counter

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

# ==========================================
# System Manager (Main)
# ==========================================
class SystemManager:
    def __init__(self):
        load_config()
        self.fall_detector = FallDetector()
        self.sleep_monitor = SleepMonitor()
        self.engine = None
        self.current_model = None
        self.last_scan = 0
        self.is_home = True
        self.mqtt = None
        self.last_intruder_alert = 0
        self.macs = [m.upper() for m in CONFIG["trusted_devices"]]
        
        if MQTT_AVAILABLE:
            try:
                self.mqtt = mqtt.Client()
                self.mqtt.connect(CONFIG["mqtt"]["host"], CONFIG["mqtt"]["port"])
                self.mqtt.loop_start()
                print("📡 MQTT Connected")
            except: pass

    def check_presence(self):
        mode = CONFIG["system"].get("force_mode", "AUTO")
        if mode != "AUTO": return (mode == "HOME")
        
        interval = CONFIG["system"]["arp_interval"]
        if time.time() - self.last_scan < interval:
            return self.is_home
            
        try:
            iface = CONFIG["system"]["arp_interface"]
            cmd = ["sudo", "arp-scan", "-l", f"--interface={iface}"]
            if not os.path.exists("/usr/bin/arp-scan"): cmd = ["arp", "-a"]
            
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            found = any(m in res.stdout.upper() for m in self.macs)
            
            if self.is_home != found:
                print(f"📡 MODE: {'HOME' if found else 'AWAY'}")
            self.is_home = found
        except: pass
        self.last_scan = time.time()
        return self.is_home

    def load_model(self, mode):
        if self.current_model == mode: return
        target = CONFIG["models"]["home_pose"] if mode == 'HOME' else CONFIG["models"]["away_detect"]
        if not os.path.exists(target):
            print(f"❌ Model missing: {target}"); return

        if self.engine:
            del self.engine
            self.engine = None
            time.sleep(0.5)
        try:
            self.engine = InferenceEngine(target, InferenceOption())
            self.current_model = mode
            print(f"✅ Loaded: {mode}")
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
                buf = cv2.resize(frame, (640, 640))
                buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB).tobytes()
                try:
                    outputs = self.engine.run([np.frombuffer(buf, dtype=np.uint8)])
                    if is_home:
                        res = self.parse_pose(outputs[0], frame.shape)
                        res.sort(key=lambda x: x['area'], reverse=True)
                        if res:
                            user = res[0]
                            sleep_status, move_count = self.sleep_monitor.process(user)
                            fall_alert, ftype = self.fall_detector.process(0, user['keypoints'], curr_time)
                            self._draw_overlay(frame, user['bbox'], f"Status: {sleep_status} | Moves: {move_count}", (0,255,0))
                            self._draw_skeleton(frame, user['keypoints'])
                            if fall_alert:
                                cv2.putText(frame, f"FALL ({ftype})", (50, 200), 1, 2, (0,0,255), 3)
                    else:
                        res = self.parse_yolov8(outputs[0], frame.shape)
                        if res:
                            intruder_alert = True
                            for det in res:
                                self._draw_overlay(frame, det['bbox'], f"INTRUDER {det['conf']:.2f}", (0,0,255))
                            cv2.putText(frame, "!!! INTRUDER !!!", (50, 200), 1, 3, (0,0,255), 3)
                            
                            # [추가] 침입자 즉시 전송
                            cool = CONFIG["thresholds"].get("alert_cooldown", 5)
                            if self.mqtt and (curr_time - self.last_intruder_alert > cool):
                                self.mqtt.publish(CONFIG["mqtt"]["topic"], "INTRUDER_DETECTED")
                                print("🚨 SENT INTRUDER ALERT")
                                self.last_intruder_alert = curr_time
                        else:
                            cv2.putText(frame, "Monitoring...", (10, 50), 1, 1, (255,255,0), 2)
                except Exception: pass

            # [복구] 1초 주기 상태 전송
            if self.mqtt and (curr_time - last_mqtt_time > 1.0):
                payload = {
                    "mode": target,
                    "status": sleep_status,
                    "moves": move_count,
                    "fall": fall_alert,
                    "intruder": intruder_alert
                }
                self.mqtt.publish(CONFIG["mqtt"]["topic"], json.dumps(payload))
                last_mqtt_time = curr_time

            cv2.putText(frame, f"MODE: {target}", (10, 30), 1, 1.5, (0,255,0) if is_home else (0,255,255), 2)
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