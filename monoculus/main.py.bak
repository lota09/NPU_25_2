#!/usr/bin/env python3
"""
DeepX NPU Integrated Security System (Modularized)
- Main Entry Point
- Config: External 'config.json'
"""

import os
import sys

# [Add] Handle --nogui before GUI libraries initialize
SHOW_GUI = True
if "--nogui" in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    SHOW_GUI = False

import cv2
import time
import struct
import numpy as np
import json
import threading
import traceback
import argparse
import logging
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Import Custom Modules
from firedetector import FireDetector
from intruderdetector import IntruderDetector
from sleepmonitor import SleepMonitor
from falldetector import FallDetector

# [Add] Robust Presence Detection
try:
    from user_presence import RobustPresenceDetector
except ImportError as e:
    print(f"⚠️ RobustPresenceDetector import failed: {e}")
    RobustPresenceDetector = None
except Exception as e:
    print(f"⚠️ RobustPresenceDetector setup error: {e}")
    RobustPresenceDetector = None

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

# Keypoint Indices (COCO) - Needed for visualization
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

# Web UI Paths
WEB_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "images")
if not os.path.exists(WEB_IMAGE_DIR):
    os.makedirs(WEB_IMAGE_DIR, exist_ok=True)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DataPusher:
    def __init__(self, backend_url="https://localhost:5000"):
        self.url = backend_url
        self.last_push_time = 0
        
    def push(self, data):
        """Asynchronous HTTP push to backend with Error Logging"""
        def _target():
            try:
                # [DEBUG]
                # print(f"📡 Pushing data: {list(data.keys())}") 
                resp = requests.post(f"{self.url}/api/update", json=data, timeout=2.0, verify=False)
            except requests.exceptions.Timeout:
                pass
            except Exception as e:
                # Detailed error for debugging connection issues
                print(f"❌ Push Error ({self.url}): {e}")
        threading.Thread(target=_target, daemon=True).start()

    def save_and_push_image(self, frame, event_type, filename):
        """Save image to web public folder and notify backend"""
        path = os.path.join(WEB_IMAGE_DIR, filename)
        cv2.imwrite(path, frame)
        # image_url is relative to public folder in Vite
        self.push({event_type: {"image_url": f"/images/{filename}", "last_time": datetime.now().strftime('%H:%M:%S')}})

def load_config(path="config.json"):
    global CONFIG
    if not os.path.exists(path):
        print(f"❌ Config file not found: {path}")
        sys.exit(1)
    with open(path, 'r') as f:
        CONFIG = json.load(f)
    print(f"✅ Configuration loaded from {path}")

class SystemManager:
    def __init__(self):
        load_config() # Loads into global CONFIG
        
        # Instantiate Modules with injected config
        self.fall_detector = FallDetector(CONFIG)
        self.sleep_monitor = SleepMonitor(CONFIG)
        self.fire_detector = FireDetector(CONFIG) 
        self.thief_detector = IntruderDetector(CONFIG) # Renamed
        
        self.pusher = DataPusher() # For Web UI
        self.engine = None
        self.engine_fire = None # Separate engine for fire
        self.current_model = None
        self.last_scan = 0
        self.is_home = True
        self.mqtt = None
        self.macs = [m.upper() for m in CONFIG["trusted_devices"]]
        self.known_ips = {} # MAC -> Last Known IP
        self.ha_data = None # Data from Home Assistant
        self.current_temperature = 21.0 # Demo Default Temp

        # Sticky Alert State (for Heartbeat persistence)
        self.sticky_fire = {'detected': False, 'type': 'Fire', 'level': 'MONITORING', 'time': 0, 'conf': 0.0}
        
        # Image Push Throttling
        self.last_img_push = 0
    
        self.executor = ThreadPoolExecutor(max_workers=2) # For parallel inference
        
        # [Add] Robust Presence Detector
        self.presence_detector = None
        if RobustPresenceDetector:
            self.presence_detector = RobustPresenceDetector(
                self.macs, 
                interface=CONFIG["system"]["arp_interface"],
                subnet_prefix=CONFIG["system"].get("subnet_prefix", "192.168.50")
            )
        
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
            data = json.loads(payload)
            self.ha_data = data
            print(f"📩 HA Update: {data}")
            
            # 1. Update Temperature
            if 'temp' in data:
                try:
                    self.current_temperature = float(data['temp'])
                except: pass
                
            # 2. Check for Log Events
            # HA sends: {"msg": "Sleep Mode Started", "status": "SLEEP_ENTRY", ...}
            if 'msg' in data and 'status' in data:
                print(f"📝 Log Event Detected: {data['msg']}") # [DEBUG]
                log_entry = {
                    "log": {
                        "time": data.get('time', datetime.now().strftime('%H:%M:%S')),
                        "message": data['msg'],
                        "type": data['status'] # SLEEP_ENTRY, DEEP_SLEEP, WAKE_UP, etc.
                    }
                }
                self.pusher.push(log_entry)
            else:
                # [DEBUG] Check why it failed if it looked relevant
                if 'msg' in data:
                    print(f"⚠️ Msg found but no status: {list(data.keys())}")
                
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

        # [Modified] Use RobustPresenceDetector if available
        if self.presence_detector:
            self.is_home = self.presence_detector.scan()
            return
        
        # Backward compatibility fallback removed for cleaner code (assuming RobustPresenceDetector works)
        # If fallback is needed, insert here. But RobustPresenceDetector should be robust.
        self.is_home = True # Safety default? No, keep existing state if scan fails.
        return

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
            print(f"✅ Main Model Loaded Successfully!")
            
            # Fire Engine (Only in AWAY) - Optional, non-blocking
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
                        print(f"⚠️ Continuing without Fire Detection...")
                        self.engine_fire = None
                else:
                    print(f"⚠️ Skipping Fire Model: Path invalid or not set.")
                    self.engine_fire = None
            
            self.current_model = mode
            print(f"✅ Model Loading Complete: {mode} mode")
        except Exception as e:
            import traceback
            print(f"❌ CRITICAL ERROR loading model: {e}")
            traceback.print_exc()
            sys.exit(1)

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
        npu_log_done = False
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
            i_conf = 0.0
            ratio = 0.0
            
            if self.engine:
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
                             sleep_status, move_count = self.sleep_monitor.process(user, self.ha_data)
                             self.last_status = sleep_status
                             
                             fall_alert, ftype = self.fall_detector.process(0, user['keypoints'], curr_time)
                             self._draw_overlay(frame, user['bbox'], f"Status: {sleep_status} | Moves: {move_count}", (0,255,0))
                             self._draw_skeleton(frame, user['keypoints'])
                             if fall_alert:
                                 cv2.putText(frame, f"FALL ({ftype})", (50, 200), 1, 2, (0,0,255), 3)
                                 self.pusher.save_and_push_image(frame, "fall", "last_fall.jpg")
                                 self.pusher.push({"fall": {"detected": True, "type": ftype}})
                     else:
                         # [AWAY MODE] Parallel Monitoring (Fire & Intruder)
                         t_submit = time.time()
                         future_fire = self.executor.submit(self._run_fire_task, frame)
                         future_intruder = self.executor.submit(self._run_intruder_task, frame, curr_time)
                         
                         fdets, f_results, t_fire = future_fire.result()
                         i_res = future_intruder.result()
                         intruder_data = i_res[:4]
                         t_intruder = i_res[4]
                         
                         t_wall = time.time() - t_submit
                         
                         if t_wall > 0:
                             sum_time = t_fire + t_intruder
                             ratio = sum_time / t_wall

                         # --- 1. Process Fire Results ---
                         if fdets:
                             for det in fdets:
                                 name = det.get('class_name', 'Fire')
                                 color = (0,0,255) if name == 'Fire' else (0,165,255)
                                 self._draw_overlay(frame, det['bbox'], f"{name} {det['score']:.2f}", color)
                         
                         if f_results:
                                 y_off = 80
                                 for cls_id, res in f_results.items():
                                     name = res['name']
                                     level = res['level']
                                     avg = res['avg_conf']
                                     color = (0,0,255) if name == 'Fire' else (0,165,255)
                                     
                                     cv2.putText(frame, f"{name}: {level} ({avg:.2f})", (10, y_off), 1, 1.5, color, 2)
                                     y_off += 40
                                     
                                     if res['is_alert']:
                                         print(f"🔥 ALERT CONDITION: {name} {level} (Avg: {avg:.2f})")
                                         if self.mqtt:
                                             topic_suffix = f"{name.upper()}_DETECTED_{level}"
                                             self.mqtt.publish(CONFIG["mqtt"]["topic"], topic_suffix)
                                         
                                         if time.time() - self.last_img_push > 1.0:
                                             self.pusher.save_and_push_image(frame, "fire", f"last_{name.lower()}.jpg")
                                             self.last_img_push = time.time()
                                         
                                         self.pusher.push({"fire": {"detected": True, "type": name, "level": level, "conf": float(avg)}})
                                         
                                         self.sticky_fire = {
                                             'detected': True, 
                                             'type': name, 
                                             'level': level, 
                                             'time': time.time(),
                                             'conf': float(avg)
                                         }
                         
                         # --- 2. Process Intruder Results ---
                         if intruder_data:
                              i_dets, i_intruder, i_should, i_conf = intruder_data
                              
                              for d in i_dets:
                                  self._draw_overlay(frame, d['bbox'], f"Person {d['conf']:.2f}", (255,0,0))
                              
                              cv2.putText(frame, f"Intruder: {'ALERT' if i_intruder else 'MONITORING'} ({i_conf:.2f})", 
                                          (10, 60), 1, 1.5, (255,0,0) if i_intruder else (255,255,0), 2)

                              if i_intruder:
                                  intruder_alert = True
                                  if i_should:
                                       print(f"🚨 INTRUDER ALERT CONDITION (Conf: {i_conf:.2f})")
                                       if self.mqtt:
                                            payload = "INTRUDER_DETECTED"
                                            self.mqtt.publish(CONFIG["mqtt"]["topic"], payload)
                                       
                                       if time.time() - self.last_img_push > 1.0:
                                            self.pusher.save_and_push_image(frame, "intruder", "last_intruder.jpg")
                                            self.last_img_push = time.time()
                                            
                                       self.pusher.push({"intruder": {"detected": True, "conf": float(i_conf)}})
                                 
                 except Exception: 
                     print("🚨 RUNTIME ERROR IN LOOP:")
                     traceback.print_exc()

            # [General Status Push - 1s Interval]
            if curr_time - last_mqtt_time > 1.0:
                if self.mqtt:
                    payload = {
                        "mode": target,
                        "status": sleep_status,
                        "moves": move_count,
                        "fall": fall_alert,
                        "intruder": intruder_alert,
                        "ha": self.ha_data 
                    }
                    try:
                        self.mqtt.publish(CONFIG["mqtt"]["topic"], json.dumps(payload))
                    except: pass
                
                web_payload = {
                    "is_home": is_home,
                    "fps": round(fps, 1),
                    "status": "MONITORING" if not (fall_alert or intruder_alert) else "ALERT",
                    "fire": {
                        "detected": self.sticky_fire['detected'] if (curr_time - self.sticky_fire['time'] < 5.0) else False,
                        "type": self.sticky_fire['type'],
                        "level": self.sticky_fire['level'] if (curr_time - self.sticky_fire['time'] < 5.0) else "MONITORING",
                        "conf": self.sticky_fire['conf'] if (curr_time - self.sticky_fire['time'] < 5.0) else 0.0
                    }, 
                    "intruder": {
                        "detected": intruder_alert,
                        "conf": i_conf
                    },
                    "fall": {
                        "detected": fall_alert,
                        "type": ftype if fall_alert else None
                    },
                    "sleep": {
                        "is_monitoring": is_home,
                        "toss_turn_count": move_count,
                        "temperature": self.current_temperature,
                        "posture": sleep_status,
                        "ha_data": self.ha_data
                    },
                    "timestamp": datetime.now().strftime('%H:%M:%S')
                }
                self.pusher.push(web_payload)
                
                last_mqtt_time = curr_time

            # [FPS Calculation & Log]
            fps_cnt += 1
            if curr_time - fps_start > 1.0:
                fps = fps_cnt / (curr_time - fps_start)
                fps_cnt = 0
                fps_start = curr_time
                
                if is_home:
                    print(f"🏠 [HOME] FPS:{fps:.1f} | Status: {sleep_status} | Moves: {move_count}")
                else: 
                    if not npu_log_done and ratio > 0:
                        status = "PARALLEL (OK)" if ratio > 1.1 else "SERIAL (Queueing)"
                        msg = f"🚨 [AWAY] FPS:{fps:.1f} | NPU: {status} | Overlap Ratio: {ratio:.2f}"
                        if ratio <= 1.1: msg += " [Warning: Serial Execution]"
                        print(msg)
                        npu_log_done = True
                    
                    if intruder_alert:
                        print(f"   >>> INTRUDER DETECTED! Confidence: {i_conf:.2f}")

            if is_home and fall_alert:
                 print(f"\n🚨 FALL DETECTED ({ftype})! Sending Alert...")

            cv2.putText(frame, f"MODE: {target}", (10, 30), 1, 1.5, (0,255,0) if is_home else (0,255,255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (640-150, 30), 1, 1.5, (255,255,255), 2)
            if SHOW_GUI:
                cv2.imshow("Security System", frame)
                if cv2.waitKey(1) == ord('q'): break
            
        cap.release()
        if SHOW_GUI:
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

    def _run_fire_task(self, frame):
        t0 = time.time()
        try:
             if not self.engine_fire: return [], {}, 0.0
             fire_buf = self.fire_detector.preprocess_frame(frame)
             fire_input = [np.frombuffer(fire_buf, dtype=np.uint8)]
             outputs = self.engine_fire.run(fire_input)
             res = self.fire_detector.process(outputs, frame.shape)
             return res[0], res[1], (time.time() - t0)
        except:
             traceback.print_exc()
             return [], {}, 0.0

    def _run_intruder_task(self, frame, curr_time):
        t0 = time.time()
        try:
             buf = cv2.resize(frame, (640, 640))
             buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB).tobytes()
             input_data = [np.frombuffer(buf, dtype=np.uint8)]
             
             outputs = self.engine.run(input_data)
             dets = self.parse_yolov8(outputs[0], frame.shape)
             intruder, should_a, conf = self.thief_detector.process(dets, curr_time)
             return (dets, intruder, should_a, conf, time.time() - t0)
        except:
             traceback.print_exc()
             return ([], False, False, 0.0, 0.0)

if __name__ == "__main__":
    SystemManager().run()
