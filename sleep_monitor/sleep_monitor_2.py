import cv2
import dx_engine as dx
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
import struct
import math
import os
from collections import deque, Counter # ★ Counter 추가 (다수결 투표용)

# ==========================================
# 1. 설정 (Settings)
# ==========================================
MODEL_PATH = "/home/orangepi/deepx_sdk/dx_app/assets/models/YOLOV5Pose640_1.dxnn"

MQTT_BROKER_HOST = "192.168.219.107"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "sleep_monitor/user/test01"

LAYER_CONFIG = [
    {"stride": 8,  "anchor_width": [19.0, 44.0, 38.0], "anchor_height": [27.0, 40.0, 94.0]},
    {"stride": 16, "anchor_width": [72.0, 103.0, 187.0], "anchor_height": [92.0, 198.0, 141.0]},
    {"stride": 32, "anchor_width": [156.0, 237.0, 373.0], "anchor_height": [287.0, 397.0, 525.0]}
]

# ★★★ 임계값 설정 ★★★
WEAK_THRESHOLD = 0.05       # 파싱용
CONF_THRESHOLD = 0.30       # 어깨/몸통용
FACE_THRESHOLD = 0.70       # 얼굴 판별 기준
MIN_BOX_AREA = 5000         # 노이즈 필터

MOVEMENT_THRESHOLD = 10      # 민감도: 6픽셀
CENTER_BUFFER_SIZE = 5      # 움직임 반응속도

# ★ [NEW] 상태 안정화 버퍼 (30프레임 = 약 1초)
# 이 값을 늘리면 더 안정적이지만 반응이 느려지고, 줄이면 빠르지만 불안정함
STATUS_BUFFER_SIZE = 30     

NOSE, L_EYE, R_EYE = 0, 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12 

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

center_history = deque(maxlen=CENTER_BUFFER_SIZE)
status_history = deque(maxlen=STATUS_BUFFER_SIZE) # 상태 기록용 큐

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def on_connect(client, userdata, flags, rc):
    if rc == 0: 
        print(f"MQTT 브로커에 연결되었습니다 (Host: {MQTT_BROKER_HOST})")
    else: 
        print(f"❌ MQTT 연결 실패 (Code: {rc})")

def open_camera_robust():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    for i in range(1, 5):
        if cap.isOpened(): break
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    return cap if cap.isOpened() else None

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def parse_yolov5_with_anchors(output, original_shape):
    detections = []
    img_h, img_w = original_shape[:2]
    try:
        if not isinstance(output, np.ndarray): return detections
        if output.dtype == np.uint8 and output.ndim == 3:
            _, num_dets, data_size = output.shape
            if data_size == 256: 
                for det_idx in range(num_dets):
                    det_bytes = output[0, det_idx, :].tobytes()
                    box_raw = np.frombuffer(det_bytes[0:16], dtype=np.float32)
                    grid_y, grid_x, anchor_idx, layer_idx = struct.unpack('4B', det_bytes[16:20])
                    conf = np.frombuffer(det_bytes[20:24], dtype=np.float32)[0]
                    
                    if conf < WEAK_THRESHOLD: continue
                    if layer_idx >= len(LAYER_CONFIG): continue
                    
                    cfg = LAYER_CONFIG[layer_idx]
                    stride = cfg["stride"]
                    aw, ah = cfg["anchor_width"][anchor_idx], cfg["anchor_height"][anchor_idx]
                    
                    xc = (grid_x - 0.5 + (box_raw[0] * 2)) * stride
                    yc = (grid_y - 0.5 + (box_raw[1] * 2)) * stride
                    w_model = (box_raw[2] ** 2) * 4 * aw
                    h_model = (box_raw[3] ** 2) * 4 * ah
                    
                    scale_x, scale_y = img_w / 640.0, img_h / 640.0
                    x1 = clamp(int((xc - w_model/2) * scale_x), 0, img_w)
                    y1 = clamp(int((yc - h_model/2) * scale_y), 0, img_h)
                    x2 = clamp(int((xc + w_model/2) * scale_x), 0, img_w)
                    y2 = clamp(int((yc + h_model/2) * scale_y), 0, img_h)
                    
                    kpts_floats = np.frombuffer(det_bytes[28:232], dtype=np.float32)
                    keypoints = []
                    for i in range(17):
                        kp_x = ((kpts_floats[i*3] * 2.0 - 0.5 + grid_x) * stride) * scale_x
                        kp_y = ((kpts_floats[i*3+1] * 2.0 - 0.5 + grid_y) * stride) * scale_y
                        kp_conf = 1.0 / (1.0 + np.exp(-kpts_floats[i*3+2]))
                        keypoints.append((kp_x, kp_y, kp_conf))
                    
                    detections.append({'bbox': (x1, y1, x2, y2), 'area': (x2-x1)*(y2-y1), 'keypoints': keypoints, 'confidence': float(conf)})
    except Exception: pass
    return detections

def get_body_center(detection):
    kpts = detection['keypoints']
    bbox = detection['bbox']
    hips = [kpts[i] for i in [L_HIP, R_HIP] if kpts[i][2] > CONF_THRESHOLD]
    if hips: return (sum(h[0] for h in hips)/len(hips), sum(h[1] for h in hips)/len(hips))
    shoulders = [kpts[i] for i in [L_SHOULDER, R_SHOULDER] if kpts[i][2] > CONF_THRESHOLD]
    if shoulders: return (sum(s[0] for s in shoulders)/len(shoulders), sum(s[1] for s in shoulders)/len(shoulders))
    return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

def calculate_distance(p1, p2):
    if not p1 or not p2: return 0
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ==========================================
# 3. ★ 핵심: 통합 판별 로직 + 안정화 ★
# ==========================================

def get_max_head_conf(kpts):
    scores = [kpts[NOSE][2], kpts[L_EYE][2], kpts[R_EYE][2], kpts[L_EAR][2], kpts[R_EAR][2]]
    return max(scores)

def determine_posture_instant(detection):
    """
    순간적인 자세 판별 (1프레임용)
    """
    kpts = detection['keypoints']
    bbox = detection['bbox']
    
    head_conf = get_max_head_conf(kpts)  
    l_shoulder_conf = kpts[L_SHOULDER][2]
    r_shoulder_conf = kpts[R_SHOULDER][2]
    
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]
    aspect_ratio = box_h / box_w if box_w > 0 else 0 
    
    has_strong_face = (head_conf > FACE_THRESHOLD)
    has_strong_body = (l_shoulder_conf > CONF_THRESHOLD and r_shoulder_conf > CONF_THRESHOLD)

    # [Case A] 이불 안 덮음 (몸이 선명) -> 정밀 로직
    if has_strong_body:
        y_diff = abs(kpts[L_SHOULDER][1] - kpts[R_SHOULDER][1])
        if has_strong_face:
            if y_diff < 25: return "SIDE" # 반전 로직
            else: return "UPRIGHT"
        else:
            # 몸은 보이는데 얼굴이 안 보임 -> 엎드림
            return "PRONE"

    # [Case B] 이불 속 (몸이 희미) -> 뭉뚱그려 판단
    else:
        # 1. 어깨가 불안정하므로 절대 어깨 높이를 믿지 않음
        # 2. 대신 박스 비율과 얼굴 유무로만 판단
        
        # 박스가 좁고 뚱뚱함 (비율 > 0.9) -> 옆으로 웅크려 잠
        if aspect_ratio > 0.9: 
            return "SIDE"
            
        # 박스가 납작함 (일반적 누움)
        if has_strong_face:
            return "UPRIGHT"
        else:
            return "PRONE"

def get_stabilized_status(current_status):
    """
    [NEW] 상태 안정화 함수 (투표 시스템)
    최근 N개의 상태 중 가장 많이 나온 상태를 반환
    """
    status_history.append(current_status)
    
    # 데이터가 아직 덜 모였으면 현재 상태 리턴
    if len(status_history) < 5:
        return current_status
        
    # 최빈값(가장 많이 나온 상태) 찾기
    counter = Counter(status_history)
    most_common_status = counter.most_common(1)[0][0]
    return most_common_status

# ==========================================
# 4. 메인 실행
# ==========================================
def main():
    print("🚀 수면 모니터링 (Stabilized Version)")
    
    client = mqtt.Client()
    client.on_connect = on_connect
    try: client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60); client.loop_start()
    except: pass

    if not os.path.exists(MODEL_PATH): print("❌ 모델 없음"); return
    ie = dx.InferenceEngine(MODEL_PATH)
    cap = open_camera_robust()
    if not cap: return
    
    cap.set(3, 640); cap.set(4, 480); cap.set(5, 30)
    
    prev_avg_center = None
    movement_counter = 0
    last_mqtt_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            input_tensor = cv2.resize(frame, (640, 640))
            input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
            input_bytes = np.array(input_tensor, dtype=np.uint8).tobytes()
            outputs = ie.run([np.frombuffer(input_bytes, dtype=np.uint8)])
            
            detections = parse_yolov5_with_anchors(outputs[0], frame.shape)
            # 노이즈 필터 (먼지 제거)
            valid_detections = [d for d in detections if d['area'] > MIN_BOX_AREA]
            
            # 기본값
            raw_status = "BED_EXIT"
            final_status = "BED_EXIT"
            status_color = (0, 0, 255)
            
            if valid_detections:
                valid_detections.sort(key=lambda x: x['area'], reverse=True)
                target = valid_detections[0]
                
                # 1. 순간 판별
                raw_status = determine_posture_instant(target)
                
                # 2. ★ 상태 안정화 (Voting) ★
                # 순간적으로 튀는 값(노이즈)을 걸러내고 다수결로 결정
                final_status = get_stabilized_status(raw_status)
                status_color = (0, 255, 0)
                
                # --- 움직임 감지 ---
                curr_raw_center = get_body_center(target)
                center_history.append(curr_raw_center)
                avg_x = sum(c[0] for c in center_history) / len(center_history)
                avg_y = sum(c[1] for c in center_history) / len(center_history)
                curr_avg_center = (avg_x, avg_y)

                if prev_avg_center:
                    dist = calculate_distance(prev_avg_center, curr_avg_center)
                    if dist > MOVEMENT_THRESHOLD:
                        movement_counter += 1
                        print(f"뒤척임 감지! (누적 : {movement_counter}회 ), 이동 거리 : {dist:.2f} 픽셀")
                
                prev_avg_center = curr_avg_center
                
                # --- 시각화 ---
                x1, y1, x2, y2 = target['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                
                head_score = get_max_head_conf(target['keypoints'])
                debug_text = f"Head: {head_score:.2f}"
                cv2.putText(frame, debug_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                kpts = target['keypoints']
                for pair in SKELETON_PAIRS:
                    if pair[0] < len(kpts) and pair[1] < len(kpts):
                        pt1 = kpts[pair[0]]
                        pt2 = kpts[pair[1]]
                        if pt1[2] > WEAK_THRESHOLD and pt2[2] > WEAK_THRESHOLD:
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
                
                for i, kp in enumerate(kpts):
                    if kp[2] > WEAK_THRESHOLD:
                        color = (0, 0, 255) if i < 5 else (255, 0, 0)
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)

            else:
                prev_avg_center = None
                center_history.clear()
                status_history.clear() # 사람이 없으면 히스토리 초기화

            # 화면에는 '안정화된' 최종 상태 표시
            cv2.putText(frame, f"STATUS: {final_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"MOVES: {movement_counter}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # --- MQTT 발행 ---
            if time.time() - last_mqtt_time > 1.0:
                payload = {"status": final_status, "movements": movement_counter, "timestamp": time.time()}
                json_str = json.dumps(payload, ensure_ascii=False)
                client.publish(MQTT_TOPIC, json_str)
                print(f"MQTT 발행 성공 : {json_str}")
                last_mqtt_time = time.time()
                
            cv2.imshow("Stabilized Sleep Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release(); cv2.destroyAllWindows(); client.loop_stop()

if __name__ == "__main__":
    main()