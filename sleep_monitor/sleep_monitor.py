import cv2
import dx_engine as dx  # DX-RT 런타임 API
import numpy as np
import time
import torch
import torchvision
from ultralytics.utils import ops
import math 
import json # 5단계: JSON 사용
import paho.mqtt.client as mqtt # 5단계: MQTT 라이브러리

# --- 1. 설정 (MQTT 설정 추가) ---
MODEL_PATH = "/home/orangepi/deepx_sdk/dx_app/assets/models/YOLOV5Pose640_1.dxnn"
CONF_THRESHOLD = 0.3 
IOU_THRESHOLD = 0.45 
N_CLASSES = 1 
INPUT_SIZE = (640, 640) 

# ★★★ 4단계: 뒤척임 민감도 (이 값을 조절하세요) ★★★
MOVEMENT_THRESHOLD = 15 # 픽셀 이동 임계값 (값이 낮을수록 민감)

# ★★★ 5단계: MQTT 설정 ★★★
MQTT_BROKER_HOST = "broker.hivemq.com" # 테스트용 공개 브로커
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "sleep_monitor/user/test01" # 데이터를 발행할 토픽 (게시판 주소)

# --- (이하 관절 인덱스, 페어, 색상, 레이어 설정은 동일) ---
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12 

KEYPOINT_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
]
SKELETON_COLOR = (255, 100, 0)
KEYPOINT_COLOR = (0, 0, 255)

LAYER_CONFIG = [
    {"stride": 8,  "anchor_width": [19.0, 44.0, 38.0], "anchor_height": [27.0, 40.0, 94.0]},
    {"stride": 16, "anchor_width": [72.0, 103.0, 187.0], "anchor_height": [92.0, 198.0, 141.0]},
    {"stride": 32, "anchor_width": [156.0, 237.0, 373.0], "anchor_height": [287.0, 397.0, 525.0]}
]

# --- 2. NPU 모델 로드 ---
print("NPU 모델을 로드합니다...")
ie = dx.InferenceEngine(MODEL_PATH)
print("모델 로드 완료.")

# --- 3. 카메라 설정 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
    exit()
    
# --- 5단계: MQTT 클라이언트 설정 ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"MQTT 브로커에 연결되었습니다 (Host: {MQTT_BROKER_HOST})")
    else:
        print(f"MQTT 연결 실패 (Code: {rc})")

client = mqtt.Client()
client.on_connect = on_connect
try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.loop_start() # MQTT 클라이언트를 백그라운드 스레드에서 실행
except Exception as e:
    print(f"MQTT 연결 오류: {e}")
    print("MQTT 없이 로컬로만 실행합니다.")


# --- 4. 전처리 함수 ---
def letter_box(image_src, new_shape=(640, 640), fill_color=(114, 114, 114)):
    src_shape = image_src.shape[:2]
    r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])
    ratio = (r, r) 
    new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if src_shape[::-1] != new_unpad:
        image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_new = cv2.copyMakeBorder(image_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return image_new, ratio, (dw, dh) 

# --- 5. 파싱 함수 ---
def ppu_decode_pose(ie_outputs, layer_config, n_classes, input_shape):
    ie_output = ie_outputs[0][0]
    num_det = ie_output.shape[0]
    KPT_SHAPE = 51
    tensor_size = n_classes + 5 + KPT_SHAPE 
    decoded_tensor = []
    
    for detected_idx in range(num_det):
        tensor = np.zeros(tensor_size, dtype=np.float32)
        data = ie_output[detected_idx].tobytes()
        box = np.frombuffer(data[0:16], dtype=np.float32)
        gy, gx, anchor, layer = np.frombuffer(data[16:20], dtype=np.uint8)
        score = np.frombuffer(data[20:24], dtype=np.float32)[0]
        kpts = np.frombuffer(data[28:28 + (KPT_SHAPE * 4)], dtype=np.float32)

        if layer >= len(layer_config): continue
        cfg = layer_config[layer]
        w, h, s = cfg["anchor_width"][anchor], cfg["anchor_height"][anchor], cfg["stride"]
        grid = np.array([gx, gy], dtype=np.float32)
        anchor_wh = np.array([w, h], dtype=np.float32)
        
        xc = (grid - 0.5 + (box[0:2] * 2)) * s
        wh = (box[2:4] ** 2) * 4 * anchor_wh
        
        tensor[0:4] = np.concatenate([xc, wh], axis=0)
        tensor[4] = score
        tensor[5] = score
        
        for i in range(17):
            start = 5 + n_classes + (i * 3)
            tensor[start:start+2] = (kpts[i*3:i*3+2] * 2 + grid - 0.5) * s
            tensor[start+2] = kpts[i*3+2]
        
        decoded_tensor.append(tensor)

    if len(decoded_tensor) == 0:
        return np.zeros((0, tensor_size), dtype=np.float32)
    
    return np.stack(decoded_tensor)

# --- 6. 후처리 및 그리기 ---
def post_process_and_draw(frame, decoded_tensor, img_shape, ratio, pad):
    boxes = ops.xywh2xyxy(decoded_tensor[..., :4])
    tensor_with_xyxy = np.concatenate([boxes, decoded_tensor[..., 4:]], axis=-1)
    
    x = torch.Tensor(tensor_with_xyxy)
    x = x[torchvision.ops.nms(x[:, :4], x[:, 4], IOU_THRESHOLD)]
    
    combined_ratio_pad = (ratio, pad)
    
    x[:, :4] = ops.scale_boxes(INPUT_SIZE, x[:, :4], img_shape, ratio_pad=combined_ratio_pad).round()
    for i in range(17):
        kpt_idx = 6 + (i * 3)
        x[:, kpt_idx:kpt_idx+2] = ops.scale_coords(INPUT_SIZE, x[:, kpt_idx:kpt_idx+2], img_shape, ratio_pad=combined_ratio_pad).round()

    all_person_keypoints = []
    
    for person in x.numpy():
        conf = person[4]
        if conf < CONF_THRESHOLD:
            continue
            
        x1, y1, x2, y2 = person[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        valid_keypoints = []
        keypoints = person[6:].reshape(17, 3) 
        all_person_keypoints.append(keypoints) 
        
        for i in range(17):
            kpt_x, kpt_y, kpt_conf = keypoints[i]
            if kpt_conf > CONF_THRESHOLD:
                px, py = int(kpt_x), int(kpt_y)
                cv2.circle(frame, (px, py), 5, KEYPOINT_COLOR, -1)
                valid_keypoints.append((i, px, py))
            else:
                valid_keypoints.append((i, -1, -1))
        
        for (idx1, idx2) in KEYPOINT_PAIRS:
            _, p1_x, p1_y = valid_keypoints[idx1]
            _, p2_x, p2_y = valid_keypoints[idx2]
            if p1_x != -1 and p2_x != -1:
                cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), SKELETON_COLOR, 2)
    
    return frame, all_person_keypoints

# --- 4단계 함수: 엉덩이 중심 계산 ---
def get_body_center(kpts):
    points = []
    if kpts[L_SHOULDER][2] > CONF_THRESHOLD:
        points.append(kpts[L_SHOULDER][:2])
    if kpts[R_SHOULDER][2] > CONF_THRESHOLD:
        points.append(kpts[R_SHOULDER][:2])
    if kpts[L_HIP][2] > CONF_THRESHOLD:
        points.append(kpts[L_HIP][:2])
    if kpts[R_HIP][2] > CONF_THRESHOLD:
        points.append(kpts[R_HIP][:2])
    
    if len(points) >= 2:
        avg = np.mean(points, axis=0)
        return (avg[0], avg[1])
    return None

# --- 4단계 함수: 거리 계산 ---
def calculate_distance(p1, p2):
    if p1 is None or p2 is None:
        return 0
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- 3단계 함수: 자세 판별 ---
def get_posture(keypoints_list):
    if not keypoints_list: 
        return "BED_EXIT" 

    kpts = keypoints_list[0] 
    
    nose_conf = kpts[NOSE][2]
    l_eye_conf = kpts[L_EYE][2]
    r_eye_conf = kpts[R_EYE][2]
    l_shoulder_conf = kpts[L_SHOULDER][2]
    r_shoulder_conf = kpts[R_SHOULDER][2]

    if nose_conf < 0.3 and l_eye_conf < 0.3 and r_eye_conf < 0.3:
        return "PRONE (엎드림)"
    if l_shoulder_conf > 0.3 and r_shoulder_conf > 0.3:
        l_shoulder_y = kpts[L_SHOULDER][1]
        r_shoulder_y = kpts[R_SHOULDER][1]
        shoulder_y_diff = abs(l_shoulder_y - r_shoulder_y)
        VERTICAL_THRESHOLD = 30.0 
        if shoulder_y_diff > VERTICAL_THRESHOLD:
            return "UPRIGHT (정자세)"
        else:
            return "SIDE (측면)"
    elif l_shoulder_conf > 0.3 or r_shoulder_conf > 0.3:
        return "SIDE (측면)"
    else:
        return "IN_BED (자세 불명)"

# --- 7. 실시간 추론 루프 (5단계 MQTT 발행 추가) ---
prev_body_center = None 
movement_counter = 0   
last_publish_time = time.time() # 5단계: 1초에 한 번만 발행하기 위한 타이머

try:
    while True:
        ret, frame_orig = cap.read()
        if not ret:
            print("오류: 프레임을 읽을 수 없습니다.")
            continue

        frame_input, ratio, pad = letter_box(frame_orig, new_shape=INPUT_SIZE)
        outputs = ie.run([frame_input])
        
        try:
            decoded_tensor = ppu_decode_pose(outputs, LAYER_CONFIG, N_CLASSES, INPUT_SIZE)
        except Exception as e:
            decoded_tensor = np.zeros((0, N_CLASSES + 5 + 51), dtype=np.float32)

        all_keypoints = [] 
        if decoded_tensor.shape[0] > 0: 
            frame_result, all_keypoints = post_process_and_draw(frame_orig, decoded_tensor, frame_orig.shape[:2], ratio, pad)
        else:
            frame_result = frame_orig 
        
        status = get_posture(all_keypoints)
        
        if status == "BED_EXIT":
            status_color = (0, 0, 255)
            prev_body_center = None
        else:
            status_color = (0, 255, 0)
            
            if all_keypoints: 
                current_body_center = get_body_center(all_keypoints[0])
                if current_body_center and prev_body_center:
                    distance = calculate_distance(prev_body_center, current_body_center)
                    if distance > MOVEMENT_THRESHOLD:
                        movement_counter += 1
                        print(f"뒤척임 감지! (누적: {movement_counter}회), 이동 거리: {distance:.2f} 픽셀")
                prev_body_center = current_body_center
            else:
                prev_body_center = None 

        cv2.putText(frame_result, f"STATUS: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        cv2.putText(frame_result, f"MOVEMENTS: {movement_counter}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # --- ★★★ 5단계: MQTT 발행 로직 ★★★ ---
        current_time = time.time()
        # 1초에 한 번씩만 MQTT 메시지 발행
        if (current_time - last_publish_time) > 1.0:
            # 보낼 데이터를 딕셔너리로 만듦
            payload = {
                "status": status,
                "movements": movement_counter,
                "timestamp": current_time
            }
            # 딕셔너리를 JSON 문자열로 변환
            payload_json = json.dumps(payload)
            
            # MQTT로 발행
            result = client.publish(MQTT_TOPIC, payload_json)
            
            # (디버깅) 터미널에 발행 상태 출력
            if result[0] == 0:
                print(f"MQTT 발행 성공: {payload_json}")
            else:
                print(f"MQTT 발행 실패 (Code: {result[0]})")

            last_publish_time = current_time # 타이머 리셋
        # --- (5단계 로직 끝) ---

        cv2.imshow("Sleep Monitor - Step 5 (MQTT)", frame_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("종료합니다...")
    client.loop_stop() # MQTT 루프 정지
    cap.release()
    cv2.destroyAllWindows()