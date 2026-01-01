#!/usr/bin/env python3
"""
Fall Detection Backend API
Flask server that runs fall detection and provides results via REST API
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time
import sys
import os
import cv2
import base64
from datetime import datetime
import random  # 테스트용 온도 데이터 생성

# Add the parent directory to sys.path to import fall_detection
sys.path.append('/root/Real-Time-Fall-Detection-using-YOLO')

from fall_detection import FallDetection

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global variables to store detection results
detection_results = {
    'is_running': False,
    'fall_count': 0,
    'last_fall_time': 0,
    'current_fps': 0,
    'status': 'stopped',
    'last_fall_image': None,  # 낙상 시 캡처된 이미지
    'debug_info': {  # 디버그 정보
        'head_velocity': 0,
        'head_y_position': 0,
        'vertical_distance': 0,
        'people_detected': 0
    }
}

# 수면 분석 데이터
sleep_analysis_data = {
    'toss_turn_history': [],  # [{time: '02:30', count: 1}, ...]
    'temperature_history': [],  # [{time: '02:30', temp: 22.5}, ...]
    'total_toss_turns': 0,
    'sleep_start_time': None,
    'is_monitoring': False
}

# 수면 상태 추적
sleep_state = {
    'last_position': None,  # 마지막 몸통 중심 위치
    'position_stable_time': None,  # 위치가 안정화된 시간
    'movement_threshold': 50  # 뒤척임으로 간주할 이동 거리 (픽셀)
}

detector = None
detection_thread = None
sleep_monitor_thread = None
latest_frame = None  # 최신 프레임 저장

def run_fall_detection():
    """Run fall detection in background thread"""
    global detector, detection_results, latest_frame

    try:
        detection_results['status'] = 'initializing'
        detector = FallDetection()
        detection_results['is_running'] = True
        detection_results['status'] = 'running'

        # Override the detector's methods to capture results
        original_detect_velocity = detector.detect_fall_by_velocity
        original_detect_posture = detector.detect_fall_by_posture

        def save_fall_image(frame):
            """낙상 감지 시 이미지 저장"""
            try:
                # 이미지를 public/images 폴더에 저장
                image_path = '/root/monoculus-app/public/images/last_fall.jpg'
                cv2.imwrite(image_path, frame)
                detection_results['last_fall_image'] = '/images/last_fall.jpg'
                print(f"✅ Fall image saved: {image_path}")
            except Exception as e:
                print(f"❌ Failed to save fall image: {e}")

        def patched_detect_velocity(person_id, head_y, current_time):
            result = original_detect_velocity(person_id, head_y, current_time)
            
            # 디버그 정보 업데이트
            if person_id in detector.head_tracker and len(detector.head_tracker[person_id]) >= 2:
                recent_data = list(detector.head_tracker[person_id])
                if len(recent_data) >= 2:
                    first_time, first_y = recent_data[0]
                    last_time, last_y = recent_data[-1]
                    time_diff = last_time - first_time
                    vertical_distance = last_y - first_y
                    velocity = vertical_distance / time_diff if time_diff > 0 else 0
                    
                    detection_results['debug_info']['head_velocity'] = velocity
                    detection_results['debug_info']['head_y_position'] = head_y
                    detection_results['debug_info']['vertical_distance'] = vertical_distance
            
            if result:
                detection_results['fall_count'] = detector.fall_count
                detection_results['last_fall_time'] = detector.last_fall_time
                if latest_frame is not None:
                    save_fall_image(latest_frame.copy())
            return result

        def patched_detect_posture(person_id, torso_metrics, current_time):
            result = original_detect_posture(person_id, torso_metrics, current_time)
            if result:
                detection_results['fall_count'] = detector.fall_count
                detection_results['last_fall_time'] = detector.last_fall_time
                if latest_frame is not None:
                    save_fall_image(latest_frame.copy())
            return result

        detector.detect_fall_by_velocity = patched_detect_velocity
        detector.detect_fall_by_posture = patched_detect_posture

        # Run webcam detection with frame capture
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            detection_results['status'] = 'error: cannot open camera'
            return
        
        frame_count = 0
        start_time = time.time()
        
        while detection_results['is_running']:
            ret, frame = cap.read()
            if not ret:
                break
            
            latest_frame = frame.copy()  # 최신 프레임 저장
            frame_count += 1
            current_time = time.time()
            
            # Pose Estimation
            if detector.model_type == 'deepx_pose':
                input_frame = cv2.resize(frame, (640, 640))
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_data = np.array(input_frame, dtype=np.uint8).tobytes()
                input_buf = np.frombuffer(input_data, dtype=np.uint8)
                outputs = detector.model.run([input_buf])
                detections = detector.parse_yolov5_pose_output(outputs[0], frame.shape)
                
                # 디버그: 감지된 사람 수
                detection_results['debug_info']['people_detected'] = len(detections)
                
                # Fall Detection
                for person_id, detection in enumerate(detections):
                    keypoints = detection['keypoints']
                    head_pos = detector.get_head_position(keypoints)
                    torso_metrics = detector.get_torso_metrics(keypoints)
                    
                    if head_pos:
                        head_x, head_y = head_pos
                        detector.detect_fall_by_velocity(person_id, head_y, current_time)
                        
                        if torso_metrics:
                            detector.detect_fall_by_posture(person_id, torso_metrics, current_time)
            
            # Update FPS
            elapsed = time.time() - start_time
            detection_results['current_fps'] = frame_count / elapsed if elapsed > 0 else 0
        
        cap.release()

    except Exception as e:
        print(f"Error in fall detection: {e}")
        detection_results['status'] = f'error: {str(e)}'
    finally:
        detection_results['is_running'] = False

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current fall detection status"""
    return jsonify(detection_results)

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start fall detection"""
    global detection_thread

    if detection_results['is_running']:
        return jsonify({'message': 'Already running'}), 400

    detection_thread = threading.Thread(target=run_fall_detection, daemon=True)
    detection_thread.start()

    return jsonify({'message': 'Fall detection started'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop fall detection"""
    detection_results['is_running'] = False
    detection_results['status'] = 'stopped'
    sleep_analysis_data['is_monitoring'] = False
    return jsonify({'message': 'Fall detection stopped'})

@app.route('/api/reset', methods=['POST'])
def reset_detection():
    """Reset fall detection counters"""
    global detector
    if detector:
        detector.fall_count = 0
        detector.last_fall_time = 0
        detector.head_tracker.clear()
        detector.posture_tracker.clear()
        detector.fall_cooldown.clear()

    detection_results['fall_count'] = 0
    detection_results['last_fall_time'] = 0
    detection_results['last_fall_image'] = None
    
    # 수면 분석 리셋
    sleep_analysis_data['toss_turn_history'] = []
    sleep_analysis_data['temperature_history'] = []
    sleep_analysis_data['total_toss_turns'] = 0
    sleep_analysis_data['sleep_start_time'] = None
    sleep_state['last_position'] = None
    sleep_state['position_stable_time'] = None

    return jsonify({'message': 'Detection reset'})

@app.route('/api/sleep-analysis', methods=['GET'])
def get_sleep_analysis():
    """수면 분석 데이터 반환 (뒤척임 횟수 + 온도 변화)"""
    return jsonify(sleep_analysis_data)

@app.route('/api/sleep/start', methods=['POST'])
def start_sleep_monitoring():
    """수면 모니터링 시작"""
    global sleep_monitor_thread
    
    sleep_analysis_data['is_monitoring'] = True
    sleep_analysis_data['sleep_start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 수면 모니터링 스레드 시작
    if sleep_monitor_thread is None or not sleep_monitor_thread.is_alive():
        sleep_monitor_thread = threading.Thread(target=monitor_sleep)
        sleep_monitor_thread.daemon = True
        sleep_monitor_thread.start()
    
    return jsonify({'message': 'Sleep monitoring started'})

@app.route('/api/sleep/stop', methods=['POST'])
def stop_sleep_monitoring():
    """수면 모니터링 중지"""
    sleep_analysis_data['is_monitoring'] = False
    return jsonify({'message': 'Sleep monitoring stopped'})

def monitor_sleep():
    """수면 중 뒤척임 감지"""
    global detector, sleep_state, sleep_analysis_data
    
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return
    
    last_temp_update = time.time()
    
    while sleep_analysis_data['is_monitoring']:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # Pose Estimation
        if detector and detector.model_type == 'deepx_pose':
            input_frame = cv2.resize(frame, (640, 640))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_data = np.array(input_frame, dtype=np.uint8).tobytes()
            input_buf = np.frombuffer(input_data, dtype=np.uint8)
            outputs = detector.model.run([input_buf])
            detections = detector.parse_yolov5_pose_output(outputs[0], frame.shape)
            
            if len(detections) > 0:
                detection = detections[0]
                keypoints = detection['keypoints']
                
                # 몸통 중심 위치 계산 (어깨와 골반의 중점)
                if len(keypoints) >= 12:
                    left_shoulder = keypoints[5][:2]
                    right_shoulder = keypoints[6][:2]
                    left_hip = keypoints[11][:2]
                    right_hip = keypoints[12][:2]
                    
                    torso_center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
                    torso_center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
                    current_position = (torso_center_x, torso_center_y)
                    
                    # 뒤척임 감지
                    if sleep_state['last_position'] is not None:
                        dx = current_position[0] - sleep_state['last_position'][0]
                        dy = current_position[1] - sleep_state['last_position'][1]
                        distance = (dx**2 + dy**2)**0.5
                        
                        # 이동 거리가 threshold를 넘으면 뒤척임으로 간주
                        if distance > sleep_state['movement_threshold']:
                            now = datetime.now()
                            time_str = now.strftime('%H:%M')
                            
                            sleep_analysis_data['total_toss_turns'] += 1
                            
                            # 히스토리에 추가 (같은 분에는 중복 추가 안함)
                            if len(sleep_analysis_data['toss_turn_history']) == 0 or \
                               sleep_analysis_data['toss_turn_history'][-1]['time'] != time_str:
                                sleep_analysis_data['toss_turn_history'].append({
                                    'time': time_str,
                                    'count': 1
                                })
                            else:
                                sleep_analysis_data['toss_turn_history'][-1]['count'] += 1
                            
                            print(f"[수면] 뒤척임 감지: {time_str}, 총 {sleep_analysis_data['total_toss_turns']}회")
                            
                            sleep_state['position_stable_time'] = time.time()
                    
                    sleep_state['last_position'] = current_position
        
        # 온도 데이터 업데이트 (1분마다)
        if time.time() - last_temp_update > 60:
            now = datetime.now()
            time_str = now.strftime('%H:%M')
            
            # 실제 온도 센서 대신 랜덤 데이터 (22.0 ~ 24.0도)
            temp = 22.5 + random.uniform(-1.0, 1.5)
            
            sleep_analysis_data['temperature_history'].append({
                'time': time_str,
                'temp': round(temp, 1)
            })
            
            last_temp_update = time.time()
        
        time.sleep(0.5)  # 0.5초마다 체크
    
    cap.release()

@app.route('/api/fire-detection', methods=['POST'])
def trigger_fire_detection():
    """화재 감지 트리거 (테스트용)"""
    fire_data = {
        'detected': True,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image': 'fire_detected.jpg',
        'temperature': 45.0,
        'smoke_level': 'high'
    }
    return jsonify(fire_data)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve fall detection images"""
    return send_from_directory('/root/monoculus-app/public/images', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
