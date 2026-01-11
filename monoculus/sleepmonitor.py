
import math
import numpy as np
from collections import deque, Counter
from datetime import datetime

# Keypoint Indices (COCO)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12 
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

class SleepMonitor:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=self.config["thresholds"]["sleep_buffer"])
        self.center_history = deque(maxlen=5)
        self.moves = 0
        self.prev_center = None
        self.move_thresh = self.config["thresholds"]["move"]
        self.conf_thresh = self.config["thresholds"]["confidence"]
        self.last_reset_date = datetime.now().date()  # 일일 리셋용
        
    def _should_count_movement(self, ha_data):
        """DEEP 모드일 때만 뒤척임 카운트"""
        if not ha_data: return False
        
        # Check status safely
        status = str(ha_data.get('status', ''))
        if not status and 'attributes' in ha_data:
            status = str(ha_data['attributes'].get('status', ''))
            
        return 'DEEP' in status.upper()
        
    def _check_daily_reset(self):
        """자정이 지나면 카운터 리셋"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            print(f"🌙 [SLEEP] Daily reset: {self.moves} movements recorded yesterday")
            self.moves = 0
            self.last_reset_date = today
        
    def process(self, det, ha_data=None):
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
        
        # Check for daily reset
        self._check_daily_reset()
        
        # 2. Movement (only count during night hours)
        curr_center = self._get_center(kpts, bbox)
        self.center_history.append(curr_center)
        avg_x = sum(c[0] for c in self.center_history) / len(self.center_history)
        avg_y = sum(c[1] for c in self.center_history) / len(self.center_history)
        current_avg = (avg_x, avg_y)
        
        if self.prev_center and self._should_count_movement(ha_data):
            dist = math.sqrt((self.prev_center[0]-current_avg[0])**2 + (self.prev_center[1]-current_avg[1])**2)
            if dist > self.move_thresh: 
                self.moves += 1
                print(f"🔄 [SLEEP] Movement detected: {dist:.1f}px (Total: {self.moves})")
            
        self.prev_center = current_avg
        return final, self.moves

    def _get_center(self, kpts, bbox):
        if kpts[L_HIP][2] > 0.3 and kpts[R_HIP][2] > 0.3:
            return ((kpts[L_HIP][0]+kpts[R_HIP][0])/2, (kpts[L_HIP][1]+kpts[R_HIP][1])/2)
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
