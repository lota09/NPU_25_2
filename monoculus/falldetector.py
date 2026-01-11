
import numpy as np
from collections import deque

# Keypoint Indices (COCO)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12 
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

class FallDetector:
    """Original Logic from verified code"""
    def __init__(self, config):
        self.config = config
        self.fall_count = 0
        self.last_fall_time = 0
        self.head_tracker = {}
        self.posture_tracker = {}
        self.fall_cooldown = {}
        
        # Load params
        p = self.config["fall_params"]
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
