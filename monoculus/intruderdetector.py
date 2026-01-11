
from collections import deque

class IntruderDetector:
    def __init__(self, config):
        self.config = config
        self.last_intruder_alert = 0
        self.cool = self.config.get("thresholds", {}).get("alert_cooldown", 5)
        
        # Config Params
        t_conf = self.config.get("thresholds", {})
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
        if not self.conf_history: return False, False, 0.0
        avg_conf = sum(self.conf_history) / len(self.conf_history)
        
        is_intruder = False
        should_alert = False

        if avg_conf >= self.conf_thresh:
            is_intruder = True
            if current_time - self.last_intruder_alert > self.cool:
                should_alert = True
                self.last_intruder_alert = current_time
            
        return is_intruder, should_alert, avg_conf
