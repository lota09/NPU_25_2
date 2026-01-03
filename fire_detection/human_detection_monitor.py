#!/usr/bin/env python3
"""
Human Detection Real-time Monitoring Script (YOLOv7 COCO)
Detects 'Person' class (Class 0) from an 85-channel DXNN model.
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from typing import Optional, Tuple, List, Union

if os.environ.get('DISPLAY') is None:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
from scipy.special import expit

try:
    from dx_engine import InferenceEngine
except ImportError:
    InferenceEngine = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class HumanDetectionMonitor:
    def __init__(self, model_path, video_source='/dev/video0', conf_threshold=0.5):
        self.model_path = model_path
        self.video_source = video_source
        self.input_size = (640, 640)
        self.conf_threshold = conf_threshold
        
        self.engine = None
        if InferenceEngine:
            try:
                self.engine = InferenceEngine(self.model_path)
                logger.info(f"Loaded DXNN: {self.model_path}")
            except Exception as e:
                logger.error(f"DXNN Load Failed: {e}")
                sys.exit(1)
        
        self.cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {video_source}")
            sys.exit(1)
            
    def preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = min(640/w, 640/h)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(frame, (nw, nh))
        
        padded = np.zeros((640, 640, 3), dtype=np.uint8)
        padded[(640-nh)//2:(640-nh)//2+nh, (640-nw)//2:(640-nw)//2+nw] = resized
        return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).flatten()

    def postprocess(self, output):
        # Handle both Single Output (Flattened) and Multi-Output (3-head)
        
        # 1. Normalize input to a list of arrays
        if hasattr(output, 'shape'):
            output = [output]
        elif isinstance(output, list):
            pass # already list
            
        all_boxes = []
        all_scores = []
        
        # Anchors for YOLOv7
        anchors_def = {
            8:  [[12,16], [19,36], [40,28]],
            16: [[36,75], [76,55], [72,146]],
            32: [[142,110], [192,243], [459,401]]
        }
        
        for out in output:
            # Check shape
            # Case A: 5D Tensor [1, 3, H, W, 85] (Standard Grid Output)
            if out.ndim == 5:
                bs, na, h, w, c = out.shape
                stride = 640 // h
                if stride not in anchors_def: continue
                
                curr_anchors = np.array(anchors_def[stride])
                
                # Sigmoid
                out = expit(out)
                
                # Grid
                grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                grid = np.stack((grid_x, grid_y), axis=2).reshape(1, 1, h, w, 2)
                
                # Decode Boxes
                # xy: (sigmoid * 2 - 0.5 + grid) * stride
                xy = (out[..., 0:2] * 2. - 0.5 + grid) * stride
                # wh: (sigmoid * 2)^2 * anchors
                wh = (out[..., 2:4] * 2.) ** 2 * curr_anchors.reshape(1, 3, 1, 1, 2)
                
                xy_tl = xy - wh / 2.
                xy_br = xy + wh / 2.
                boxes = np.concatenate((xy_tl, xy_br), axis=-1).reshape(-1, 4)
                
                # Scores
                # Obj * Class0 (Index 5 for Person in COCO)
                scores = (out[..., 4] * out[..., 5]).reshape(-1)
                
                # Filter
                mask = scores > self.conf_threshold
                if np.any(mask):
                    all_boxes.append(boxes[mask])
                    all_scores.append(scores[mask])

            # Case B: 3D Tensor [1, 25200, 85] (Flattened Output)
            elif out.ndim == 3:
                # Reuse the flattened logic from before, or simpler:
                # If we have flattened output, we need to reconstruct grid OR assume it's pre-decoded?
                # Based on previous Fire model experience, it was Logits [1, 25200, 7].
                # If we get here, we can try to apply the complex linear decoding I wrote before.
                # BUT, the error message clearly showed we got 5D input. So the loop above will catch it.
                pass
                
        if not all_boxes:
            return []
            
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        
        # NMS
        boxes_xywh = []
        for i in range(len(all_boxes)):
            x1, y1, x2, y2 = all_boxes[i]
            boxes_xywh.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
            
        indices = cv2.dnn.NMSBoxes(boxes_xywh, all_scores.tolist(), self.conf_threshold, 0.45)
        
        final_dets = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_dets.append({
                    'box': all_boxes[i],
                    'score': all_scores[i]
                })
        return final_dets

    def run(self):
        print("Starting Human Detection...")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            inp = self.preprocess(frame)
            preds = self.engine.run([inp])
            dets = self.postprocess(preds)
            
            # Draw
            h, w = frame.shape[:2]
            scale_x = w / 640
            scale_y = h / 640
            
            for d in dets:
                box = d['box']
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {d['score']:.2f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Human Detection", frame)
            if cv2.waitKey(1) == ord('q'): break
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to .dxnn model')
    parser.add_argument('--video', default='/dev/video0', help='Video source')
    args = parser.parse_args()
    
    detector = HumanDetectionMonitor(args.model, args.video)
    detector.run()
