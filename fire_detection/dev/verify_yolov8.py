import cv2
import numpy as np
import sys
import os

from dx_engine import InferenceEngine

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # YOLOv8 expects RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 640x640
    img_resized = cv2.resize(img_rgb, (640, 640))
    
    # Flatten for NPU input (UINT8)
    return img_resized.flatten(), img_rgb

def postprocess(output):
    # Output shape: [1, 84, 8400]
    # 0-3: Box (cx, cy, w, h)
    # 4-83: Class Scores (Logits?? Usually ONNX export from Ultralytics is stripped?)
    # Note: Ultralytics ONNX typically contains Sigmoid? Or not?
    # We will assume Logits and apply Sigmoid if range is outside [0,1].
    
    # Transpose to [1, 8400, 84] for easier handling
    data = output[0].transpose(1, 0) # [8400, 84]
    
    scores = data[:, 4:] # [8400, 80]
    
    # Check range
    min_val = np.min(scores)
    max_val = np.max(scores)
    print(f"Raw Output Stats - Min: {min_val:.4f}, Max: {max_val:.4f}")
    
    if max_val > 1.0 or min_val < 0.0:
        print("Applying Sigmoid...")
        scores = sigmoid(scores)
    
    max_score = np.max(scores)
    max_idx = np.argmax(scores)
    
    # unravel index (row, col)
    row_idx = max_idx // 80 
    class_idx = max_idx % 80
    
    print(f"Max Score: {max_score:.4f} at Anchor {row_idx}, Class {class_idx}")
    
    box = data[row_idx, :4]
    print(f"Box for Max Score: {box}")
    
    # Check Class 0 (Person)
    person_scores = scores[:, 0]
    max_person = np.max(person_scores)
    print(f"Max Person Score: {max_person:.4f}")

    return max_person

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_yolov8.py <model.dxnn> <image.jpg>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    
    # from dx_rt import InferenceEngine # Already imported globally

    ie = InferenceEngine(model_path)
    input_data, _ = preprocess(img_path)
    
    # Run
    # Wrap in list
    outputs = ie.run([input_data])
    
    print(f"Output Shapes: {[o.shape for o in outputs]}")
    
    postprocess(outputs[0])
