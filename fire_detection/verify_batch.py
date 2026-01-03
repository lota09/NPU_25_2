import sys
import cv2
import numpy as np
import time
import os
from dx_engine import InferenceEngine

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_inference(engine, image_path, is_human_model=False):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return None

    # Resize to 640x640
    img_resized = cv2.resize(img, (640, 640))
    input_data = img_resized.flatten().astype(np.uint8)
    
    start = time.time()
    outputs = engine.run([input_data])
    dt = time.time() - start
    
    output = outputs[0]
    
    # Apply Sigmoid if logits
    if output.max() > 1.5 or output.min() < -1.5:
        output = sigmoid(output)
        
    # Output structure: [1, 25200, 85] (COCO) or [1, 25200, 7] (Fire)
    # Fire: 0-3=Box, 4=Obj, 5=Fire, 6=Smoke
    # Human (COCO): 4=Obj, 5=Person ...
    
    obj_conf = output[..., 4]
    
    if is_human_model:
        # Assuming COCO 80 classes, Person is Class 0 (Index 5)
        cls_conf = output[..., 5] # Class 0 only
    else:
        # Fire model: Class 0 (Fire) is Index 5
        cls_conf = output[..., 5] # Class 0 only
        
    scores = obj_conf * cls_conf
    max_score = scores.max()
    
    print(f"| {os.path.basename(image_path):<15} | {max_score:.4f} | {dt*1000:.1f}ms |")
    return max_score

def verify_batch(model_path, image_dir):
    print(f"Loading Model: {model_path}")
    if not os.path.exists(model_path):
        print("Model file not found!")
        sys.exit(1)
        
    engine = InferenceEngine(model_path)
    
    # Check model output size to guess if it's Human (85) or Fire (7)
    # We can't easily inspect shape without running, so run once dry.
    dummy = np.zeros((1, 640, 640, 3), dtype=np.uint8).flatten()
    outputs = engine.run([dummy])
    shape = outputs[0].shape
    print(f"Model Output Shape: {shape}")
    
    is_human = False
    if shape[-1] > 10: # COCO usually 85, Fire usually 7
        is_human = True
        print("Detected COCO-like model (Human Detection Mode)")
    else:
        print("Detected Custom model (Fire Detection Mode)")
        
    print("-" * 40)
    print(f"| {'Image':<15} | {'Score':<6} | {'Time':<6} |")
    print("-" * 40)
    
    images = ["fire.jpg", "camper.jpg", "room.jpg", "room2.jpg", "thief.jpg"]
    
    for img_name in images:
        full_path = os.path.join(image_dir, img_name)
        run_inference(engine, full_path, is_human)
        
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_batch.py <model.dxnn> <image_directory>")
    else:
        verify_batch(sys.argv[1], sys.argv[2])
