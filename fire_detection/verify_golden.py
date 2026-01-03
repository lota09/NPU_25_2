import cv2
import numpy as np
import time
from dx_engine import InferenceEngine

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def verify(model_path, image_path):
    print(f"Loading model: {model_path}")
    engine = InferenceEngine(model_path)
    
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image")
        return

    # Preprocess
    # 640x640, BGR->RGB, Div 255 IS EMBEDDED IN MODEL NORM?
    # Wait, YOLOV7-2.json HAD preprocessing (Resize, Div, RGB, Transpose).
    # IF the DXNN was compiled with these, I should input RAW IMAGE (Resized to 640)?
    # Or does the model expect RAW uint8 HWC?
    # Usually: Resize -> Flatten -> DXNN (which does Norm/Transpose).
    
    # Resize
    img_resized = cv2.resize(img, (640, 640))
    # Pass as UINT8 flat buffer
    input_data = img_resized.flatten().astype(np.uint8)
    
    print("Running inference...")
    start = time.time()
    outputs = engine.run([input_data])
    print(f"Inference time: {time.time() - start:.4f}s")
    
    # Output: [1, 25200, 85]
    output = outputs[0]
    print(f"Output shape: {output.shape}")
    
    # Sigmoid? 
    # Usually YOLO ONNX export includes Sigmoid?
    # Or exclude?
    # Inspect showed "Sigmoid" nodes? No, Inspect showed Input/Output.
    # Let's assume Output is Logits if raw. But "output" name suggests end result.
    # Let's check ranges.
    
    print(f"Min: {output.min()}, Max: {output.max()}")
    
    # Decode
    # 4 (box) + 1 (obj) + 80 (cls)
    
    # If Max > 1.0, unlikely sigmoid.
    if output.max() > 1.5:
        print("Output seems to be logits (applying sigmoid)")
        output = sigmoid(output)
    
    # Check Max Confidence
    # Obj * Cls
    obj_conf = output[..., 4]
    cls_conf = output[..., 5:]
    
    max_obj = obj_conf.max()
    print(f"Max Objectness: {max_obj}")
    
    # Find max class score
    scores = obj_conf[..., None] * cls_conf
    final_max = scores.max()
    print(f"Max Final Score: {final_max}")
    
    if final_max > 0.5:
        print("✅ SUCCESS: High confidence detected!")
    else:
        print("❌ FAILURE: Low confidence.")

import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_golden.py <model.dxnn> <image.jpg>")
        # Default fallback
        verify("YOLOV7-2.dxnn", "assets/inputs/image.jpg")
    else:
        verify(sys.argv[1], sys.argv[2])
