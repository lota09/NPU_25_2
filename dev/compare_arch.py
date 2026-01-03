
import onnx
import sys
import os

def check_shape(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    try:
        model = onnx.load(path)
        output = model.graph.output[0]
        dim_str = []
        for d in output.type.tensor_type.shape.dim:
            if d.dim_value:
                dim_str.append(str(d.dim_value))
            elif d.dim_param:
                dim_str.append(d.dim_param)
            else:
                dim_str.append("?")
        print(f"Model: {os.path.basename(path)}")
        print(f"  - Output Name: {output.name}")
        print(f"  - Output Shape: [{', '.join(dim_str)}]")
        
        # Heuristic
        dims = [int(d) if d.isdigit() else 0 for d in dim_str]
        if 8400 in dims:
            print("  => DETECTED: YOLOv8 Architecture (8400 columns = 80x80 + 40x40 + 20x20 in anchor-free format)")
        elif 25200 in dims:
            print("  => DETECTED: YOLOv7/v5 Architecture (25200 columns = 3 anchors * (80x80 + 40x40 + 20x20))")
        else:
            print("  => Unknown Architecture")
            
    except Exception as e:
        print(f"Error loading {path}: {e}")

print("--- Comparing Architectures ---")
check_shape("fire_detection/yolov7/v7_human_opset12.onnx")
check_shape("fire_detection/models/yolov8n_human.onnx")
