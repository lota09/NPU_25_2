
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
        
        # Last Dimension Analysis
        last_dim = int(dim_str[-1]) if dim_str[-1].isdigit() else 0
        num_classes = last_dim - 5
        print(f"  => Last Dimension: {last_dim}")
        print(f"  => inferred Classes: {num_classes} (if YOLO format)")
            
    except Exception as e:
        print(f"Error loading {path}: {e}")

print("--- Inspecting Fire Model ---")
check_shape("fire_detection/models/v7_opset12_single.onnx")
