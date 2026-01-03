
import sys
import numpy as np
try:
    from dx_engine import InferenceEngine
except ImportError:
    print("Error: dx_engine not found. Run on OrangePi.")
    sys.exit(1)

def check_model(model_path):
    print(f"--- Checking Model: {model_path} ---")
    try:
        engine = InferenceEngine(model_path)
        # We don't need to run inference to get output info if the API supports it, 
        # but dx_engine often reveals shapes only after run or via internal properties if exposed.
        # Let's try running on dummy input to be sure.
        
        # Assume 640x640 input
        dummy_input = np.zeros((640*640*3,), dtype=np.uint8)
        
        print("Running dummy inference...")
        outputs = engine.run([dummy_input])
        
        print(f"Number of Outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"Output [{i}] Shape: {out.shape}")
            # Check last dimension
            last_dim = out.shape[-1]
            print(f"  -> Last Dimension: {last_dim}")
            
            if last_dim == 7:
                print("  => CONFIRMED: 7 Channels (5 Box + 2 Classes: Fire, Smoke)")
            elif last_dim == 6:
                print("  => Single Class (5 Box + 1 Class)")
            elif last_dim == 85:
                print("  => COCO 80 Classes (5 Box + 80 Classes)")
            else:
                print(f"  => Unknown Channel Count: {last_dim}")

    except Exception as e:
        print(f"Error checking model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_dxnn_shape.py <model.dxnn>")
        sys.exit(1)
    
    check_model(sys.argv[1])
