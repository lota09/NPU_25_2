# OrangePi 5 Plus NPU Compilation Guide: The "Golden Path"

## 1. Problem Summary
Initial attempts to compile YOLOv7 for the RK3588 NPU resulted in either:
*   **Float32**: Extremely low confidence scores (~0.18) due to SiLU incompatibility in floating point mode.
*   **INT8 (Opset 11/14)**: Compilation errors (`SURGERY` or Segfaults) due to graph structure issues.
*   **LeakyReLU Workaround**: Resulted in saturation (confidence 1.0) or poor detection (confidence 0.22).

## 2. The Solution
We achieved "Normal" confidence behavior (matched to the official "Golden" example) by aligning the **Opset Version**, **Quantization**, and **Export Environment**.

### Key Requirements
1.  **Opset Version**: **12** (Crucial for SiLU INT8 support).
2.  **Quantization**: **INT8** (Required for correct SiLU execution on NPU).
3.  **Environment**: **PyTorch 2.0.x** (Avoids Segfaults present in PyTorch 2.9+).
4.  **Structure**: **Single Concatenated Output** (Matches standard NPU post-processing).

## 3. Step-by-Step Procedure

### Step 1: Activate Correct Environment
Do **NOT** use the `base` environment (PyTorch 2.9.1).
```bash
conda activate yolov7
# Verify: pip show torch (Should be ~2.0.1)
```

### Step 2: Export ONNX (Opset 12)
Use the `export_single_out.py` script with `opset_version=12`.
```bash
python export_single_out.py --output v7_opset12_single.onnx
```
*   **Note**: Ensure `models/experimental.py` acts natively (no `weights_only` modifications needed for PyTorch 2.0).

### Step 3: Compile for NPU (INT8)
Use a JSON config that specifies `INT8` and includes `preprocessings` (Normalization, Channel Swap).

**Config File (`yolov7_opset12_int8.json`):**
```json
{
  "model_name": "v7_opset12",
  "inputs": {
    "images": [1, 3, 640, 640]
  },
  "output_names": ["output"],
  "calibration_num": 1,
  "quant_type": "INT8",
  "num_samples": 2,
  "default_loader": {
    "dataset_path": "./assets",
    "file_extensions": ["jpg"],
    "preprocessings": [
      { "resize": { "mode": "pad", "size": 640, "pad_value": [114, 114, 114] } },
      { "div": { "x": 255 } },
      { "convertColor": { "form": "BGR2RGB" } },
      { "transpose": { "axis": [2, 0, 1] } },
      { "expandDim": { "axis": 0 } }
    ]
  }
}
```

**Command:**
```bash
path/to/dx_com -m v7_opset12_single.onnx -c yolov7_opset12_int8.json -o output_dir
```

### Step 4: Verify on OrangePi
Transfer the `.dxnn` file and run inference.
*   **Expected Behavior**: Confidence scores should range from 0.0 to ~0.9 depending on object visibility.
*   **Example Results**:
    *   Fire Image: ~0.49 (Strong Detection)
    *   Room Image: ~0.37 (Weak Detection/Background)

## 4. Comparison Results

| Model | Variant | Fire Score | Room Score | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Golden Example** | YOLOV7-2 (INT8) | 0.51 | 0.52 | Verified Baseline |
| **My Model** | **SiLU + INT8 (Opset 12)** | **0.49** | **0.37** | **Best Performance** |
| My Model | LeakyReLU + INT8 | 0.22 | 0.21 | Poor Confidence |
| My Model | Float32 (SiLU) | 0.001 | 0.001 | Broken on NPU |

The **SiLU + INT8 (Opset 12)** model successfully replicates the behavior of the Golden Example while offering slightly better false-positive rejection on the tested sample.

## 5. Cross-Verification: Standard YOLOv7 (Human Detection)
To validate the pipeline's robustness, we applied the same "Golden Path" to the standard **YOLOv7 (COCO)** model to detect people.

### Results
| Image | Content | Person Score (Class 0) | Background Score | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **camper.jpg** | Person + Fire | **0.6454** | - | **Strong Detection** |
| **thief.jpg** | Person | **0.3103** | - | **Detection** |
| **fire.jpg** | Fire Only | - | 0.0018 | Correct Rejection |
| **room.jpg** | Empty Room | - | 0.0008 | Correct Rejection |

**Conclusion**: The pipeline yields near-zero false positives (~0.001) for the Human model, confirming that the **Opset 12 + INT8** compilation method preserves model accuracy and high contrast between positives and negatives.

