import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import sys

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((640, 640))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # NCHW
    arr = np.expand_dims(arr, 0)  # [1, 3, 640, 640]
    return arr

def run_onnx(onnx_path, img_path):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    arr = preprocess(img_path)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: arr})
    print(f'Output shape: {output[0].shape}')
    print(f'Output stats: min={output[0].min()}, max={output[0].max()}, mean={output[0].mean()}')
    print(f'Output sample: {output[0].flatten()[:10]}')

if __name__ == '__main__':
    onnx_path = sys.argv[1]
    img_path = sys.argv[2]
    run_onnx(onnx_path, img_path)
