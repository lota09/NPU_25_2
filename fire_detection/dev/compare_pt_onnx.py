import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../yolov7'))
from models.experimental import attempt_load
import onnxruntime as ort
import numpy as np
from PIL import Image
import sys
import cv2

def preprocess(img_path, nhwc=False):
    img = Image.open(img_path).convert('RGB').resize((640, 640))
    arr = np.array(img).astype(np.float32) / 255.0
    if nhwc:
        arr = np.expand_dims(arr, 0)  # [1, 640, 640, 3]
    else:
        arr = arr.transpose(2, 0, 1)  # NCHW
        arr = np.expand_dims(arr, 0)  # [1, 3, 640, 640]
    return arr

def run_pytorch(pt_path, arr):
    model = attempt_load(pt_path, map_location='cpu')
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(arr)
        out = model(inp)[0].cpu().numpy()
    return out

def run_onnx(onnx_path, arr):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: arr})[0]
    return out

def main(pt_path, onnx_path, img_path):
    import os
    import cv2
    os.makedirs('outputs', exist_ok=True)

    # Sweep용: fire.jpg, thief.jpg 모두 테스트
    img_paths = [
        ('assets/fire.jpg', 'fire.jpg'),
        ('assets/room2.jpg', 'room2.jpg')
    ]
    # threshold sweep (속도 개선: 0.6~0.95)
    # 다양한 후처리 조합 sweep
    best = None
    for obj_thres in np.arange(0.3, 1.0, 0.05):
        for cls_thres in np.arange(0.3, 1.0, 0.05):
            results = {"and": {}, "obj": {}, "cls": {}, "max": {}}
            for path, label in img_paths:
                arr = preprocess(path, nhwc='nhwc' in onnx_path.lower())
                try:
                    out_onnx = run_onnx(onnx_path, arr)
                    if out_onnx.ndim == 5:
                        out_onnx = out_onnx.reshape(-1, out_onnx.shape[-1])
                    elif out_onnx.ndim == 4:
                        out_onnx = out_onnx.reshape(-1, out_onnx.shape[-1])
                    elif out_onnx.ndim == 3:
                        out_onnx = out_onnx[0]
                    conf = 1 / (1 + np.exp(-out_onnx[:, 4]))
                    class_scores = 1 / (1 + np.exp(-out_onnx[:, 5:]))
                    fire_conf = class_scores[:, 0]
                    # (1) and
                    mask_and = (conf > obj_thres) & (fire_conf > cls_thres)
                    # 박스 시각화: 가장 보수적인 조건(and)만 저장
                    if obj_thres == 0.5 and cls_thres == 0.5 and label in ["fire.jpg", "room2.jpg"]:
                        dets = out_onnx[mask_and]
                        img0 = cv2.imread(path)
                        h0, w0 = img0.shape[:2]
                        for det in dets:
                            # det: [x, y, w, h, obj, fire, smoke]
                            # 좌표 변환 (YOLOv7: cx,cy,w,h → x1,y1,x2,y2)
                            cx, cy, w, h = det[0], det[1], det[2], det[3]
                            x1 = int((cx - w/2) * w0)
                            y1 = int((cy - h/2) * h0)
                            x2 = int((cx + w/2) * w0)
                            y2 = int((cy + h/2) * h0)
                            conf_score = 1 / (1 + np.exp(-det[4]))
                            fire_score = 1 / (1 + np.exp(-det[5]))
                            smoke_score = 1 / (1 + np.exp(-det[6])) if det.shape[0] > 6 else 0
                            color = (0,0,255) if fire_score > smoke_score else (255,0,0)
                            cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                            label_str = f"fire {fire_score:.2f}" if fire_score > smoke_score else f"smoke {smoke_score:.2f}"
                            cv2.putText(img0, label_str, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        out_path = f"outputs/{label.replace('.jpg','')}_onnx_vis.jpg"
                        cv2.imwrite(out_path, img0)
                        print(f"[ONNX] {label} 시각화 결과: {out_path}")
                    # (2) objectness만
                    mask_obj = (conf > obj_thres)
                    # (3) class만
                    mask_cls = (fire_conf > cls_thres)
                    # (4) max(objectness, class)
                    mask_max = (np.maximum(conf, fire_conf) > max(obj_thres, cls_thres))
                    results["and"][label] = np.count_nonzero(mask_and)
                    results["obj"][label] = np.count_nonzero(mask_obj)
                    results["cls"][label] = np.count_nonzero(mask_cls)
                    results["max"][label] = np.count_nonzero(mask_max)
                except Exception as e:
                    results["and"][label] = results["obj"][label] = results["cls"][label] = results["max"][label] = f'실패: {e}'
            # 각 방식별로 fire.jpg에서 1개 이상, thief.jpg에서 0개일 때만 출력
            for mode in ["and", "obj", "cls", "max"]:
                if results[mode].get('fire.jpg', 0) > 0 and results[mode].get('thief.jpg', 0) == 0:
                    print(f'{mode} | objectness>{obj_thres:.2f}, class>{cls_thres:.2f}  fire.jpg: {results[mode].get("fire.jpg")}, thief.jpg: {results[mode].get("thief.jpg")}  (성공!)')
                    if best is None:
                        best = (mode, obj_thres, cls_thres, results[mode].get('fire.jpg'))
    if best:
        print(f'\n최적 threshold: {best[0]} | objectness>{best[1]:.2f}, class>{best[2]:.2f}  fire.jpg에서 {best[3]}개 검출, thief.jpg에서 0개')
    else:
        print('\n성공 조건을 만족하는 threshold/후처리 조합이 없습니다.')

if __name__ == '__main__':
    pt_path = sys.argv[1]
    onnx_path = sys.argv[2]
    img_path = sys.argv[3]
    main(pt_path, onnx_path, img_path)
