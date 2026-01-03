import os
import sys
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2


def preprocess(img_path, nhwc=False, size=(640, 640)):
    img = Image.open(img_path).convert('RGB').resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    if nhwc:
        arr = np.expand_dims(arr, 0)
    else:
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, 0)
    return arr, np.array(img)


def run_onnx(onnx_path, arr):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: arr})[0]
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh_to_xyxy(boxes):
    # boxes: [N,4] with center x,y,w,h
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def visualize(onnx_path, img_path, output_path, nhwc=False, conf_thres=0.5, cls_thres=0.5):
    arr, pil_img = preprocess(img_path, nhwc=nhwc)
    H, W = pil_img.shape[0], pil_img.shape[1]
    out = run_onnx(onnx_path, arr)

    # reshape to (N,7)
    if out.ndim == 5:
        out = out.reshape(-1, out.shape[-1])
    elif out.ndim == 4:
        out = out.reshape(-1, out.shape[-1])
    elif out.ndim == 3:
        out = out[0]

    # coords and logits
    coords = out[:, :4]
    obj_logits = out[:, 4]
    class_logits = out[:, 5:]
    obj_conf = sigmoid(obj_logits)
    class_conf = sigmoid(class_logits)
    # assume class 0 == fire
    fire_conf = class_conf[:, 0]

    mask = (obj_conf > conf_thres) & (fire_conf > cls_thres)
    if mask.sum() == 0:
        print(f'No detections above thresholds for {os.path.basename(img_path)}')
    boxes = coords[mask]
    scores = (obj_conf[mask] * fire_conf[mask])

    # convert coords to pixel xyxy (assume coords are in px with image size 640)
    # If coordinates are normalized, multiply by W/H; try to detect scale
    # Heuristic: if max coord > 1 and <= max(W,H)*2 assume pixel coords
    if boxes.size > 0:
        maxc = boxes.max()
    else:
        maxc = 0
    if maxc <= 1.0:
        # normalized (0..1)
        boxes[:, 0] *= W
        boxes[:, 1] *= H
        boxes[:, 2] *= W
        boxes[:, 3] *= H
    # convert center xywh -> xyxy
    boxes_xyxy = xywh_to_xyxy(boxes)
    # clip
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, W - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, H - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, W - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, H - 1)

    # NMS
    keep = nms(boxes_xyxy, scores, iou_thres=0.45)
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]

    img = pil_img.copy()
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = b.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f'fire {scores[i]:.2f}'
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f'Saved visualization: {output_path}  (boxes: {len(boxes_xyxy)})')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python dev/onnx_visualize.py <onnx> <img1> <img2> [output_dir]')
        sys.exit(1)
    onnx_path = sys.argv[1]
    img1 = sys.argv[2]
    img2 = sys.argv[3]
    outdir = sys.argv[4] if len(sys.argv) > 4 else 'outputs'
    visualize(onnx_path, img1, os.path.join(outdir, os.path.basename(img1)))
    visualize(onnx_path, img2, os.path.join(outdir, os.path.basename(img2)))
