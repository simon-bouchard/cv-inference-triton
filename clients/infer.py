import numpy as np
import tritonclient.http as httpclient
from PIL import Image, ImageDraw

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def preprocess(image_path, size=640):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((size, size))
    arr = np.array(img_resized).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr, img, orig_w, orig_h


def postprocess(output, orig_w, orig_h, conf_thresh=0.25, iou_thresh=0.45, input_size=640):
    # output: [1, 84, 8400] → [8400, 84]
    preds = output[0].transpose(1, 0)

    boxes_cx = preds[:, 0]
    boxes_cy = preds[:, 1]
    boxes_w = preds[:, 2]
    boxes_h = preds[:, 3]
    scores = preds[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > conf_thresh
    boxes_cx, boxes_cy, boxes_w, boxes_h = (
        boxes_cx[mask],
        boxes_cy[mask],
        boxes_w[mask],
        boxes_h[mask],
    )
    confidences, class_ids = confidences[mask], class_ids[mask]

    # cx,cy,w,h → x1,y1,x2,y2 (still in input_size space)
    x1 = boxes_cx - boxes_w / 2
    y1 = boxes_cy - boxes_h / 2
    x2 = boxes_cx + boxes_w / 2
    y2 = boxes_cy + boxes_h / 2

    # scale back to original image size
    x1 = x1 * orig_w / input_size
    y1 = y1 * orig_h / input_size
    x2 = x2 * orig_w / input_size
    y2 = y2 * orig_h / input_size

    # NMS
    keep = nms(x1, y1, x2, y2, confidences, iou_thresh)

    return list(zip(x1[keep], y1[keep], x2[keep], y2[keep], confidences[keep], class_ids[keep]))


def nms(x1, y1, x2, y2, scores, iou_thresh):
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
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def draw(image, detections):
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, conf, cls_id in detections:
        label = f"{COCO_CLASSES[cls_id]} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")
    image.save("data/result.jpg")
    print(f"Saved data/result.jpg with {len(detections)} detections")


# --- main ---
client = httpclient.InferenceServerClient(url="localhost:8000")

image_arr, orig_img, orig_w, orig_h = preprocess("data/sample.jpg")

inputs = [httpclient.InferInput("images", image_arr.shape, "FP32")]
inputs[0].set_data_from_numpy(image_arr)
outputs = [httpclient.InferRequestedOutput("output0")]

response = client.infer("yolov8s", inputs=inputs, outputs=outputs)
output = response.as_numpy("output0")

detections = postprocess(output, orig_w, orig_h)
for x1, y1, x2, y2, conf, cls_id in detections:
    print(f"{COCO_CLASSES[cls_id]:20s} {conf:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

draw(orig_img, detections)
