import numpy as np
import tritonclient.http as httpclient

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

client = httpclient.InferenceServerClient(url="localhost:8000")

with open("data/sample.jpg", "rb") as f:
    image_bytes = f.read()

# wrap bytes in a numpy array of dtype object
image_data = np.array([image_bytes], dtype=object)

inputs = [httpclient.InferInput("image_raw", [1], "BYTES")]
inputs[0].set_data_from_numpy(image_data)

outputs = [
    httpclient.InferRequestedOutput("boxes"),
    httpclient.InferRequestedOutput("scores"),
    httpclient.InferRequestedOutput("class_ids"),
]

response = client.infer("yolov8s_pipeline", inputs=inputs, outputs=outputs)

boxes     = response.as_numpy("boxes")
scores    = response.as_numpy("scores")
class_ids = response.as_numpy("class_ids")

print(f"{len(scores)} detections:")
for i in range(len(scores)):
    cls = COCO_CLASSES[class_ids[i]]
    conf = scores[i]
    box = boxes[i]
    print(f"  {cls:20s} {conf:.2f}  [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
