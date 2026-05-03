# clients/infer.py
import numpy as np
import tritonclient.http as httpclient
from PIL import Image


def preprocess(image_path, size=640):
    img = Image.open(image_path).convert("RGB").resize((size, size))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # add batch dim → [1, 3, 640, 640]
    return img


client = httpclient.InferenceServerClient(url="localhost:8000")

image = preprocess("data/sample.jpg")

inputs = [httpclient.InferInput("images", image.shape, "FP32")]
inputs[0].set_data_from_numpy(image)

outputs = [httpclient.InferRequestedOutput("output0")]

response = client.infer("yolov8s", inputs=inputs, outputs=outputs)

output = response.as_numpy("output0")
print("Output shape:", output.shape)  # should be [1, 84, 8400]
