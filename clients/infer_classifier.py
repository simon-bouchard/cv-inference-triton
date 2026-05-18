# clients/infer_classifier.py
"""
Send a JPEG image to the geoclassifier_pipeline and print the predicted region and confidence.
"""
import sys

import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.jpg"
with open(image_path, "rb") as f:
    image_bytes = f.read()

inputs = [httpclient.InferInput("image_raw", [1], "BYTES")]
inputs[0].set_data_from_numpy(np.array([image_bytes], dtype=object))

outputs = [
    httpclient.InferRequestedOutput("label"),
    httpclient.InferRequestedOutput("confidence"),
]

response = client.infer("geoclassifier_pipeline", inputs=inputs, outputs=outputs)

label = response.as_numpy("label")[0].decode()  # type: ignore[index]
confidence = float(response.as_numpy("confidence")[0])  # type: ignore[index]

print(f"{label}  ({confidence:.1%})")
