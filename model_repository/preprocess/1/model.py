import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
import io


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            image_raw = pb_utils.get_input_tensor_by_name(request, "image_raw")
            image_bytes = image_raw.as_numpy()[0]

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((640, 640))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, axis=0)

            out = pb_utils.Tensor("images", arr)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

        return responses

    def finalize(self):
        pass
