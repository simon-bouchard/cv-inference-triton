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

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = img.size
            scale = min(640 / w, 640 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            pad = Image.new("RGB", (640, 640), (114, 114, 114))
            pad.paste(img, ((640 - new_w) // 2, (640 - new_h) // 2))
            arr = np.array(pad).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, axis=0)

            out = pb_utils.Tensor("images", arr)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

        return responses

    def finalize(self):
        pass
