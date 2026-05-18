# model_repository/geoclassifier_postprocess/1/model.py
"""
Triton Python backend: convert raw logits to a region label and confidence score.
"""
import numpy as np
import triton_python_backend_utils as pb_utils

LABELS = [
    "Abitibi-Temiscamingue", "Bas-Saint-Laurent", "Capitale-Nationale",
    "Centre-du-Quebec", "Chaudiere-Appalaches", "Cote-Nord", "Estrie",
    "Gaspesie-Iles-de-la-Madeleine", "Lanaudiere", "Laurentides", "Laval",
    "Mauricie", "Monteregie", "Montreal", "Nord-du-Quebec",
    "Outaouais", "Saguenay-Lac-Saint-Jean",
]


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        pass

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()[0]  # [17]

            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()

            idx = int(probs.argmax())
            label = np.array([LABELS[idx]], dtype=object)
            confidence = np.array([probs[idx]], dtype=np.float32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("label", label),
                pb_utils.Tensor("confidence", confidence),
            ]))

        return responses

    def finalize(self) -> None:
        pass
