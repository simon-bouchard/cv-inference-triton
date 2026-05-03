import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.conf_thresh = 0.25
        self.iou_thresh  = 0.45

    def execute(self, requests):
        responses = []
        for request in requests:
            output0 = pb_utils.get_input_tensor_by_name(request, "output0").as_numpy()

            # [1, 84, 8400] → [8400, 84]
            preds = output0[0].transpose(1, 0)

            cx, cy, w, h = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
            scores       = preds[:, 4:]
            class_ids    = np.argmax(scores, axis=1)
            confidences  = scores[np.arange(len(scores)), class_ids]

            mask = confidences > self.conf_thresh
            cx, cy, w, h       = cx[mask], cy[mask], w[mask], h[mask]
            confidences        = confidences[mask]
            class_ids          = class_ids[mask]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            keep = self._nms(x1, y1, x2, y2, confidences)

            boxes     = np.stack([x1[keep], y1[keep], x2[keep], y2[keep]], axis=1).astype(np.float32)
            scores_out = confidences[keep].astype(np.float32)
            ids_out    = class_ids[keep].astype(np.int32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("boxes",     boxes),
                pb_utils.Tensor("scores",    scores_out),
                pb_utils.Tensor("class_ids", ids_out),
            ]))

        return responses

    def _nms(self, x1, y1, x2, y2, scores):
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep  = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w   = np.maximum(0, xx2 - xx1)
            h   = np.maximum(0, yy2 - yy1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
            order = order[np.where(iou <= self.iou_thresh)[0] + 1]
        return keep

    def finalize(self):
        pass
