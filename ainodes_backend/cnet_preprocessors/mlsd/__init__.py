import cv2
import numpy as np
import torch
import os

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.cnet_preprocessors.mlsd.models.mbv2_mlsd_large import MobileV2_MLSD_Large
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.cnet_preprocessors.mlsd.utils import pred_lines

annotator_ckpts_path = "models/annotators"


remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"


class MLSDdetector:
    def __init__(self):
        model_path = os.path.join(annotator_ckpts_path, "mlsd_large_512_fp32.pth")
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        self.model = model.cuda().eval()

    def __call__(self, input_image, thr_v, thr_d):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        with torch.no_grad():
            lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        return img_output[:, :, 0]
