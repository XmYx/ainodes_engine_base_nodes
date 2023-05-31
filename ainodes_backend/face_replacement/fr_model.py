import os

import numpy as np
import requests
from PIL import Image
import onnxruntime
import insightface

from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import poorman_wget


class FaceReplacementModel():

    def __init__(self):
        super().__init__()
        use_gpu = True

        providers = onnxruntime.get_available_providers()
        #providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
        """providers = [("TensorrtExecutionProvider", {'trt_engine_cache_path': '.',
                                                    'trt_int8_use_native_calibration_table': False,
                                                    'device_id': '0',
                                                    'trt_max_partition_iterations': '1000',
                                                    'trt_force_sequential_engine_build': False,
                                                    'trt_min_subgraph_size': True,
                                                    'trt_max_workspace_size': '1073741824',
                                                    'trt_fp16_enable': True,
                                                    'trt_int8_enable': False,
                                                    'trt_dla_enable': False,
                                                    'trt_dla_core': False,
                                                    'trt_dump_subgraphs': False,
                                                    'trt_engine_cache_enable': False,
                                                    'trt_engine_decryption_enable': False,
                                                    'trt_context_memory_sharing_enable': False,
                                                    'trt_layer_norm_fp32_fallback': False,
                                                    'trt_timing_cache_enable': False,
                                                    'trt_force_timing_cache_match': False,
                                                    'trt_detailed_build_log': False,
                                                    'trt_build_heuristics_enable': False,
                                                    'trt_sparsity_enable': False,
                                                    'trt_builder_optimization_level': '3',
                                                    'trt_auxiliary_streams': '-1'})]"""
        model_path = 'models/other/roopVideoFace_v10.onnx'
        if not os.path.isfile(model_path):
            import gdown
            url = '1JxYLMECet8pU3Jq7EEcCw5p5xnFWTg5I'
            gdown.download(id=url, output=model_path, quiet=False)


        self.face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        self.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        self.source_face_tensor = None

    def __call__(self, source_face_img, target_face_img, recalc=True):

        source_face = np.array(source_face_img).astype(np.uint8)
        #source_face_img_array = cv2.cvtColor(source_face, cv2.COLOR_RGB2BGR)  # Assuming the PIL image is RGB

        target_img = np.array(target_face_img).astype(np.uint8)
        #target_img_array = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)  # Assuming the PIL image is RGB

        face_tensor = self.get_face(target_img)
        if recalc or self.source_face_tensor == None:
            self.source_face_tensor = self.get_face(source_face)

        if face_tensor:

            result = self.face_swapper.get(target_img, face_tensor, self.source_face_tensor, paste_back=True)

            image = Image.fromarray(result)
            return image
        else:
            return target_face_img

    def get_face(self, frame):
        analysed = self.face_analyser.get(frame)
        try:
            return sorted(analysed, key=lambda x: x.bbox[0])[0]
        except IndexError:
            return None

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print("File downloaded successfully!")
    else:
        print("Failed to download the file.")