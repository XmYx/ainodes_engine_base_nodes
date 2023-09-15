import os
import secrets

import torch

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor2pil, pil2tensor
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.ip_adapter import IPAdapterXL
from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import diffusers_models, diffusers_indexed, \
    scheduler_type_values, SchedulerType, get_scheduler_class
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_DIFF_SDIPADAPTER = get_next_opcode()
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline,
                       StableDiffusionXLImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionXLControlNetPipeline)


class DiffSDIPAdapterWidget(QDMNodeContentWidget):
    def initUI(self):
        checkpoint_folder = "models/ip_adapter"
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.dropdown = self.create_combo_box(checkpoint_files, "Model:")
        if checkpoint_files == []:
            self.dropdown.addItem("Please place a model in models/checkpoints")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/models/checkpoints,")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.prompt = self.create_text_edit("Prompt", placeholder="Prompt or Negative Prompt (use 2x Conditioning Nodes for Stable Diffusion),\n"
                                                                  "and connect them to a K Sampler.\n"
                                                                  "If you want to control your resolution,\n"
                                                                  "or use an init image, use an Empty Latent Node.")

        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_SDIPADAPTER)
class DiffSDIpNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers IP Adapter"
    op_code = OP_NODE_DIFF_SDIPADAPTER
    op_title = "Diffusers - IP Adapter"
    content_label_objname = "diff_ipadapter_node"
    category = "aiNodes Base/NEW_CATEGORY"
    NodeContent_class = DiffSDIPAdapterWidget
    dim = (340, 460)
    output_data_ports = [0]
    custom_input_socket_name = ["PIPE", "IMAGE", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,5,1], outputs=[5,1])
        self.pipe = None
    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)
        image = self.getInputData(1)

        if image is not None:
            image = tensor2pil(image[0])

        image_encoder_path = "models/ip_adapter/image_encoder"
        ip_ckpt = "models/ip_adapter/ip-adapter_sdxl.bin"
        device = gs.device
        pipe = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
        if pipe.device.type != "cuda":
            pipe.to("cuda")

        seed = secrets.randbelow(999999999999)
        images = pipe.generate(pil_image=image, num_samples=1, num_inference_steps=30, seed=seed   )[0]
        image = pil2tensor(images)
        del pipe
        return [[image]]


    def remove(self):
        try:
            if self.pipe != None:
                self.pipe.to("cpu")
                del self.pipe
                self.pipe = None
        except:
            self.pipe = None
        super().remove()