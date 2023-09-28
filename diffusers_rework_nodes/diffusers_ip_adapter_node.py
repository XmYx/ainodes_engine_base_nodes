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
        self.n_prompt = self.create_text_edit("Linguistic Negative Prompt", placeholder="Linguistic Negative Prompt")
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")

        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)

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
    #custom_input_socket_name = ["PIPE", "IMAGE", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,5,6,1], outputs=[5,1])
        self.pipe = None
    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)
        image = self.getInputData(1)
        data = self.getInputData(2)

        if image is not None:
            image = tensor2pil(image[0])
        download_ip_adapter_xl()
        image_encoder_path = "models/ip_adapter/models/image_encoder"
        ip_ckpt = "models/ip_adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
        device = gs.device
        if self.pipe == None:
            from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.ip_adapter.ip_adapter import IPAdapterPlusXL
            self.pipe = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
        if self.pipe.device.type != "cuda":
            self.pipe.to("cuda")
        image = image.resize((512, 512))
        args = {
            "pil_image": image,
            "num_samples": 1,
            "seed" : secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text()),
            "prompt":self.content.prompt.toPlainText(),
            "negative_prompt":self.content.n_prompt.toPlainText(),
            "guidance_scale":self.content.scale.value(),
            "num_inference_steps":self.content.steps.value(),
        }

        if data is not None:
            data.update(args)
        else:
            data = args

        data["prompt_2"] = None
        del data["prompt_2"]
        data["negative_prompt_2"] = None
        del data["negative_prompt_2"]

        image = self.pipe.generate(**data)[0]
        image = pil2tensor(image)
        #del pipe
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

from huggingface_hub import hf_hub_download

def download_ip_adapter_xl():

    ip_ckpt = "models/ip-adapter_sdxl_vit-h.bin"
    repo_id = "h94/IP-Adapter"
    target_dir = "models/ip_adapter"
    image_encoder_path = "models/ip_adapter/image_encoder"


    adapter_files = ["ip-adapter-plus_sdxl_vit-h.bin"]
    image_encoder_files = ["config.json", "model.safetensors"]

    os.makedirs(target_dir, exist_ok=True)

    for file in adapter_files:
        subfolder = "sdxl_models"
        if not os.path.isfile(os.path.join(target_dir, subfolder, file)):
            hf_hub_download(repo_id=repo_id,
                            filename=file,
                            subfolder=subfolder,
                            local_dir=target_dir,
                            local_dir_use_symlinks=False)
    for file in image_encoder_files:
        subfolder = "models/image_encoder"
        if not os.path.isfile(os.path.join(target_dir, subfolder, file)):
            hf_hub_download(repo_id=repo_id,
                            filename=file,
                            subfolder=subfolder,
                            local_dir=target_dir,
                            local_dir_use_symlinks=False)
