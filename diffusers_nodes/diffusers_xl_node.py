import secrets
import subprocess

import torch
from diffusers import DiffusionPipeline

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, torch_gc
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFFBASE_XL = get_next_opcode()

def dont_apply_watermark(images: torch.FloatTensor):
    #self.pipe.watermarker.apply_watermark = dont_apply_watermark

    return images

class DiffBaseXLWidget(QDMNodeContentWidget):
    def initUI(self):

        self.token = self.create_line_edit("Token")

        self.prompt = self.create_text_edit("Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=512, step=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=512, step=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")


        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFFBASE_XL)
class DiffBaseXLNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "XL"
    op_code = OP_NODE_DIFFBASE_XL
    op_title = "Diffusers XL - Base"
    content_label_objname = "sd_xlbase_node"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = DiffBaseXLWidget
    dim = (340, 800)
    output_data_ports = [0, 1]
    exec_port = 2

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,5,1])
        self.path = "nichijoufan777/stable-diffusion-xl-base-0.9"
        self.pipe = None
    def evalImplementation_thread(self, index=0):

        token = self.content.token.text()

        assert token != "", "Token must be a valid HF Token string"
        subprocess.call(["pip", "install", "git+https://github.com/patrickvonplaten/invisible-watermark.git@remove_onnxruntime_depedency"])

        if not self.pipe:
            self.pipe = self.getInputData(0)
            if not self.pipe:
                self.pipe = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False, use_auth_token=token)

        self.pipe.to("cuda")
        self.pipe.watermark.apply_watermark = dont_apply_watermark
        prompt = self.content.prompt.toPlainText()
        height = self.content.height_val.value()
        width = self.content.width_val.value()
        num_inference_steps = self.content.steps.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        eta = self.content.eta.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = None

        image = self.pipe(prompt = prompt,
                    height = height,
                    width = width,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    eta = eta,
                    generator = generator,
                    latents = latents).images[0]

        tensor = pil2tensor(image)

        self.pipe.to("cpu")
        torch_gc()

        return [self.pipe, [tensor]]

    def remove(self):
        if self.pipe is not None:
            try:
                self.pipe.to("cpu")
                del self.pipe
                self.pipe = None

                torch_gc()
            except:
                pass
        super().remove()
