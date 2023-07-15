import secrets
import subprocess

import torch
from diffusers import DiffusionPipeline

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, torch_gc, tensor2pil
from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import scheduler_type_values, SchedulerType, \
    get_scheduler
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFFREFINE_XL = get_next_opcode()

def dont_apply_watermark(images: torch.FloatTensor):

    return images


class DiffRefineXLWidget(QDMNodeContentWidget):
    def initUI(self):

        self.token = self.create_line_edit("Token")
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")

        self.prompt = self.create_text_edit("Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt")
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")
        self.strength = self.create_double_spin_box("Strength", min_val=0.01, max_val=1.00, default_val=1.0, step=0.01)
        self.score = self.create_double_spin_box("Aesthetic score", min_val=0.01, max_val=25.00, default_val=6.0, step=0.01)
        self.n_score = self.create_double_spin_box("Negative Aesthetic score", min_val=0.01, max_val=25.00, default_val=2.5, step=0.01)


        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFFREFINE_XL)
class DiffRefineXLNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "XL"
    op_code = OP_NODE_DIFFREFINE_XL
    op_title = "Diffusers XL - Refine"
    content_label_objname = "sd_xlrefine_node"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = DiffRefineXLWidget
    dim = (340, 800)
    output_data_ports = [0, 1]
    exec_port = 2

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,5,2,1], outputs=[4,5,1])
        self.path = "stabilityai/stable-diffusion-xl-refiner-0.9"
        self.pipe = None
    def evalImplementation_thread(self, index=0):


        #subprocess.call(["pip", "install", "git+https://github.com/patrickvonplaten/invisible-watermark.git@remove_onnxruntime_depedency"])


        image = self.getInputData(1)

        #assert image is not None, "Image has to be a valid tensor"

        if image is not None:
            image = tensor2pil(image[0])
        latent = self.getInputData(2)
        token = self.content.token.text()

        assert token != "", "Token must be a valid HF Token string"

        if not self.pipe:
            self.pipe = self.getInputData(0)
            if not self.pipe:
                self.pipe = DiffusionPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False, use_auth_token=token)
        self.pipe.watermark.apply_watermark = dont_apply_watermark

        self.pipe.to("cuda")

        prompt = self.content.prompt.toPlainText()
        score = self.content.score.value()
        n_score = self.content.n_score.value()
        num_inference_steps = self.content.steps.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        eta = self.content.eta.value()
        strength = self.content.strength.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = None
        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        self.pipe = get_scheduler(self.pipe, scheduler_enum)
        if image is not None:
            input_image = image
        else:
            input_image = latent[0]
        image = self.pipe(  prompt = prompt,
                            image=input_image,
                            num_inference_steps = num_inference_steps,
                            guidance_scale = guidance_scale,
                            negative_prompt = negative_prompt,
                            eta = eta,
                            generator = generator,
                            latents = latents,
                            strength=strength,
                            aesthetic_score=score,
                            negative_aesthetic_score=n_score
                            ).images[0]

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
