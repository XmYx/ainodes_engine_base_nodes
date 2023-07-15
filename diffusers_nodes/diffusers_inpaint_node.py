import secrets
import subprocess

import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, torch_gc, tensor2pil
from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import scheduler_type_values, SchedulerType, \
    get_scheduler
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_INPAINT = get_next_opcode()

def dont_apply_watermark(images: torch.FloatTensor):
    #self.pipe.watermarker.apply_watermark = dont_apply_watermark

    return images

class DiffInpaintWidget(QDMNodeContentWidget):
    def initUI(self):

        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")

        self.prompt = self.create_text_edit("Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt")
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")
        self.strength = self.create_double_spin_box("Strength", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)

        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_INPAINT)
class DiffInpaintNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Inpaint"
    op_code = OP_NODE_DIFF_INPAINT
    op_title = "Diffusers InPaint"
    content_label_objname = "sd_diff_inpaint_node"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = DiffInpaintWidget
    dim = (340, 800)
    output_data_ports = [0, 1]
    exec_port = 2

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,5,5,1], outputs=[4,5,1])
        self.path = "nichijoufan777/stable-diffusion-xl-base-0.9"
        self.pipe = None
    def evalImplementation_thread(self, index=0):
        from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, torch_gc, tensor2pil

        if not self.pipe:
            self.pipe = self.getInputData(0)
            if not self.pipe:
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-2-inpainting",
                            torch_dtype=torch.float16)

        masks = self.getInputData(1)
        images = self.getInputData(2)


        self.pipe.to("cuda")
        prompt = self.content.prompt.toPlainText()
        num_inference_steps = self.content.steps.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        eta = self.content.eta.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = None
        strength = self.content.strength.value()

        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        self.pipe = get_scheduler(self.pipe, scheduler_enum)


        image = self.pipe(prompt = prompt,
                    image = tensor2pil(images[0]),
                    mask_image = tensor2pil(masks[0]),
                    num_inference_steps = num_inference_steps,
                    strength=strength,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    eta = eta,
                    generator = generator).images[0]

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
