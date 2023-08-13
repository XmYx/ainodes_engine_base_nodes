import secrets
import subprocess

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, torch_gc
from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import scheduler_type_values, SchedulerType, \
    get_scheduler
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFFBASE_XL = get_next_opcode()

def dont_apply_watermark(images: torch.FloatTensor):

    return images

class DiffBaseXLWidget(QDMNodeContentWidget):
    def initUI(self):

        self.token = self.create_line_edit("Token")
        self.return_type = self.create_combo_box(["latent", "pil"], "Return Type")
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")

        self.prompt = self.create_text_edit("Linguistic Prompt", placeholder="Linguistic Prompt")
        self.prompt_2 = self.create_text_edit("Classic Prompt", placeholder="Classic Prompt")
        self.n_prompt = self.create_text_edit("Linguistic Negative Prompt", placeholder="Linguistic Negative Prompt")
        self.n_prompt_2 = self.create_text_edit("Classic Negative Prompt", placeholder="Classic Negative Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=1024, step=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=1024, step=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.denoising_end = self.create_double_spin_box("Denoising End", min_val=0, max_val=1.0, default_val=1.0, step=0.01)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")

        self.guidance_rescale = self.create_double_spin_box("Guidance Rescale", min_val=0.00, max_val=25.00, default_val=0.0, step=0.01)

        self.original_width = self.create_spin_box("Orig Width", min_val=64, max_val=4096, default_val=1024, step=64)
        self.original_height = self.create_spin_box("Orig height", min_val=64, max_val=4096, default_val=1024, step=64)
        self.target_width = self.create_spin_box("Target width", min_val=64, max_val=4096, default_val=1024, step=64)
        self.target_height = self.create_spin_box("Target height", min_val=64, max_val=4096, default_val=1024, step=64)

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
    dim = (340, 1152)
    output_data_ports = [0,1,2,3]
    exec_port = 4
    use_gpu = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,6,1], outputs=[4,5,2,6,1])
        self.path = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = None
    def evalImplementation_thread(self, index=0):
        tensor = None
        return_latent = None
        token = self.content.token.text()

        #assert token != "", "Token must be a valid HF Token string"
        #subprocess.call(["pip", "install", "git+https://github.com/patrickvonplaten/invisible-watermark.git@remove_onnxruntime_depedency"])

        # if not self.pipe:
        #     self.pipe = self.getInputData(0)
        #     if not self.pipe:
        #         self.pipe = StableDiffusionXLPipeline.from_pretrained(self.path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        #         self.pipe.disable_attention_slicing()
        pipe = self.getInputData(0)
        gpu_id = self.content.gpu_id.currentText()
        from ainodes_frontend import singleton as gs
        pipe.to(f"{gs.device.type}:{gpu_id}")
        pipe.watermark.apply_watermark = dont_apply_watermark
        prompt = self.content.prompt.toPlainText()
        prompt_2 = self.content.prompt_2.toPlainText()
        prompt_2 = prompt if prompt_2 == "" else prompt_2
        height = self.content.height_val.value()
        width = self.content.width_val.value()
        num_inference_steps = self.content.steps.value()
        denoising_end = self.content.denoising_end.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        negative_prompt_2 = self.content.n_prompt_2.toPlainText()
        negative_prompt_2 = negative_prompt if negative_prompt_2 == "" else negative_prompt_2

        data = self.getInputData(1)
        if data is not None:
            if "prompt" in data:
                prompt = data["prompt"]
            if "prompt_2" in data:
                prompt_2 = data["prompt_2"]
            if "negative_prompt" in data:
                negative_prompt = data["negative_prompt"]
            if "negative_prompt_2" in data:
                negative_prompt_2 = data["negative_prompt_2"]

        eta = self.content.eta.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        generator = torch.Generator("cuda").manual_seed(seed)
        latents = None
        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        pipe = get_scheduler(pipe, scheduler_enum)

        guidance_rescale = self.content.guidance_rescale.value()
        original_size = (self.content.original_width.value(), self.content.original_height.value())
        target_size = (self.content.target_width.value(), self.content.target_height.value())

        return_type = self.content.return_type.currentText()

        image = pipe(  prompt = prompt,
                            prompt_2=prompt_2,
                            height = height,
                            width = width,
                            num_inference_steps = num_inference_steps,
                            denoising_end=denoising_end,
                            guidance_scale = guidance_scale,
                            negative_prompt = negative_prompt,
                            negative_prompt_2=negative_prompt_2,
                            eta = eta,
                            generator = generator,
                            latents = latents,
                            guidance_rescale = guidance_rescale,
                            original_size = original_size,
                            target_size = target_size,
                            output_type = return_type
                            ).images[0]

        data = {
            "prompt":prompt,
            "prompt_2":prompt_2,
            "negative_prompt":negative_prompt,
            "negative_prompt_2":negative_prompt,
            "denoising_end":denoising_end
        }
        pipe.to("cpu")
        torch_gc()
        if return_type == "pil":
            tensor = pil2tensor(image)
            return [pipe, [tensor], [None], data]
        else:
            return [pipe, [None], [image], data]


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
