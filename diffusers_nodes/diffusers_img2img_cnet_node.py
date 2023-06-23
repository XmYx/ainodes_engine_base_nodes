import secrets
from typing import Union, Optional, Dict, Any, Tuple, List

import torch
from diffusers.models.controlnet import ControlNetOutput

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, torch_gc, \
    get_torch_device
from ainodes_frontend import singleton as gs

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetImg2ImgPipeline

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import multiForward, diffusers_models, \
    diffusers_indexed, scheduler_type_values, get_scheduler, SchedulerType

#MANDATORY
OP_NODE_DIFF_IMG2IMG_PIPELINE = get_next_opcode()

#NODE WIDGET
class DiffusersImg2ImgPipeLineWidget(QDMNodeContentWidget):
    def initUI(self):
        self.models = self.create_combo_box([item["name"] for item in diffusers_models], "Model")
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")

        self.reload = self.create_check_box("Reload")
        self.prompt = self.create_text_edit("Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=512, step=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=512, step=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.strength = self.create_double_spin_box("Strength", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")
        self.create_main_layout(grid=1)



#NODE CLASS
@register_node(OP_NODE_DIFF_IMG2IMG_PIPELINE)
class DiffusersImg2ImgPipeLineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - "
    op_code = OP_NODE_DIFF_IMG2IMG_PIPELINE
    op_title = "Diffusers - StableDiffusionImg2ImgControlNet"
    content_label_objname = "diffusers_img2imgcnet_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersImg2ImgPipeLineWidget
    dim = (340, 700)
    output_data_ports = [0]
    exec_port = 2

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[5,6,1])
        self.content.setMinimumHeight(600)

        self.content.setMaximumHeight(600)
        self.pipe = None

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        data = self.getInputData(0)
        images = self.getInputData(1)

        image = pixmap_to_tensor(images[0])

        control_images = []
        controlnets = []
        cnet_scales = []

        def replace_forward_with(control_net_model, new_forward):
            def forward_with_self(*args, **kwargs):
                return new_forward(control_net_model, *args, **kwargs)

            return forward_with_self

        if data:
            device = gs.device

            if "control_diff" in data:
                for control in data["control_diff"]:
                    cnet = ControlNetModel.from_pretrained(control["name"], torch_dtype=torch.float16).to(device)
                    cnet.start_control = control["start"]
                    cnet.stop_control = control["stop"]
                    controlnets.append(cnet)
                    control_images.append(control["image"])
                    cnet_scales.append(control["scale"])
        else:
            data = {}
        reload = None

        guess_mode = False
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]
        #if reload or not self.pipe:

        if self.content.reload.isChecked() or not self.pipe:
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                model_name, controlnet=controlnets, torch_dtype=torch.float16, safety_checker=None,
            ).to(gs.device)



        if hasattr(self.pipe, "controlnet"):
            self.pipe.controlnet.forward = replace_forward_with(self.pipe.controlnet, multiForward)
        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        self.pipe = get_scheduler(self.pipe, scheduler_enum)

        #self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        prompt = self.content.prompt.toPlainText()
        height = self.content.height_val.value()
        width = self.content.width_val.value()
        num_inference_steps = self.content.steps.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        eta = self.content.eta.value()
        strength = self.content.strength.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())

        generator = torch.Generator(gs.device).manual_seed(seed)
        latents = None

        image = self.pipe(prompt = prompt,
                    image = image,
                    control_image = control_images,
                    height = height,
                    width = width,
                    num_inference_steps = num_inference_steps,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    eta = eta,
                    strength = strength,
                    generator = generator,
                    latents = latents,
                    controlnet_conditioning_scale = cnet_scales,
                    guess_mode = guess_mode).images[0]

        pixmap = tensor_image_to_pixmap(image)
        print(pixmap)
        return [pixmap]

    def onWorkerFinished(self, result):
        self.busy = False
        print(result)

        self.setOutput(0, result)
        self.executeChild(2)

    def remove(self):
        print("REMOVING", self)
        if self.pipe:
            self.pipe.to("cpu")
            del self.pipe
            torch_gc()
        super().remove()

