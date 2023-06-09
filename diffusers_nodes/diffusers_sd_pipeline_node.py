import secrets

import torch

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap, pixmap_to_pil_image, torch_gc, \
    get_torch_device
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, \
    StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from custom_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import multiForward, diffusers_models, \
    diffusers_indexed, scheduler_type_values, get_scheduler, SchedulerType

#MANDATORY
OP_NODE_DIFF_PIPELINE = get_next_opcode()

#NODE WIDGET
class DiffusersPipeLineWidget(QDMNodeContentWidget):
    def initUI(self):
        self.models = self.create_combo_box([item["name"] for item in diffusers_models], "Model")
        self.reload = self.create_check_box("Reload")
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")
        self.prompt = self.create_text_edit("Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=512, step_value=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=512, step_value=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step_value=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")
        self.create_main_layout(grid=1)

#NODE CLASS
@register_node(OP_NODE_DIFF_PIPELINE)
class DiffusersPipeLineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - "
    op_code = OP_NODE_DIFF_PIPELINE
    op_title = "Diffusers - StableDiffusionControlNet"
    content_label_objname = "diffusers_pipeline_node"
    category = "Diffusers"
    NodeContent_class = DiffusersPipeLineWidget
    dim = (340, 700)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[5,1])
        self.content.setMinimumHeight(600)

        self.content.setMaximumHeight(600)
        self.pipe = None
        self.control_params = []


    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        data = self.getInputData(0)
        control_images = []
        controlnets = []
        cnet_scales = []

        def replace_forward_with(control_net_model, new_forward):
            def forward_with_self(*args, **kwargs):
                return new_forward(control_net_model, *args, **kwargs)

            return forward_with_self

        reload = True if self.content.reload.isChecked() or self.pipe == None else None


        guess_mode = False if control_images else True
        device = get_torch_device()
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]

        if data:
            if "control_diff" in data:
                control_params = []
                x = 0
                for control in data["control_diff"]:
                    control_params.append(control["name"])
                    if len(self.control_params) != 0:
                        if len(control_params) >= x:
                            if not reload and control_params[x] != self.control_params[x]:
                                reload = True
                        else:
                            reload = True
                    if reload:
                        cnet = ControlNetModel.from_pretrained(control["name"], torch_dtype=torch.float16).to(get_torch_device())
                    else:
                        cnet = self.pipe.controlnet.nets[x]
                    cnet.start_control = control["start"]
                    cnet.stop_control = control["stop"]
                    control_images.append(control["image"])
                    cnet_scales.append(control["scale"])
                    if reload:
                        controlnets.append(cnet)

                    x += 1
                self.control_params = control_params


        do_hijack = None
        if len(controlnets) > 0:
            diffusion_class = StableDiffusionControlNetPipeline
            do_hijack = True
        else:
            diffusion_class = StableDiffusionPipeline

        if self.content.reload.isChecked() or self.pipe == None:
            self.pipe = diffusion_class.from_pretrained(
                model_name, controlnet=controlnets, torch_dtype=torch.float16, safety_checker=None
            ).to(device)
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            print(self.pipe.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
            self.pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
            print(
                self.pipe.unet.conv_out.state_dict()["weight"].stride()
            )  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works
            if do_hijack:
                self.pipe.controlnet.forward = replace_forward_with(self.pipe.controlnet, multiForward)
        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        self.pipe = get_scheduler(self.pipe, scheduler_enum)
        prompt = self.content.prompt.toPlainText()
        height = self.content.height_val.value()
        width = self.content.width_val.value()
        num_inference_steps = self.content.steps.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        eta = self.content.eta.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        generator = torch.Generator(device).manual_seed(seed)
        latents = None
        if isinstance(self.pipe, StableDiffusionControlNetPipeline):
            image = self.pipe(prompt = prompt,
                        image = control_images,
                        height = height,
                        width = width,
                        num_inference_steps = num_inference_steps,
                        guidance_scale = guidance_scale,
                        negative_prompt = negative_prompt,
                        eta = eta,
                        generator = generator,
                        latents = latents,
                        controlnet_conditioning_scale = cnet_scales,
                        guess_mode = guess_mode).images[0]
        else:
            image = self.pipe(prompt = prompt,
                        height = height,
                        width = width,
                        num_inference_steps = num_inference_steps,
                        guidance_scale = guidance_scale,
                        negative_prompt = negative_prompt,
                        eta = eta,
                        generator = generator,
                        latents = latents).images[0]

        torch_gc()
        return [[pil_image_to_pixmap(image)]]



    def remove(self):
        print("REMOVING", self)
        if self.pipe:
            self.pipe.to("cpu")
            del self.pipe
            torch_gc()

        super().remove()

