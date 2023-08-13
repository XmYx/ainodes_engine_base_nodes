import secrets
from typing import Union, List, Optional, Dict, Any, Tuple

import diffusers
import qrcode
import torch
from PIL import Image
from diffusers.models.controlnet import ControlNetOutput

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, torch_gc, get_torch_device, pil2tensor
from ainodes_frontend import singleton as gs
import os

#from .diffusers_restart_pipeline import StableDiffusionPipeline

#diffusers.StableDiffusionPipeline = StableDiffusionPipeline

from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,
                       StableDiffusionControlNetImg2ImgPipeline, DDIMScheduler, StableDiffusionPipeline, AutoencoderKL)
from diffusers.models.attention_processor import AttnProcessor2_0
from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import (multiForward, diffusers_models,
                                                                                  diffusers_indexed, scheduler_type_values,
                                                                                  get_scheduler, SchedulerType)

#MANDATORY
OP_NODE_DIFF_PIPELINE = get_next_opcode()

#NODE WIDGET
class DiffusersPipeLineWidget(QDMNodeContentWidget):
    def initUI(self):
        self.models = self.create_combo_box([item["name"] for item in diffusers_models], "Model")

        lora_folder = gs.loras
        lora_files = [f for f in os.listdir(lora_folder) if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth'))]
        if lora_files == []:
            self.dropdown.addItem("Please place a lora in models/loras")
            print(f"LORA LOADER NODE: No model file found at {os.getcwd()}/models/loras,")
            print(f"LORA LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.dropdown = self.create_combo_box(lora_files, "Lora")
        self.use_test = self.create_check_box("use_test_model")

        self.reload = self.create_check_box("Reload")
        self.load_lora = self.create_check_box("Load LORA")
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")
        self.prompt = self.create_text_edit("Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=512, step=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=512, step=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
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
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersPipeLineWidget
    dim = (340, 700)
    output_data_ports = [0, 1]
    exec_port = 2

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,1], outputs=[5,6,1])
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
        starts = []
        stops = []
        scales = []

        prompt = self.content.prompt.toPlainText()
        height = self.content.height_val.value()
        width = self.content.width_val.value()
        num_inference_steps = self.content.steps.value()
        guidance_scale = self.content.scale.value()
        negative_prompt = self.content.n_prompt.toPlainText()
        eta = self.content.eta.value()
        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())



        if data:

            if "prompt" in data:
                prompt = data["prompt"]
            if "n_prompt" in data:
                negative_prompt = data["n_prompt"]
            if "guidance_scale" in data:
                guidance_scale = data["guidance_scale"]
            if "seed" in data:
                seed = data["seed"]
            if "steps" in data:
                num_inference_steps = data["steps"]
            if "width" in data:
                width = data["width"]
            if "height" in data:
                height = data["height"]

            if "control_diff" in data:
                control_params = []
                x = 0

                if len(data["control_diff"]) != self.control_params:
                    reload = True

                for control in data["control_diff"]:
                    control_params.append(control["name"])
                    # if len(control_params) >= x:
                    #     if not reload and control_params[x] != self.control_params[x]:
                    #         reload = True
                    # else:
                    #     reload = True
                    if reload:
                        cnet = ControlNetModel.from_pretrained(control["name"], torch_dtype=torch.float16).to(get_torch_device())
                    else:
                        cnet = self.pipe.controlnet.nets[x]
                    starts.append(control["start"])
                    stops.append(control["stop"])
                    scales.append(control["scale"])
                    # cnet.start_control = control["start"]
                    # cnet.stop_control = control["stop"]
                    control_images.append(control["image"])
                    cnet_scales.append(float(control["scale"]))
                    if reload:
                        controlnets.append(cnet)

                    x += 1
                self.control_params = control_params
        generator = torch.Generator(device).manual_seed(seed)
        latents = None


        do_hijack = False
        if len(controlnets) > 0:
            diffusion_class = StableDiffusionControlNetPipeline
            do_hijack = False
        else:
            diffusion_class = StableDiffusionPipeline

        if self.content.reload.isChecked() or self.pipe == None or reload:
            if len(controlnets) == 1:
                controlnets = controlnets[0]
            local_test = self.content.use_test.isChecked()
            if local_test:
                model_name = "C:/Users/mix/Documents/GitHub/ainodes-engine/src/sd-scripts/segmind/resume"
            self.pipe = diffusion_class.from_pretrained(
                model_name, controlnet=controlnets, torch_dtype=torch.float16, safety_checker=None, local_files_only=local_test).to(device)

            #self.load_lora()

            # self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            # print(self.pipe.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
            # self.pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
            # print(
            #     self.pipe.unet.conv_out.state_dict()["weight"].stride()
            # )  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works
        if do_hijack:
            if hasattr(self.pipe, "controlnet"):
                self.pipe.controlnet.forward = replace_forward_with(self.pipe.controlnet, multiForward)
        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        self.pipe = get_scheduler(self.pipe, scheduler_enum)

        # self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.scheduler.use_sigma = True



        if isinstance(self.pipe, StableDiffusionControlNetPipeline):

            print(cnet_scales, starts, stops)
            if len(cnet_scales) == 1 or len(starts) == 1 or len(stops) == 1:
                cnet_scales = cnet_scales[0]
                starts = starts[0]
                stops = stops[0]


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
                        guess_mode = False,
                        control_guidance_start = starts,
                        control_guidance_end = stops).images[0]
        else:
            if self.content.load_lora.isChecked():
                print("loading lora")

                self.pipe.load_lora_weights("models/loras", weight_name=self.content.dropdown.currentText())

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

        data = {
            "prompt":prompt,
            "n_prompt":negative_prompt,
            "steps":num_inference_steps,
            "seed":seed,
            "guidance_scale":guidance_scale,
            "width":width,
            "height":height
        }

        return [[pil2tensor(image)], data]

    def load_lora(self):
        from .diffusers_lora_loader import install_lora_hook
        install_lora_hook(self.pipe)
        lora1 = self.pipe.apply_lora("models/loras/add_detail.safetensors", alpha=0.8)


    def remove(self):
        print("REMOVING", self)
        if self.pipe:
            self.pipe.to("cpu")
            del self.pipe
            torch_gc()

        super().remove()




class SegmindQrGenerator():
    
    def __init__(self):
        super().__init__()
        
        cnet_1 = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16).to("cuda")
        cnet_2 = ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16).to("cuda")

        cnet_1.start_control = 0
        cnet_1.stop_control = 100
        cnet_1.conditioning_scale = 1.0

        cnet_2.start_control = 23
        cnet_2.stop_control = 100
        cnet_2.conditioning_scale = 1.0

        
        controlnets = [cnet_1, cnet_2]

        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "dreamlike-photoreal-2.0", torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")
        
        self.img2img_pipe = StableDiffusionControlNetImg2ImgPipeline(vae = self.txt2img_pipe.vae,
                                                    text_encoder = self.txt2img_pipe.text_encoder,
                                                    tokenizer = self.txt2img_pipe.tokenizer,
                                                    unet = self.txt2img_pipe.unet,
                                                    scheduler = self.txt2img_pipe.scheduler,
                                                    controlnet = controlnets,
                                                    safety_checker= None,
                                                    feature_extractor= None,
                                                    requires_safety_checker=False)

        self.img2img_pipe.controlnet.forward = replace_forward_with(self.img2img_pipe.controlnet, multiForward)

    def __call__(self,
                 prompt="",
                 qr_string="",
                 qrq=1.0,
                 steps=25):


        qr_image = create_qr_code(qr_string)

        image = self.txt2img_pipe(prompt=prompt, num_inference_steps=steps, width=768, height=768).images[0]


        strength = 1.0 * qrq
        guidance_scale = 6.5 / qrq

        self.img2img_pipe.controlnet.cnets[1].start_control = 20 - (-qrq * 5)
        self.img2img_pipe.controlnet.cnets[1].conditioning_scale = 1.0 - (-qrq)



        control_images = [image, qr_image]

        image = self.img2img_pipe(prompt, image, control_images, height=768, width=768, strength=strength, num_inference_steps=steps, guidance_scale=guidance_scale)

        return image


def replace_forward_with(control_net_model, new_forward):
    def forward_with_self(*args, **kwargs):
        return new_forward(control_net_model, *args, **kwargs)
    return forward_with_self

def create_qr_code(data, version=40, box=1, border=2, fit=True):
    # Create qr code instance
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box,
        border=border,
    )

    # Add data to qr code
    qr.add_data(data)

    qr.make(fit=fit)

    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white")

    # Resize the image to 512x512
    resized_img = img.resize((768,768), Image.ANTIALIAS)

    return resized_img

def multiForward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    controlnet_cond: List[torch.tensor],
    conditioning_scale: List[float],
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guess_mode: bool = False,
    return_dict: bool = True,
) -> Union[ControlNetOutput, Tuple]:

    mid_block_res_sample = None
    down_block_res_samples = None

    for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):

        percentage = 100 - (int(timestep) / 10)
        if hasattr(controlnet, "start_control"):
            start = controlnet.start_control
        else:
            start = 0
        if hasattr(controlnet, "stop_control"):
            stop = controlnet.stop_control
        else:
            stop = 100
        if hasattr(controlnet, "conditioning_scale"):
            scale = controlnet.conditioning_scale
        if start <= percentage <= stop:
            print("DOING CNET", percentage)
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                guess_mode,
                return_dict,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                if down_block_res_samples is not None:
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                    ]
                else:
                    down_block_res_samples = down_samples
                if mid_block_res_sample is not None:
                    mid_block_res_sample += mid_sample
                else:
                    mid_block_res_sample = mid_sample

    return down_block_res_samples, mid_block_res_sample

        
# genny = SegmindQrGenerator()
#
#
# image = genny(prompt="A beautiful butterfly",
#               qr_string="www.google.com")
#
#
# image.save("test_qr.png")
#
        
        
        