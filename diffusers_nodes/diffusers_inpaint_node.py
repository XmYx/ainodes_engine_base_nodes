import secrets
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageDraw
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline, AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline

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
        change_pipe = False


        if not self.pipe or change_pipe:
            self.pipe = self.getInputData(0)
            if not self.pipe or change_pipe:
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-2-inpainting",
                            torch_dtype=torch.float16)
                # self.pipe = AutoPipelineForInpainting.from_pretrained("SG161222/RealVisXL_V2.0",
                #                                                  torch_dtype=torch.float16, variant="fp16").to("cuda")
        masks = self.getInputData(1)[0]
        images = self.getInputData(2)[0]


        # print(masks.shape, images.shape)


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

        img = tensor2pil(images).convert("RGB")
        mask_img = tensor2pil(masks).convert("RGB")
        mask_img = dilate_mask(mask_img, 12)


        image = self.pipe(prompt = prompt,
                    image = img,
                    mask_image = mask_img,
                    width=img.size[0],
                    height=img.size[1],
                    num_inference_steps = num_inference_steps,
                    strength=strength,
                    guidance_scale = guidance_scale,
                    negative_prompt = negative_prompt,
                    eta = eta,
                    generator = generator).images[0]

        self.pipe.watermark = None
        # image = self.pipe(prompt=prompt,
        #                   image=img,
        #                   mask_image=mask_img,
        #                   num_inference_steps = num_inference_steps,
        #                   width=img.size[0],
        #                   height=img.size[1],
        #                   strength=strength,
        #                   generator=generator
        #
        #
        #                   ).images[0]
        # image = crop_inner_image(
        #     img, img.size[0], img.size[1]
        # )
        #
        # image = crop_fethear_ellipse(
        #     img,
        #     30,
        #     int(10 / 3 // 2),
        #     int(10 / 3 // 2),
        # )
        #
        tensor = pil2tensor(image)

        self.pipe.to("cpu")
        torch_gc()

        return [self.pipe, tensor]

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


def crop_inner_image(outpainted_img, width_offset, height_offset):
    width, height = outpainted_img.size

    center_x, center_y = int(width / 2), int(height / 2)

    # Crop the image to the center
    cropped_img = outpainted_img.crop(
        (
            center_x - width_offset,
            center_y - height_offset,
            center_x + width_offset,
            center_y + height_offset,
        )
    )
    prev_step_img = cropped_img.resize((width, height), resample=Image.LANCZOS)
    # resized_img = resized_img.filter(ImageFilter.SHARPEN)

    return prev_step_img

def crop_fethear_ellipse(image, feather_margin=30, width_offset=0, height_offset=0):
    # Create a blank mask image with the same size as the original image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Calculate the ellipse's bounding box
    ellipse_box = (
        width_offset,
        height_offset,
        image.width - width_offset,
        image.height - height_offset,
    )

    # Draw the ellipse on the mask
    draw.ellipse(ellipse_box, fill=255)

    # Apply the mask to the original image
    result = Image.new("RGBA", image.size)
    result.paste(image, mask=mask)

    # Crop the resulting image to the ellipse's bounding box
    cropped_image = result.crop(ellipse_box)

    # Create a new mask image with a black background (0)
    mask = Image.new("L", cropped_image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw an ellipse on the mask image
    draw.ellipse(
        (
            0 + feather_margin,
            0 + feather_margin,
            cropped_image.width - feather_margin,
            cropped_image.height - feather_margin,
        ),
        fill=255,
        outline=0,
    )

    # Apply a Gaussian blur to the mask image
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_margin / 2))
    cropped_image.putalpha(mask)
    res = Image.new(cropped_image.mode, (image.width, image.height))
    paste_pos = (
        int((res.width - cropped_image.width) / 2),
        int((res.height - cropped_image.height) / 2),
    )
    res.paste(cropped_image, paste_pos)

    return res


def erode_mask(mask_img, erosion_size=12):
    # Convert the PIL Image to a NumPy array
    mask_array = np.array(mask_img)

    # Create the erosion kernel
    kernel = np.ones((erosion_size, erosion_size), np.uint8)

    # Erode the mask
    eroded_mask_array = cv2.erode(mask_array, kernel)

    # Convert back to a PIL Image
    eroded_mask_img = Image.fromarray(eroded_mask_array)

    return eroded_mask_img


def dilate_mask(mask_img, dilation_size=12):
    # Convert the PIL Image to a NumPy array
    mask_array = np.array(mask_img)

    # Create the dilation kernel
    kernel = np.ones((dilation_size, dilation_size), np.uint8)

    # Dilate the mask
    dilated_mask_array = cv2.dilate(mask_array, kernel)

    # Convert back to a PIL Image
    dilated_mask_img = Image.fromarray(dilated_mask_array)

    return dilated_mask_img