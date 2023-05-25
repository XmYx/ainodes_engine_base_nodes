import numpy as np
from PIL import Image
from einops import rearrange

from .ksampler_node import get_fixed_seed
from ..ainodes_backend import pil_image_to_pixmap, pixmap_to_pil_image

import torch
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


from kandinsky2 import get_kandinsky2

from ..image_nodes.output_node import ImagePreviewNode
from ..video_nodes.video_save_node import VideoOutputNode

OP_NODE_KANDINSKY = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

class KandinskyWidget(QDMNodeContentWidget):
    seed_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)
    text_signal = QtCore.Signal(str)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.task = self.create_combo_box(["TXT2IMG", "INPAINT"], "Task")
        self.prompt = self.create_text_edit("Prompt:")
        self.seed = self.create_line_edit("Seed:")
        self.steps = self.create_spin_box("Steps:", 1, 10000, 25)
        self.cfg_scale = self.create_spin_box("Guidance Scale:", 0, 1000, 4)
        self.w_param = self.create_spin_box("Width:", 64, 2048, 512, 64)
        self.h_param = self.create_spin_box("Height:", 64, 2048, 512, 64)
        self.strength = self.create_double_spin_box("Strength:", 0.00, 1.00, 0.01, 0.84)
        self.sampler = self.create_combo_box(["p_sampler", "ddim_sampler", "plms_sampler"], "Sampler:")
        self.prior_cf_scale = self.create_spin_box("Prior Scale:", 0, 1000, 4)
        self.prior_steps = self.create_spin_box("Prior Scale:", 0, 1000, 5)
        self.button = QtWidgets.QPushButton("Run")


@register_node(OP_NODE_KANDINSKY)
class KandinskyNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_KANDINSKY
    op_title = "Kandinsky"
    content_label_objname = "kandinsky_node"
    category = "Sampling"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[5,5,6,1], outputs=[5,1])
        self.content.button.clicked.connect(self.evalImplementation)

        # Create a worker object
    def initInnerClasses(self):
        self.content = KandinskyWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 550
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.seed = ""
        self.content.seed_signal.connect(self.setSeed)
        self.content.progress_signal.connect(self.setProgress)
        self.progress_value = 0
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.text_signal.connect(self.set_prompt)
        self.busy = False
        self.task = None
        self.latent_rgb_factors = torch.tensor([
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=torch.float, device='cuda')

    @QtCore.Slot(str)
    def set_prompt(self, text):
        self.content.prompt.setText(text)


    def evalImplementation_thread(self, prompt_override=None, args=None, init_image=None):

        task = self.content.task.currentText()
        if task == "TXT2IMG":
            task_type = 'text2img'
        else:
            task_type = 'inpainting'

        if f"kandinsky" not in gs.models or self.task != task_type:
            gs.models["kandinsky"] = get_kandinsky2('cuda', task_type=task_type, model_version='2.1', use_flash_attention=False)
            self.task = task_type
        masks = self.getInputData(0)
        images = self.getInputData(1)
        data = self.getInputData(2)

        prompt = self.content.prompt.toPlainText()
        if data:
            if "prompt" in data:
                prompt = data["prompt"]
                self.content.text_signal.emit(prompt)
            else:
                prompt = self.content.prompt.toPlainText()
        num_steps = self.content.steps.value()
        guidance_scale = self.content.cfg_scale.value()
        h = self.content.h_param.value()
        w = self.content.w_param.value()
        sampler = self.content.sampler.currentText()
        prior_cf_scale = self.content.prior_cf_scale.value()
        prior_steps = self.content.prior_steps.value()
        self.seed = self.content.seed.text()
        try:
            self.seed = int(self.seed)
        except:
            self.seed = get_fixed_seed('')
        return_images = []
        return_pil_images = []
        strength = self.content.strength.value()
        if prompt_override is not None:
            num_steps = args.steps
            images = init_image
            prompt = prompt_override
            strength = args.strength
            guidance_scale = int(args.scale)
            h = args.H
            w = args.W
            print(prompt, strength, guidance_scale, num_steps)
            self.seed = args.seed
        torch.manual_seed(self.seed)

        if images is not None:
            for image in images:

                if task_type == "text2img":

                    pil_img = pixmap_to_pil_image(image)
                    return_pil_images = gs.models["kandinsky"].generate_img2img(
                        prompt,
                        pil_img,
                        strength=strength,
                        num_steps=num_steps,
                        batch_size=1,
                        guidance_scale=guidance_scale,
                        h=h,
                        w=w,
                        sampler=sampler,
                        prior_cf_scale=prior_cf_scale,
                        prior_steps=str(prior_steps),
                    )
                else:
                    pil_img = pixmap_to_pil_image(image)
                    img_mask = pixmap_to_pil_image(masks[0]).convert("L")

                    # Get the original dimensions
                    #original_height, original_width = img_mask.size

                    # Calculate the new dimensions
                    #new_width = int(original_width // 8)
                    #new_height = int(original_height // 8)

                    # Resize the image
                    resized_img_mask = img_mask.resize(pil_img.size)

                    return_pil_images = gs.models["kandinsky"].generate_inpainting(prompt,
                                                                                    pil_img,
                                                                                    np.array(resized_img_mask),
                                                                                    num_steps=num_steps,
                                                                                    batch_size=1,
                                                                                    guidance_scale=guidance_scale,
                                                                                    h=pil_img.size[1],
                                                                                    w=pil_img.size[0],
                                                                                    sampler="ddim_sampler",
                                                                                    prior_cf_scale=prior_cf_scale,
                                                                                    prior_steps=str(prior_steps),
                                                                                    negative_prior_prompt="",
                                                                                    negative_decoder_prompt="",
                    )
        else:
            return_pil_images = gs.models["kandinsky"].generate_text2img(
                prompt,
                num_steps=num_steps,
                batch_size=1,
                guidance_scale=guidance_scale,
                h=h, w=w,
                sampler=sampler,
                prior_cf_scale=prior_cf_scale,
                prior_steps=str(prior_steps),
                callback=self.callback

            )
        for image in return_pil_images:
            pixmap = pil_image_to_pixmap(image)
            return_images.append(pixmap)
        return return_images
    def callback(self, tensors):
        print("cb")
        i = tensors["i"]
        #self.content.progress_signal.emit(1)
        #if self.content.tensor_preview.isChecked():
        if i < self.content.steps.value():

            latent = torch.einsum('...lhw,lr -> ...rhw', tensors["denoised"][0], self.latent_rgb_factors)
            latent = (((latent + 1) / 2)
                      .clamp(0, 1)  # change scale from -1..1 to 0..1
                      .mul(0xFF)  # to 0..255
                      .byte())
            # Copying to cpu as numpy array
            latent = rearrange(latent, 'c h w -> h w c').detach().cpu().numpy()
            img = Image.fromarray(latent)
            img = img.resize((img.size[0] * 8, img.size[1] * 8), resample=Image.LANCZOS)
            latent_pixmap = pil_image_to_pixmap(img)
            self.setOutput(0, [latent_pixmap])
            if len(self.getOutputs(0)) > 0:
                nodes = self.getOutputs(0)
                for node in nodes:
                    if isinstance(node, ImagePreviewNode):
                        node.content.preview_signal.emit(latent_pixmap)
                    if isinstance(node, VideoOutputNode):
                        frame = np.array(img)
                        node.content.video.add_frame(frame, dump=node.content.dump_at.value())


    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        self.progress_value = 0
        self.executeChild(output_index=1)

    @QtCore.Slot()
    def setSeed(self):
        self.content.seed.setText(str(self.seed))

    @QtCore.Slot()
    def setProgress(self, progress=None):
        if progress != 100 and progress != 0:
            self.progress_value = self.progress_value + self.single_step
        self.content.progress_bar.setValue(self.progress_value)

    def onInputChanged(self, socket=None):
        pass