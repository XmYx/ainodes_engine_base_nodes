import inspect
import random
import secrets
import threading
import time

import numpy as np
from einops import rearrange

from .ksampler_node import get_fixed_seed
from ..ainodes_backend import common_ksampler, torch_gc, pil_image_to_pixmap, pixmap_to_pil_image

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QPixmap

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from queue import Queue

from kandinsky2 import get_kandinsky2

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
        self.create_main_layout()
    def create_widgets(self):
        self.prompt = self.create_text_edit("Prompt:")
        self.seed = self.create_line_edit("Seed:")
        self.steps = self.create_spin_box("Steps:", 1, 10000, 25)
        self.cfg_scale = self.create_spin_box("Guidance Scale:", 0, 1000, 4)
        self.w_param = self.create_spin_box("Width:", 64, 2048, 512, 64)
        self.h_param = self.create_spin_box("Height:", 64, 2048, 512, 64)
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
        super().__init__(scene, inputs=[5,6,1], outputs=[5,1])
        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False

        # Create a worker object
    def initInnerClasses(self):
        self.content = KandinskyWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 550
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.seed = ""
        #self.content.fix_seed_button.clicked.connect(self.setSeed)
        self.content.seed_signal.connect(self.setSeed)
        self.content.progress_signal.connect(self.setProgress)
        self.progress_value = 0
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.text_signal.connect(self.set_prompt)
    @QtCore.Slot(str)
    def set_prompt(self, text):
        self.content.prompt.setText(text)

    def eval(self, index=0):
        self.markDirty(True)
        self.content.eval_signal.emit()

    def onMarkedDirty(self):
        self.value = None

    def evalImplementation_thread(self):

        if "kandinsky" not in gs.models:
            gs.models["kandinsky"] = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)
        images = self.getInputData(0)
        data = self.getInputData(1)
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
        torch.manual_seed(self.seed)
        return_images = []
        return_pil_images = []
        if images is not None:
            for image in images:
                pil_img = pixmap_to_pil_image(image)
                return_pil_images = gs.models["kandinsky"].generate_img2img(
                    prompt,
                    pil_img,
                    strength=0.7,
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
            return_pil_images = gs.models["kandinsky"].generate_text2img(
                prompt,
                num_steps=num_steps,
                batch_size=1,
                guidance_scale=guidance_scale,
                h=h, w=w,
                sampler=sampler,
                prior_cf_scale=prior_cf_scale,
                prior_steps=str(prior_steps)
            )

        for image in return_pil_images:
            pixmap = pil_image_to_pixmap(image)
            return_images.append(pixmap)
        return return_images

    def callback(self, tensors):
        self.setProgress()
        return
        for key, value in tensors.items():
            if key == 'i':
                print(value)
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        if gs.logging:
            print("K SAMPLER:", self.content.steps.value(), "steps,", self.content.sampler.currentText(), " seed: ", self.seed)
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)

        self.busy = False
        #self.content.progress_signal.emit(100)
        self.progress_value = 0
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)

        return True
    @QtCore.Slot()
    def setSeed(self):
        self.content.seed.setText(str(self.seed))
    @QtCore.Slot()
    def setProgress(self, progress=None):
        if progress != 100 and progress != 0:
            self.progress_value = self.progress_value + self.single_step
        #print(self.progress_value)
        self.content.progress_bar.setValue(self.progress_value)
    def onInputChanged(self, socket=None):
        pass
