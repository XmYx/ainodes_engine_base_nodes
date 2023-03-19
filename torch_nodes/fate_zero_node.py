import inspect
import secrets
import threading

import numpy as np
from einops import rearrange

from ..ainodes_backend import common_ksampler, torch_gc

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QPixmap

from ainodes_frontend import singleton
gs = singleton.Singleton.instance()
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ..ainodes_backend.video_diffusion.p2p_infer import load_fate_zero, infer_fate

OP_NODE_FATE_ZERO = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]


class FateZeroWidget(QDMNodeContentWidget):
    seed_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        self.schedulers = self.create_combo_box(SCHEDULERS, "Scheduler:")
        self.sampler = self.create_combo_box(SAMPLERS, "Sampler:")
        self.seed = self.create_line_edit("Seed:")
        self.steps = self.create_spin_box("Steps:", 1, 10000, 10)
        self.start_step = self.create_spin_box("Start Step:", 0, 1000, 0)
        self.last_step = self.create_spin_box("Last Step:", 1, 1000, 5)
        self.stop_early = self.create_check_box("Stop Sampling Early")
        self.force_denoise = self.create_check_box("Force full denoise", checked=True)
        self.disable_noise = self.create_check_box("Disable noise generation")
        self.iterate_seed = self.create_check_box("Iterate seed")
        self.denoise = self.create_double_spin_box("Denoise:", 0.00, 2.00, 0.01, 1.00)
        self.guidance_scale = self.create_double_spin_box("Guidance Scale:", 1.01, 100.00, 0.01, 7.50)
        self.button = QtWidgets.QPushButton("Run")
        self.fix_seed_button = QtWidgets.QPushButton("Fix Seed")
        self.create_button_layout([self.button, self.fix_seed_button])
        self.progress_bar = self.create_progress_bar("progress", 0, 100, 0)

@register_node(OP_NODE_FATE_ZERO)
class FateZeroNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_FATE_ZERO
    op_title = "Fate Zero"
    content_label_objname = "fate_zero_node"
    category = "sampling"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[2,3,3,1], outputs=[5,2,1])
        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False

        # Create a worker object
    def initInnerClasses(self):
        self.content = FateZeroWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 550
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.seed = ""
        self.content.fix_seed_button.clicked.connect(self.setSeed)
        self.content.seed_signal.connect(self.setSeed)
        self.content.progress_signal.connect(self.setProgress)
        self.progress_value = 0
    def evalImplementation(self, index=0):
        load_fate_zero()
        infer_fate()

        self.markDirty(True)
        if self.value is None:
            # Start the worker thread
            if self.busy == False:
                self.busy = True
                self.content.progress_signal.emit(0)
                #thread0 = threading.Thread(target=self.k_sampling)
                #thread0.start()
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def k_sampling(self):
        load_fate_zero()
        infer_fate()
        return None
    def decode_sample(self, sample):
        x_samples = gs.models["sd"].model.decode_first_stage(sample.half())
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
        return x_sample

    def callback(self, tensors):
        self.setProgress()
        return
        for key, value in tensors.items():
            if key == 'i':
                print(value)
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        print("K SAMPLER:", self.content.steps.value(), "steps,", self.content.sampler.currentText(), " seed: ", self.seed)
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result[0])
        self.setOutput(1, result[1])
        self.busy = False
        self.content.progress_signal.emit(100)
        self.progress_value = 0
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)
        return
    @QtCore.Slot()
    def setSeed(self):
        self.content.seed.setText(str(self.seed))
    @QtCore.Slot(int)
    def setProgress(self, progress=None):
        if progress != 100 and progress != 0:
            self.progress_value = self.progress_value + self.single_step
        #print(self.progress_value)
        self.content.progress_bar.setValue(self.progress_value)
    def onInputChanged(self, socket=None):
        pass

