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

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import CalcNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

#from PyQt5.Qsci import QsciScintilla, QsciLexerPython

OP_NODE_K_SAMPLER = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

class KSamplerWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_layouts()
        self.setLayout(self.main_layout)

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
        self.denoise = self.create_double_spin_box("Denoise:", 0.00, 2.00, 0.01, 1.00)
        self.guidance_scale = self.create_double_spin_box("Guidance Scale:", 1.01, 100.00, 0.01, 7.50)
        self.button = QtWidgets.QPushButton("Run")
        self.fix_seed_button = QtWidgets.QPushButton("Fix Seed")

    def create_layouts(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 25)
        self.main_layout.addLayout(self.schedulers.layout)
        self.main_layout.addLayout(self.sampler.layout)
        self.main_layout.addLayout(self.seed.layout)
        self.main_layout.addLayout(self.steps.layout)
        self.main_layout.addLayout(self.start_step.layout)
        self.main_layout.addLayout(self.last_step.layout)
        stop_early_layout = QtWidgets.QVBoxLayout()
        stop_early_layout.addWidget(self.stop_early)
        stop_early_layout.addWidget(self.force_denoise)
        stop_early_layout.addWidget(self.disable_noise)
        stop_early_layout.addWidget(self.denoise)
        self.main_layout.addLayout(stop_early_layout)
        self.main_layout.addLayout(self.guidance_scale.layout)
        self.main_layout.addLayout(self.create_button_layout([self.button, self.fix_seed_button]))

    """def serialize(self):
        res = super().serialize()
        res['scheduler'] = self.schedulers.currentText()
        res['sampler'] = self.sampler.currentText()
        res['seed'] = self.seed.text()
        res['steps'] = self.steps.value()
        res['guidance_scale'] = self.guidance_scale.value()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.schedulers.setCurrentText(data['scheduler'])
            self.sampler.setCurrentText(data['sampler'])
            self.seed.setText(data['seed'])
            self.steps.setValue(data['steps'])
            self.guidance_scale.setValue(data['guidance_scale'])
            return True & res
        except Exception as e:
            dumpException(e)
        return res"""


@register_node(OP_NODE_K_SAMPLER)
class KSamplerNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_K_SAMPLER
    op_title = "K Sampler"
    content_label_objname = "K_sampling_node"
    category = "sampling"
    def __init__(self, scene):
        super().__init__(scene, inputs=[2,3,3,1], outputs=[5,2,1])
        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.content = KSamplerWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 500
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.input_socket_name = ["EXEC", "COND", "N COND", "LATENT"]
        self.output_socket_name = ["EXEC", "LATENT", "IMAGE"]
        self.seed = ""
        self.content.fix_seed_button.clicked.connect(self.setSeed)

    def evalImplementation(self, index=0):


        self.markDirty(True)
        if self.value is None:
            # Start the worker thread
            if self.busy == False:
                self.busy = True
                thread0 = threading.Thread(target=self.k_sampling)
                thread0.start()
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            return self.value

    def onMarkedDirty(self):
        self.value = None
    def k_sampling(self, progress_callback=None):
        try:
            cond_node, index = self.getInput(2)
            cond = cond_node.getOutput(index)
        except Exception as e:
            print(e)
            cond = None
        try:
            n_cond_node, index = self.getInput(1)
            n_cond = n_cond_node.getOutput(index)
        except:
            n_cond = None
        try:
            latent_node, index = self.getInput(0)
            latent = latent_node.getOutput(index)
        except:
            latent = torch.zeros([1, 4, 512 // 8, 512 // 8])
        self.seed = self.content.seed.text()
        try:
            self.seed = int(self.seed)
        except:
            self.seed = secrets.randbelow(99999999)
        try:
            last_step = self.content.steps.value() if self.content.stop_early.isChecked() == False else self.content.last_step.value()
            sample = common_ksampler(device="cuda",
                                     seed=self.seed,
                                     steps=self.content.steps.value(),
                                     start_step=self.content.start_step.value(),
                                     last_step=last_step,
                                     cfg=self.content.guidance_scale.value(),
                                     sampler_name=self.content.sampler.currentText(),
                                     scheduler=self.content.schedulers.currentText(),
                                     positive=cond,
                                     negative=n_cond,
                                     latent=latent,
                                     disable_noise=self.content.disable_noise.isChecked(),
                                     force_full_denoise=self.content.force_denoise.isChecked(),
                                     denoise=self.content.denoise.value())

            return_sample = sample.cpu().half()

            x_samples = gs.models["sd"].model.decode_first_stage(sample.half())
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
            image = Image.fromarray(x_sample.astype(np.uint8))
            qimage = ImageQt(image)
            pixmap = QPixmap().fromImage(qimage)
            self.value = pixmap
            del sample
            del x_samples
            x_samples = None
            sample = None
            torch_gc()
            self.onWorkerFinished([pixmap, return_sample])
        except:
            self.busy = False
            if len(self.getOutputs(2)) > 0:
                self.executeChild(output_index=2)
        return [pixmap, return_sample]
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        print("K SAMPLER:", self.content.steps.value(), "steps,", self.content.sampler.currentText(), " seed: ", self.seed)
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result[0])
        self.setOutput(1, result[1])
        self.busy = False
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)
        return
    def setSeed(self):
        self.content.seed.setText(str(self.seed))
    def onInputChanged(self, socket=None):
        pass

