import inspect
import random
import secrets
import threading
import time

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
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from queue import Queue

OP_NODE_K_SAMPLER = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

class KSamplerWidget(QDMNodeContentWidget):
    seed_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
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

@register_node(OP_NODE_K_SAMPLER)
class KSamplerNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_K_SAMPLER
    op_title = "K Sampler"
    content_label_objname = "K_sampling_node"
    category = "Sampling"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[6,2,3,3,1], outputs=[5,2,1])
        self.content.button.clicked.connect(self.eval)
        #pass

        # Create a worker object
    def initInnerClasses(self):
        self.content = KSamplerWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 600
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.seed = ""
        self.content.fix_seed_button.clicked.connect(self.setSeed)
        self.content.seed_signal.connect(self.setSeed)
        self.content.progress_signal.connect(self.setProgress)
        self.progress_value = 0
        self.content.eval_signal.connect(self.evalImplementation)


    @QtCore.Slot()
    def evalImplementation_thread(self, cond_override = None, args = None, latent_override=None):
        #pass
        # Add a task to the task queue
        cond_list = self.getInputData(3)
        n_cond_list = self.getInputData(2)

        latent_list = self.getInputData(1)
        data = self.getInputData(0)
        last_step = self.content.steps.value() if self.content.stop_early.isChecked() == False else self.content.last_step.value()
        short_steps = last_step - self.content.start_step.value()
        steps = self.content.steps.value()
        self.single_step = 100 / steps if self.content.start_step.value() == 0 and last_step == steps else short_steps
        self.progress_value = 0


        if latent_list == None:
            latent_list = [torch.zeros([1, 4, 512 // 8, 512 // 8])]

        self.seed = self.content.seed.text()
        try:
            self.seed = int(self.seed)
        except:
            self.seed = get_fixed_seed('')
        if self.content.iterate_seed.isChecked() == True:
            self.content.seed_signal.emit()
            self.seed += 1
        return_pixmaps = []
        return_samples = []
        generator = torch.manual_seed(self.seed)
        try:
            x=0
            if data:
                if "model" in data:
                    from diffusers.utils import pt_to_pil

                    if data["model"] == "deepfloyd_1":


                        image = gs.models["deepfloyd_1"](prompt_embeds=cond_list[0][0], negative_prompt_embeds=n_cond_list[0][0],
                                                         generator=generator, output_type="pt", num_inference_steps=steps).images
                        gs.models["deepfloyd_1"].to("cpu")

                    elif data["model"] == "deepfloyd_2":
                        generator = torch.manual_seed(self.seed)
                        image = gs.models["deepfloyd_2"](
                            image=latent_list[0], prompt_embeds=cond_list[0][0], negative_prompt_embeds=n_cond_list[0][0],
                            generator=generator, output_type="pt"
                        ).images
                        gs.models["deepfloyd_2"].to("cpu")

                    elif data["model"] == "deepfloyd_3":
                        if gs.logging:
                            print("UPSCALE X4")
                        latent = latent_list[0]
                        if latent.shape[1] == 4:  # Checking if tensor has 4 channels
                            latent = latent[:, 1:, :, :]  # Stripping the first channel
                        image = gs.models["deepfloyd_3"](prompt=data["prompt"], image=latent, generator=generator, noise_level=100).images
                        gs.models["deepfloyd_3"].to("cpu")

                if data["model"] not in ["deepfloyd_3"]:
                    img = pt_to_pil(image)[0]
                else:
                    img = image[0]
                qimage = ImageQt(img)
                pixmap = QPixmap().fromImage(qimage)

                #if len(self.getOutputs(0)) > 0:
                #    node = self.getOutputs(0)[0]
                #    if hasattr(node.content, "preview_signal"):
                #        # print("emitting")
                #        node.content.preview_signal.emit(pixmap)
                self.content.progress_signal.emit(0)
                return_pixmaps.append(pixmap)
                return_samples.append(image)



            else:

                if cond_override is not None:
                    cond_list = cond_override[0]
                    n_cond_list = cond_override[1]

                if len(cond_list) < len(latent_list):
                    new_cond_list = []
                    for x in range(len(latent_list)):
                        new_cond_list.append(cond_list[0])
                    # cond_list = len(latent_list) * cond_list[0]
                    cond_list = new_cond_list

                for cond in cond_list:
                    if len(latent_list) == len(cond_list):
                        latent = latent_list[x]
                    else:
                        latent = latent_list[0]
                    if len(n_cond_list) == len(cond_list):
                        n_cond = n_cond_list[x]
                    else:
                        n_cond = n_cond_list[0]
                    for i in cond:
                        if 'control_hint' in i[1]:
                            cond = self.apply_control_net(cond)

                    self.denoise = self.content.denoise.value()
                    self.steps = self.content.steps.value()
                    self.cfg = self.content.guidance_scale.value()
                    self.start_step = self.content.start_step.value()
                    self.sampler_name = self.content.sampler.currentText()
                    self.scheduler = self.content.schedulers.currentText()

                    if data is not None:
                        self.update_vars(data)

                    if cond_override is not None:
                        denoise = 1.0 if args.strength == 0 or not args.use_init else args.strength
                        if latent_override is not None:
                            latent = latent_override

                        self.seed = args.seed
                        self.steps = args.steps
                        self.cfg = args.scale
                        self.start_step = 0

                        print("SD SAMPLER NODE OVERRIDE", steps, cfg, denoise, args.use_init)



                    sample = common_ksampler(device="cuda",
                                             seed=self.seed,
                                             steps=self.steps,
                                             start_step=self.start_step,
                                             last_step=self.last_step,
                                             cfg=self.cfg,
                                             sampler_name=self.sampler_name,
                                             scheduler=self.scheduler,
                                             positive=cond,
                                             negative=n_cond,
                                             latent=latent,
                                             disable_noise=self.content.disable_noise.isChecked(),
                                             force_full_denoise=self.content.force_denoise.isChecked(),
                                             denoise=self.denoise,
                                             callback=self.callback)

                    for c in cond:
                        if "control" in c[1]:
                            del c[1]["control"]

                    return_sample = sample.detach().cpu().half()
                    return_samples.append(return_sample)
                    x_sample = self.decode_sample(sample)
                    image = Image.fromarray(x_sample.astype(np.uint8))
                qimage = ImageQt(image)
                pixmap = QPixmap().fromImage(qimage)

                #if len(self.getOutputs(0)) > 0:
                #    node = self.getOutputs(0)[0]
                #    if hasattr(node.content, "preview_signal"):
                #        #print("emitting")
                #        node.content.preview_signal.emit(pixmap)

                self.content.progress_signal.emit(0)
                return_pixmaps.append(pixmap)
                x+=1
            return [return_pixmaps, return_samples]
        except Exception as e:
            return_pixmaps, return_samples = None, None
            print(e)
        return [return_pixmaps, return_samples]
    def decode_sample(self, sample):
        x_sample = gs.models["vae"].decode(sample.half())
        x_sample = 255. * x_sample[0].detach().numpy()
        return x_sample

    def callback(self, tensors):
        self.setProgress()
        return
        for key, value in tensors.items():
            if key == 'i':
                print(value)

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):

        super().onWorkerFinished(None)

        if gs.logging:
            print("K SAMPLER:", self.content.steps.value(), "steps,", self.content.sampler.currentText(), " seed: ", self.seed, "images", result[0])
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result[0])
        self.setOutput(1, result[1])


        #self.content.progress_signal.emit(100)
        self.progress_value = 0
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)

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
    def apply_control_net(self, conditioning, progress_callback=None):
        cnet_string = 'controlnet'
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = gs.models[cnet_string]
            c_net.set_cond_hint(t[1]['control_hint'], t[1]['control_strength'])
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control'].control_model.cpu()
            #del c_net
            c.append(n)
        return c

def get_fixed_seed(seed):
    if seed is None or seed == '':
        sign = random.choice([-1, 1])
        value = secrets.randbelow(999999999999999999)
        return sign * value

def enable_misc_optimizations():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark_limit = 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if torch.backends.cudnn.benchmark:
        print("Enabled CUDNN Benchmark Sucessfully")
    else:
        print("CUDNN Benchmark Disabled")
    if torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32:
        print("Enabled CUDA & CUDNN TF32 Sucessfully")
    else:
        print("CUDA & CUDNN TF32 Disabled")
    if not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction and not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        print("CUDA Matmul fp16/bf16 Reduced Precision Reduction Disabled")
    else:
        print("CUDA Matmul fp16/bf16 Reduced Precision Reduction Expected Value Mismatch")