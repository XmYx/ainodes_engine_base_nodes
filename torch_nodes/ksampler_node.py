import copy
import os
import random
import secrets
import numpy as np
from einops import rearrange

from ..ainodes_backend import tensor_image_to_pixmap, get_torch_device, common_ksampler

import torch
from PIL import Image
from PIL.ImageQt import ImageQt, QImage
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QPixmap

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode, handle_ainodes_exception
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from queue import Queue

from ..image_nodes.image_preview_node import ImagePreviewNode
from ..video_nodes.video_save_node import VideoOutputNode

OP_NODE_K_SAMPLER = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "dpmpp_2m_alt", "ddim", "uni_pc", "uni_pc_bh2"]

class KSamplerWidget(QDMNodeContentWidget):
    seed_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)
    preview_signal = QtCore.Signal(object)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.schedulers = self.create_combo_box(SCHEDULERS, "Scheduler:")
        self.sampler = self.create_combo_box(SAMPLERS, "Sampler:")
        self.seed = self.create_line_edit("Seed:", placeholder="Leave empty for random seed")
        self.steps = self.create_spin_box("Steps:", 1, 10000, 10)
        self.start_step = self.create_spin_box("Start Step:", 0, 1000, 0)
        self.last_step = self.create_spin_box("Last Step:", 1, 1000, 5)
        self.stop_early = self.create_check_box("Stop Sampling Early")
        self.force_denoise = self.create_check_box("Force full denoise", checked=True)
        self.preview_type = self.create_combo_box(["taesd", "quick-rgb"], "Preview Type")
        self.tensor_preview = self.create_check_box("Show Tensor Preview", checked=True)
        self.disable_noise = self.create_check_box("Disable noise generation")
        self.iterate_seed = self.create_check_box("Iterate seed")
        self.use_internal_latent = self.create_check_box("Use latent from loop")
        self.denoise = self.create_double_spin_box("Denoise:", 0.00, 25.00, 0.01, 1.00)
        self.guidance_scale = self.create_double_spin_box("Guidance Scale:", 1.01, 100.00, 0.01, 7.50)
        #self.button = QtWidgets.QPushButton("Run")
        self.fix_seed_button = QtWidgets.QPushButton("Fix Seed")
        self.create_button_layout([self.fix_seed_button])
        self.progress_bar = self.create_progress_bar("progress", 0, 100, 0)

@register_node(OP_NODE_K_SAMPLER)
class KSamplerNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/k_sampler.png"
    op_code = OP_NODE_K_SAMPLER
    op_title = "K Sampler"
    content_label_objname = "K_sampling_node"
    category = "aiNodes Base/Sampling"

    NodeContent_class = KSamplerWidget
    #dim = (256, 800)

    make_dirty = True


    custom_input_socket_name = ["CONTROLNET", "VAE", "MODEL", "DATA", "LATENT", "NEG COND", "POS COND", "EXEC"]

    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[4,4,4,6,2,3,3,1], outputs=[5,2,1])
        self.seed = ""
        self.content.fix_seed_button.clicked.connect(self.setSeed)
        self.content.seed_signal.connect(self.setSeed)
        self.content.progress_signal.connect(self.setProgress)
        self.content.preview_signal.connect(self.handle_preview)
        self.device = get_torch_device()
        self.grNode.height = 750
        self.grNode.width = 320
        self.content.setMinimumWidth(316)
        self.content.setMinimumHeight(500)
        self.update_all_sockets()

    #     # Create a worker object
    # def initInnerClasses(self):
    #     self.content = KSamplerWidget(self)
    #     self.grNode = CalcGraphicsNode(self)
    #     self.grNode.icon = self.icon
    #     self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)
    #     self.progress_value = 0
    #     self.content.eval_signal.connect(self.evalImplementation_thread)
    #     self.content.button.clicked.connect(self.evalImplementation)


    def set_rgb_factor(self, type="classic"):
        if hasattr(self, "latent_rgb_factors"):
            self.latent_rgb_factors.to("cpu")
            del self.latent_rgb_factors
        if type == "classic":
            self.latent_rgb_factors = torch.tensor([
                #   R        G        B
                [0.298, 0.207, 0.208],  # L1
                [0.187, 0.286, 0.173],  # L2
                [-0.158, 0.189, 0.264],  # L3
                [-0.184, -0.271, -0.473],  # L4
            ], dtype=torch.float, device="cpu")
        else:
            self.latent_rgb_factors = torch.tensor([
                #   R        G        B
                [0.3920, 0.4054, 0.4549],
                [-0.2634, -0.0196, 0.0653],
                [0.0568, 0.1687, -0.0755],
                [-0.3112, -0.2359, -0.2076]
            ], dtype=torch.float, device="cpu")

    def evalImplementation_thread_(self):
        from nodes import common_ksampler as ksampler
        print("running new")
        model = self.getInputData(2)

        seed = self.content.seed.text()
        try:
            seed = int(seed)
        except:
            seed = get_fixed_seed('')
        if self.content.iterate_seed.isChecked() == True:
            self.content.seed_signal.emit()
            seed += 1
        steps = self.content.steps.value()
        cfg = self.content.guidance_scale.value()
        sampler_name = self.content.sampler.currentText()
        scheduler = self.content.schedulers.currentText()
        positive = self.getInputData(6)
        negative = self.getInputData(5)
        latent_image = self.getInputData(4)
        denoise = self.content.denoise.value()

        print(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

        return [ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                               denoise=denoise)]

    #@QtCore.Slot()
    def evalImplementation_thread(self, cond_override = None, args = None, latent_override=None):
        self.progress_value = 0
        vae = self.getInputData(1)
        unet = self.getInputData(2)
        data = self.getInputData(3)
        latent = self.getInputData(4)
        n_cond = self.getInputData(5)
        cond = self.getInputData(6)

        if latent is None:
            latent = torch.zeros([1, 4, 512 // 8, 512 // 8])
        latent = {"samples":latent}
        seed = self.content.seed.text()
        try:
            seed = int(seed)
        except:
            seed = get_fixed_seed('')
        if self.content.iterate_seed.isChecked() == True:
            self.content.seed_signal.emit()
            seed += 1
        steps = self.content.steps.value()
        cfg = self.content.guidance_scale.value()
        sampler_name = self.content.sampler.currentText()
        scheduler = self.content.schedulers.currentText()
        start_step = 0
        denoise = self.content.denoise.value()
        force_full_denoise = self.content.force_denoise.isChecked()
        if cond_override:
            cond = cond_override[0]
            n_cond = cond_override[1]
            denoise = 1.0 if args.strength == 0 or not args.use_init else args.strength
            force_full_denoise = True if denoise == 1.0 else False
            latent = {"samples": latent_override}
            seed = args.seed
            steps = args.steps
            cfg = args.scale
            start_step = 0
            print("Generating using override seed: [", seed, "]", denoise)
        if cond is not None:
            self.last_step = steps if self.content.stop_early.isChecked() == False else self.content.last_step.value()
            short_steps = self.last_step - self.content.start_step.value()
            self.single_step = int(100 / steps) if self.content.start_step.value() == 0 and self.last_step == steps else int(short_steps)
            generator = torch.manual_seed(seed)
            from comfy import model_base
            self.model_version = "xl" if type(unet.model) in [model_base.SDXL, model_base.SDXLRefiner] else "classic"
            self.set_rgb_factor(self.model_version)
            self.preview_mode = self.content.preview_type.currentText()
            taesd_decoder_version = "taesd_decoder.pth" if self.model_version == "classic" else "taesdxl_decoder.pth"
            if self.preview_mode == "taesd" and os.path.isfile(taesd_decoder_version):
                from comfy.taesd.taesd import TAESD
                self.taesd = TAESD(None, f"models/vae/{taesd_decoder_version}").to("cuda")
            else:
                print(f"TAESD enabled, but models/vae/{taesd_decoder_version} was not found, switching to simple RGB Preview")
                self.preview_mode = "quick-rgb"

            sample = common_ksampler(model=unet,
                                     seed=seed,
                                     steps=steps,
                                     cfg=cfg,
                                     sampler_name=sampler_name,
                                     scheduler=scheduler,
                                     positive=cond,
                                     negative=n_cond,
                                     latent=latent,
                                     denoise=denoise,
                                     disable_noise=self.content.disable_noise.isChecked(),
                                     start_step=start_step,
                                     last_step=steps,
                                     force_full_denoise=force_full_denoise,
                                     callback=self.callback)

            x_sample = self.decode_sample(sample[0]["samples"], vae)
            return_samples = sample[0]["samples"].detach()
            return_latents = x_sample.detach()
            if self.content.tensor_preview.isChecked():
                if len(self.getOutputs(2)) > 0:
                    nodes = self.getOutputs(0)
                    for node in nodes:
                        if isinstance(node, ImagePreviewNode):
                            node.content.preview_signal.emit(tensor_image_to_pixmap(x_sample))

            return [return_latents, {"samples": return_samples}]
        else:
            return [None, None]

        # unet = self.getInputData(2)
        #
        #
        #
        # from ..ainodes_backend import tensor_image_to_pixmap, get_torch_device, common_ksampler
        # self.content.progress_signal.emit(0)
        #
        # #pass
        # # Add a task to the task queue
        # cond_list = [self.getInputData(6)]
        # n_cond_list = [self.getInputData(5)]
        # # print("C", cond_list, isinstance(cond_list, dict))
        # if isinstance(cond_list[0], dict):
        #     cond_list = cond_list[0]["conds"]
        #
        #     # print(len(cond_list))
        #
        #     if len(cond_list) == 1:
        #         cond_list = [cond_list]
        #
        # if isinstance(n_cond_list[0], dict):
        #     n_cond_list = n_cond_list[0]["conds"]
        #
        #     if len(n_cond_list) == 1:
        #         n_cond_list = [n_cond_list]
        #
        # self.steps = self.content.steps.value()
        # latent_list = self.getInputData(4)
        #
        # pre_latent = torch.zeros([1, 4, 512 // 8, 512 // 8])
        #
        # if latent_list != None:
        #     latent_list = [latent_list.get("samples")]
        # else:
        #     latent_list = [pre_latent]
        #
        # data = self.getInputData(3)
        #
        # from comfy import model_base
        #
        # self.model_version = "xl" if type(unet.model) in [model_base.SDXL, model_base.SDXLRefiner] else "classic"
        # self.set_rgb_factor(self.model_version)
        # self.preview_mode = self.content.preview_type.currentText()
        # taesd_decoder_version = "taesd_decoder.pth" if self.model_version == "classic" else "taesdxl_decoder.pth"
        # if self.preview_mode == "taesd" and os.path.isfile(taesd_decoder_version):
        #     from comfy.taesd.taesd import TAESD
        #     self.taesd = TAESD(None, f"models/vae/{taesd_decoder_version}").to("cuda")
        # else:
        #     print(f"TAESD enabled, but models/vae/{taesd_decoder_version} was not found, switching to simple RGB Preview")
        #     self.preview_mode = "quick-rgb"
        #
        # vae = self.getInputData(1)
        # control_model = self.getInputData(0)
        # #unet.cuda()
        #
        # assert unet is not None, "UNET NOT FOUND, MAKE SURE TO LOAD A MODEL AND CONNECT IT'S OUTPUTS"
        # assert vae is not None, "VAE NOT FOUND"
        # assert cond_list is not None, "POSITIVE CONDITIONING NOT FOUND, MAKE SURE TO ADD A CONDITIONING NODE"
        # assert n_cond_list is not None, "POSITIVE CONDITIONING NOT FOUND, MAKE SURE TO ADD A CONDITIONING NODE"
        #
        #
        #
        # return_latents = []
        # return_samples = []
        # try:
        #     x=0
        #     if cond_override is not None:
        #         cond_list = cond_override[0]
        #         n_cond_list = cond_override[1]
        #
        #     if len(cond_list) < len(latent_list):
        #         new_cond_list = []
        #         for x in range(len(latent_list)):
        #             new_cond_list.append(cond_list[0])
        #         # cond_list = len(latent_list) * cond_list[0]
        #         cond_list = new_cond_list
        #     cpu_s = None
        #     for cond in cond_list:
        #         self.seed = self.content.seed.text()
        #         try:
        #             self.seed = int(self.seed)
        #         except:
        #             self.seed = get_fixed_seed('')
        #         if self.content.iterate_seed.isChecked() == True:
        #             self.content.seed_signal.emit()
        #             self.seed += 1
        #         generator = torch.manual_seed(self.seed)
        #
        #         if len(latent_list) == len(cond_list):
        #             latent = latent_list[x]
        #         else:
        #             latent = latent_list[0]
        #         if len(n_cond_list) == len(cond_list):
        #             n_cond = n_cond_list[x]
        #         else:
        #             n_cond = n_cond_list[0]
        #         for i in cond:
        #             if 'control_hint' in i[1]:
        #                 cond = self.apply_control_net(cond, control_model)
        #
        #         self.denoise = self.content.denoise.value()
        #         self.steps = self.content.steps.value()
        #         self.cfg = self.content.guidance_scale.value()
        #         self.start_step = self.content.start_step.value()
        #         self.sampler_name = self.content.sampler.currentText()
        #         self.scheduler = self.content.schedulers.currentText()
        #         noise_mask = None
        #         if data is not None:
        #             self.update_vars(data)
        #             if 'noise_mask' in data:
        #                 noise_mask = data['noise_mask']
        #
        #         if cond_override:
        #             self.denoise = 1.0 if args.strength == 0 or not args.use_init else args.strength
        #             latent = latent_override
        #             self.seed = args.seed
        #             self.steps = args.steps
        #             self.cfg = args.scale
        #             self.start_step = 0
        #             print("using seed", self.seed)
        #         self.last_step = self.steps if self.content.stop_early.isChecked() == False else self.content.last_step.value()
        #         short_steps = self.last_step - self.content.start_step.value()
        #
        #         self.single_step = int(100 / self.steps) if self.content.start_step.value() == 0 and self.last_step == self.steps else int(short_steps)
        #
        #
        #         self.progress_value = 0
        #         if self.content.use_internal_latent.isChecked():
        #             if cpu_s is not None:
        #                 latent = cpu_s
        #                 self.denoise = self.content.denoise.value()
        #             else:
        #                 self.denoise = 1.0
        #         latent_dict = {}
        #         latent_dict["samples"] = latent
        #         sample = common_ksampler(model=unet,
        #                                  seed=self.seed,
        #                                  steps=self.steps,
        #                                  cfg=self.cfg,
        #                                  sampler_name=self.sampler_name,
        #                                  scheduler=self.scheduler,
        #                                  positive=cond,
        #                                  negative=n_cond,
        #                                  latent=latent_dict,
        #                                  denoise=self.denoise,
        #                                  disable_noise=self.content.disable_noise.isChecked(),
        #                                  start_step=self.start_step,
        #                                  last_step=self.last_step,
        #                                  force_full_denoise=self.content.force_denoise.isChecked(),
        #                                  callback=self.callback)
        #
        #         for c in cond:
        #             if "control" in c[1]:
        #                 del c[1]["control"]
        #         x_sample = self.decode_sample(sample[0]["samples"], vae)
        #         return_samples = sample[0]["samples"].detach()
        #         return_latents = x_sample.detach()
        #         if self.content.tensor_preview.isChecked():
        #             if len(self.getOutputs(2)) > 0:
        #                 nodes = self.getOutputs(0)
        #                 for node in nodes:
        #                     if isinstance(node, ImagePreviewNode):
        #                         node.content.preview_signal.emit(tensor_image_to_pixmap(x_sample))
        #
        #
        #         x+=1
        #     if self.preview_mode == "taesd":
        #         del self.taesd
        #     return [return_latents, {"samples":return_samples}]
        #
        # except Exception as e:
        #     handle_ainodes_exception()
        #     return_pixmaps, return_samples = None, None
        #     print(e)
        # return [return_pixmaps, return_samples]
    def decode_sample(self, sample, vae):
        decoded = vae.decode_tiled(sample)
        return decoded

    def callback(self, i, tensors, *args, **kwargs):
        # i = tensors["i"]
        self.content.progress_signal.emit(self.single_step)
        if self.content.tensor_preview.isChecked():
            if i < self.last_step - 2:
                self.content.preview_signal.emit(tensors)
    def handle_preview(self, tensors):



        if self.preview_mode == "quick-rgb":

            latent_image = tensors[0].permute(1, 2, 0).cpu() @ self.latent_rgb_factors

            latents_ubyte = (((latent_image + 1) / 2)
                             .clamp(0, 1)  # change scale from -1..1 to 0..1
                             .mul(0xFF)  # to 0..255
                             .byte())

            np_frame = latents_ubyte.numpy()
            # Convert numpy array to QImage
            h, w, c = np_frame.shape
            latent_image = QImage(np_frame.data, w, h, c * w, QImage.Format.Format_RGB888)

            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(latent_image)

            scaled_size = pixmap.size() * 8
            pixmap = pixmap.scaled(scaled_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                   QtCore.Qt.TransformationMode.SmoothTransformation)

        elif self.preview_mode == "taesd":
            x_sample = self.taesd.decoder(tensors)[0].detach()
            x_sample = x_sample.sub(0.5).mul(2)

            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
            h, w, c = x_sample.shape
            np_frame = x_sample.astype(np.uint8)
            byte_data = np_frame.tobytes()
            image = QtGui.QImage(byte_data, w, h, c * w, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(image)


        if len(self.getOutputs(0)) > 0:
            nodes = self.getOutputs(0)
            for node in nodes:
                if isinstance(node, ImagePreviewNode):
                    node.content.preview_signal.emit(pixmap)
                if isinstance(node, VideoOutputNode):
                    frame = np.array(np_frame)
                    node.content.video.add_frame(frame, dump=node.content.dump_at.value())


    def setSeed(self):
        self.content.seed.setText(str(self.seed))

    def setProgress(self, progress=None):

        self.progress_value += self.single_step
        if progress < 100:
            self.content.progress_bar.setValue(int(self.progress_value))
        else:
            self.content.progress_bar.setValue(100)
    def apply_control_net(self, conditioning, c_net, progress_callback=None):
        cnet_string = 'controlnet'


        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]




            #c_net.control_model.control_start = n[1]["control_start"]
            #c_net.control_model.control_stop = n[1]["control_stop"]
            #c_net.control_model.control_model_name = n[1]["control_model_name"]
            c_net.set_cond_hint(t[1]['control_hint'], t[1]['control_strength'])
            if 'control' in t[1]:
                #print("AND SETTING UP MULTICONTROL")
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control'].control_model.cpu()
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