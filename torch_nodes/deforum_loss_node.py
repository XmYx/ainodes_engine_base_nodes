import inspect
import random
import secrets
import threading
import time
from types import SimpleNamespace

import numpy as np
from einops import rearrange

from ..ainodes_backend import common_ksampler, torch_gc, CompVisVDenoiser

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QPixmap

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ..ainodes_backend.conditioning import exposure_loss, make_mse_loss, get_color_palette, make_clip_loss_fn
from ..ainodes_backend.conditioning import make_rgb_color_match_loss, blue_loss_fn, threshold_by, make_aesthetics_loss_fn, mean_loss_fn, var_loss_fn, exposure_loss
from ..ainodes_backend.model_wrap import CFGDenoiserWithGrad

from ..ainodes_backend.k_diffusion import external as k_diffusion_external


#from queue import Queue

OP_NODE_DEFORUM_LOSS = get_next_opcode()


class DeforumLossWidget(QDMNodeContentWidget):
    seed_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):

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


#@register_node(OP_NODE_DEFORUM_LOSS)
class KSamplerNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_DEFORUM_LOSS
    op_title = "Deforum Aesthetic Functions"
    content_label_objname = "deforum_loss_node"
    category = "aiNodes Base/Sampling"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[5,5,1], outputs=[1])
        self.content.button.clicked.connect(self.evalImplementation)
        pass
        gs.use_deforum_loss = None


        # Create a worker object
    def initInnerClasses(self):
        self.content = DeforumLossWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 550
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.content.eval_signal.connect(self.evalImplementation)

    def eval(self, index=0):
        self.markDirty(True)
        self.content.eval_signal.emit()

    def onMarkedDirty(self):
        self.value = None

    def evalImplementation_thread(self):
        # Add a task to the task queue
        return "Done"

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        self.markDirty(False)
        self.markInvalid(False)
        pass
        if len(self.getOutputs(0)) > 0:
            self.executeChild(output_index=0)
        return True

    def get_args(self):
        args = SimpleNamespace()
        args.init_mse_scale = 0.0
        args.init_mse_image = None
        args.W = 512
        args.H = 512
        args.use_alpha_as_mask = False
        args.strength = 1.0
        args.steps = 25
        args.clamp_start
        args.clamp_stop
        args.sampler
        args.colormatch_scale
        args.colormatch_image = None
        args.decode_method
        args.colormatch_n_colors
        args.ignore_sat_weight
        args.clip_scale
        args.aesthetics_scale
        args.exposure_scale
        args.blue_scale
        args.mean_scale
        args.var_scale
        args.clamp_grad_threshold
        args.grad_threshold_type
        args.clamp_schedule
        args.grad_inject_timing
        args.gradient_wrt
        args.gradient_add_to
        args.cond_uncond_sync

        return args

    def create_loss_fns(self):

        gs.use_deforum_loss = True

        if gs.models["sd"].model.parameterization == "v":
            model_wrap = CompVisVDenoiser(self.model_denoise, quantize=True)
        else:
            model_wrap = k_diffusion_external.CompVisDenoiser(self.model_denoise, quantize=True)

        # Init MSE loss image
        init_mse_image = None
        if args.init_mse_scale and args.init_mse_image != None and args.init_mse_image != '':
            init_mse_image, mask_image = load_img(args.init_mse_image,
                                                  shape=(args.W, args.H),
                                                  use_alpha_as_mask=args.use_alpha_as_mask)
            init_mse_image = init_mse_image.to("cuda")
            init_mse_image = repeat(init_mse_image, '1 ... -> b ...', b=1)

        assert not (args.init_mse_scale != 0 and (
                    args.init_mse_image is None or args.init_mse_image == '')), "Need an init image when init_mse_scale != 0"

        t_enc = int((1.0 - args.strength) * args.steps)

        # Noise schedule for the k-diffusion samplers (used for masking)
        k_sigmas = model_wrap.get_sigmas(args.steps)
        args.clamp_schedule = dict(
            zip(k_sigmas.tolist(), np.linspace(args.clamp_start, args.clamp_stop, args.steps + 1)))
        k_sigmas = k_sigmas[len(k_sigmas) - t_enc - 1:]



        if args.colormatch_scale != 0:
            assert args.colormatch_image is not None, "If using color match loss, colormatch_image is needed"
            colormatch_image, _ = load_img(args.colormatch_image)
            colormatch_image = colormatch_image.to('cpu')
            del (_)
        else:
            colormatch_image = None

        # Loss functions
        if args.init_mse_scale != 0:
            if args.decode_method == "linear":
                mse_loss_fn = make_mse_loss(root.model.linear_decode(
                    root.model.get_first_stage_encoding(root.model.encode_first_stage(init_mse_image.to(root.device)))))
            else:
                mse_loss_fn = make_mse_loss(init_mse_image)
        else:
            mse_loss_fn = None

        if args.colormatch_scale != 0:
            _, _ = get_color_palette(root, args.colormatch_n_colors, colormatch_image,
                                     verbose=True)  # display target color palette outside the latent space
            if args.decode_method == "linear":
                grad_img_shape = (int(args.W / 8), int(args.H / 8))
                colormatch_image = root.model.linear_decode(root.model.get_first_stage_encoding(
                    root.model.encode_first_stage(colormatch_image.to(root.device))))
                colormatch_image = colormatch_image.to('cpu')
            else:
                grad_img_shape = (args.W, args.H)
            color_loss_fn = make_rgb_color_match_loss(root,
                                                      colormatch_image,
                                                      n_colors=args.colormatch_n_colors,
                                                      img_shape=grad_img_shape,
                                                      ignore_sat_weight=args.ignore_sat_weight)
        else:
            color_loss_fn = None

        if args.clip_scale != 0:
            clip_loss_fn = make_clip_loss_fn(root, args)
        else:
            clip_loss_fn = None

        if args.aesthetics_scale != 0:
            aesthetics_loss_fn = make_aesthetics_loss_fn(root, args)
        else:
            aesthetics_loss_fn = None

        if args.exposure_scale != 0:
            exposure_loss_fn = exposure_loss(args.exposure_target)
        else:
            exposure_loss_fn = None

        loss_fns_scales = [
            [clip_loss_fn, args.clip_scale],
            [blue_loss_fn, args.blue_scale],
            [mean_loss_fn, args.mean_scale],
            [exposure_loss_fn, args.exposure_scale],
            [var_loss_fn, args.var_scale],
            [mse_loss_fn, args.init_mse_scale],
            [color_loss_fn, args.colormatch_scale],
            [aesthetics_loss_fn, args.aesthetics_scale]
        ]

        # Conditioning gradients not implemented for ddim or PLMS

        callback = SamplerCallback(args=args,
                                   root=root,
                                   mask=mask,
                                   init_latent=init_latent,
                                   sigmas=k_sigmas,
                                   sampler=sampler,
                                   verbose=False).callback

        clamp_fn = threshold_by(threshold=args.clamp_grad_threshold, threshold_type=args.grad_threshold_type,
                                clamp_schedule=args.clamp_schedule)

        grad_inject_timing_fn = make_inject_timing_fn(args.grad_inject_timing, model_wrap, args.steps)

        cfg_model = CFGDenoiserWithGrad(model_wrap,
                                        loss_fns_scales,
                                        clamp_fn,
                                        args.gradient_wrt,
                                        args.gradient_add_to,
                                        args.cond_uncond_sync,
                                        decode_method=args.decode_method,
                                        grad_inject_timing_fn=grad_inject_timing_fn,
                                        # option to use grad in only a few of the steps
                                        grad_consolidate_fn=None,  # function to add grad to image fn(img, grad, sigma)
                                        verbose=False)


    def onInputChanged(self, socket=None):
        pass
