import inspect
import os
import random
import secrets
import threading
import time

import numpy as np
from einops import rearrange

from ..ainodes_backend import common_ksampler, torch_gc, pil_image_to_pixmap

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QPixmap

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode, handle_ainodes_exception
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from queue import Queue

from ..image_nodes.output_node import ImagePreviewNode
from ..video_nodes.video_save_node import VideoOutputNode

OP_NODE_ONNX = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "dpmpp_2m_alt", "ddim", "uni_pc", "uni_pc_bh2"]

class OnnxWidget(QDMNodeContentWidget):
    seed_signal = QtCore.Signal()
    #progress_signal = QtCore.Signal(int)
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

@register_node(OP_NODE_ONNX)
class OnnxNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_ONNX
    op_title = "ONNX"
    content_label_objname = "onnx_sampling_node"
    category = "Sampling"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[6,2,3,3,1], outputs=[5,2,1])
        #pass

        # Create a worker object
    def initInnerClasses(self):
        self.content = OnnxWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 600
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.seed = ""
        self.content.fix_seed_button.clicked.connect(self.setSeed)
        self.content.seed_signal.connect(self.setSeed)
        #self.content.progress_signal.connect(self.setProgress)
        self.progress_value = 0
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.button.clicked.connect(self.content.eval_signal)



    @QtCore.Slot()
    def evalImplementation_thread(self, cond_override = None, args = None, latent_override=None):


        done = optimize("runwayml/stable-diffusion-v1-5", Path("unopt"), Path("opt") )

        return [None]
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):

        super().onWorkerFinished(None)

        #if gs.logging:
        #    print("K SAMPLER:", self.content.steps.value(), "steps,", self.content.sampler.currentText(), " seed: ", self.seed, "images", result[0])
        self.markDirty(False)
        self.markInvalid(False)


        #self.content.progress_signal.emit(100)
        self.progress_value = 0
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)
    @QtCore.Slot(str)
    def setSeed(self):
        self.content.seed.setText(str(self.seed))
    @QtCore.Slot(int)
    def setProgress(self, progress=None):
        if progress != 100 and progress != 0:
            self.progress_value = self.progress_value + self.single_step
        #print(self.progress_value)
        self.content.progress_bar.setValue(self.progress_value)

import argparse
import json
import shutil
import warnings
from pathlib import Path

import onnxruntime as ort
#import PySimpleGUI as sg
import torch
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline
from packaging import version
#from user_script import get_base_model_name

from olive.model import ONNXModel
from olive.workflows import run as olive_run

def optimize(
    model_id: str,
    unoptimized_model_dir: Path,
    optimized_model_dir: Path,
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        exit(1)

    ort.set_default_logger_severity(4)
    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
    shutil.rmtree(optimized_model_dir, ignore_errors=True)
    #os.makedirs("unopt")
    #os.makedirs("opt")

    # The model_id and base_model_id are identical when optimizing a standard stable diffusion model like
    # runwayml/stable-diffusion-v1-5. These variables are only different when optimizing a LoRA variant.
    base_model_id = "runwayml/stable-diffusion-v1-5"

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)

    model_info = dict()

    for submodel_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with open(f"models/configs/olive/config_{submodel_name}.json", "r") as fin:
            olive_config = json.load(fin)

        if submodel_name in ("unet", "text_encoder"):
            olive_config["input_model"]["config"]["model_path"] = model_id
        else:
            # Only the unet & text encoder are affected by LoRA, so it's better to use the base model ID for
            # other models: the Olive cache is based on the JSON config, and two LoRA variants with the same
            # base model ID should be able to reuse previously optimized copies.
            olive_config["input_model"]["config"]["model_path"] = base_model_id

        olive_run(olive_config)

        footprints_file_path = (
            Path(f"footprints/{submodel_name}_gpu-dml_footprints.json")
        )
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            for _, footprint in footprints.items():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint

            #assert conversion_footprint and optimizer_footprint

            unoptimized_olive_model = ONNXModel(**conversion_footprint["model_config"]["config"])
            optimized_olive_model = ONNXModel(**optimizer_footprint["model_config"]["config"])

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_olive_model.model_path),
                },
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    print("\nCreating ONNX pipeline...")
    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(model_info["vae_encoder"]["unoptimized"]["path"].parent),
        vae_decoder=OnnxRuntimeModel.from_pretrained(model_info["vae_decoder"]["unoptimized"]["path"].parent),
        text_encoder=OnnxRuntimeModel.from_pretrained(model_info["text_encoder"]["unoptimized"]["path"].parent),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(model_info["unet"]["unoptimized"]["path"].parent),
        scheduler=pipeline.scheduler,
        safety_checker=OnnxRuntimeModel.from_pretrained(model_info["safety_checker"]["unoptimized"]["path"].parent),
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=True,
    )

    print("Saving unoptimized models...")
    onnx_pipeline.save_pretrained(unoptimized_model_dir)

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    print("Copying optimized models...")
    shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns("weights.pb"))
    for submodel_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / "model.onnx"
        shutil.copyfile(src_path, dst_path)