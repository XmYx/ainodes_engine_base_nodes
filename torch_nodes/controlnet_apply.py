import copy
import threading
import time
import numpy as np
import torch
from qtpy import QtWidgets, QtCore

from ..ainodes_backend import torch_gc, pixmap_to_pil_image

from ainodes_frontend import singleton as gs
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ..ainodes_backend.cnet_preprocessors.refonly.hook import ControlModelType, ControlParams, UnetHook

OP_NODE_CN_APPLY = get_next_opcode()
class CNApplyWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
        self.main_layout.setContentsMargins(15, 15, 15, 25)
    def create_widgets(self):
        self.strength = self.create_double_spin_box("Strength", 0.01, 100.00, 0.01, 1.00)
        self.control_net_selector = self.create_combo_box(["controlnet", "t2i", "reference"], "Control Style")
        self.button = QtWidgets.QPushButton("Run")
        self.create_button_layout([self.button])

@register_node(OP_NODE_CN_APPLY)
class CNApplyNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_CN_APPLY
    op_title = "Apply ControlNet"
    content_label_objname = "CN_apply_node"
    category = "ControlNet"

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,3,1], outputs=[3,1])

        self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        self.latest_network = None
        # Create a worker object
    def initInnerClasses(self):
        self.content = CNApplyWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 340
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.content.eval_signal.connect(self.evalImplementation)
    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.markDirty(True)
        self.markInvalid(True)
        latent_node, index = self.getInput(0)
        image_list = latent_node.getOutput(index)
        return_list = []
        style = self.content.control_net_selector.currentText()
        if style == "reference":
            for image in image_list:
                result = self.apply_ref_control(image)
                return_list.append(result)
        else:
            cond_node, index = self.getInput(1)
            conditioning_list = cond_node.getOutput(index)
            if len(conditioning_list) == 1:
                for image in image_list:
                    result = self.add_control_image(conditioning_list[0], image)
                    return_list.append(result)
            elif len(conditioning_list) == len(image_list):
                x = 0
                for image in image_list:
                    result = self.add_control_image(conditioning_list[x], image)
                    return_list.append(result)
        return return_list

    def onMarkedDirty(self):
        self.value = None

    def apply_ref_control(self, image):

        if self.latest_network is not None:
            try:
                self.latest_network.restore(gs.models["sd"].model.model.diffusion_model)
            except:
                pass

        #unet = gs.models["sd"].model.model.diffusion_model

        gs.models["sd"].model.cuda()

        model_net = None

        image = pixmap_to_pil_image(image)

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        # c = []
        control_hint = image.movedim(-1, 1).to("cuda")

        # input_image = HWC3(np.asarray(input_image))

        # control = detected_map

        control_model_type = ControlModelType.AttentionInjection

        forward_params = []

        forward_param = ControlParams(
            control_model=model_net,
            hint_cond=control_hint,
            weight=1.0,
            guidance_stopped=False,
            start_guidance_percent=0,
            stop_guidance_percent=100,
            advanced_weighting=None,
            control_model_type=control_model_type,
            global_average_pooling=False,
            hr_hint_cond=control_hint,
            batch_size=1,
            instance_counter=0,
            is_vanilla_samplers=True,
            cfg_scale=7.5,
            soft_injection=True,
            cfg_injection=False,
        )
        forward_params.append(forward_param)

        del model_net
        self.latest_network = UnetHook(lowvram=False)
        self.latest_network.hook(model=gs.models["sd"].model.model.diffusion_model, sd_ldm=gs.models["sd"].model, control_params=forward_params)
        return "Done"

    def add_control_image(self, conditioning, image, progress_callback=None):
        image = pixmap_to_pil_image(image)
        #image = image.convert("RGB")
        #print("DEBUG IMAGE", image)

        #image.save("CNET.png", "PNG")

        array = np.array(image)
        #print("ARRAY", array)

        image = np.array(image).astype(np.float32) / 255.0
        print("IMAGE", image)


        image = torch.from_numpy(image)[None,]
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['control_hint'] = control_hint
            n[1]['control_strength'] = self.content.strength.value()
            c.append(n)
        return c
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        self.markDirty(False)
        self.markInvalid(False)
        self.busy = False
        self.setOutput(0, result)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(1)
        return
    def onInputChanged(self, socket=None):
        pass
