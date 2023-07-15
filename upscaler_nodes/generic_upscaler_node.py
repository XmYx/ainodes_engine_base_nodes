import math
import os

import numpy as np
import torch
from PIL import Image
from qtpy import QtWidgets, QtGui, QtCore

from ..ainodes_backend.model_loader import UpscalerLoader
from ..ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, pil2tensor

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_TORCH_UPSCALER = get_next_opcode()

class UpscalerWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()

    def create_widgets(self):
        checkpoint_folder = gs.upscalers

        print(os.getcwd())

        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.dropdown = self.create_combo_box(checkpoint_files, "Models")
        if checkpoint_files == []:
            self.dropdown.addItem(f"Please place a model in {gs.upscalers}")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/{gs.upscalers},")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")


class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_TORCH_UPSCALER)
class UpscalerNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/upscaler.png"
    op_code = OP_NODE_TORCH_UPSCALER
    op_title = "Torch Upscaler"
    content_label_objname = "torch_upscaler_node"
    category = "aiNodes Base/Upscalers"

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        self.loader = UpscalerLoader()

    def initInnerClasses(self):
        self.content = UpscalerWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 340
        self.grNode.height = 180
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        #try:
        model_name = self.content.dropdown.currentText()
        model_path = f"{gs.upscalers}/{model_name}"
        loaded = self.loader.load_model(model_path, model_name)
        print("UPSCALER LOADER:", loaded)

        images = self.getInputData(0)
        return_pixmaps = []
        try:
            if images:
                for image in images:

                    img = tensor2pil(image).convert("RGB")
                    img = np.array(img).astype(np.float32) / 255.0
                    img = torch.from_numpy(img)[None,]

                    in_img = img.movedim(-1, -3).to("cuda")

                    tile = 128 + 64
                    overlap = 8
                    gs.models[model_name].to("cuda")
                    s = tiled_scale(in_img, lambda a: gs.models[model_name](a), tile_x=tile, tile_y=tile,
                                                overlap=overlap, upscale_amount=gs.models[model_name].scale, pbar=None)
                    gs.models[model_name].cpu()
                    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0) * 255
                    img = s[0].detach().numpy().astype(np.uint8)
                    img = Image.fromarray(img)
                    pixmap = pil2tensor(img)
                    return_pixmaps.append(pixmap)
        except:
            return_pixmaps = []

        return return_pixmaps

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if result:
            self.setOutput(0, result)
            self.markDirty(False)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
    def onInputChanged(self, socket=None):
        pass

def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))

@torch.inference_mode()
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, pbar = None):
    #print(samples.shape)

    output = torch.empty((samples.shape[0], out_channels, round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)), device="cpu")
    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        out = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
        out_div = torch.zeros((s.shape[0], out_channels, round(s.shape[2] * upscale_amount), round(s.shape[3] * upscale_amount)), device="cpu")
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                        mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                        mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                out[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += ps * mask
                out_div[:,:,round(y*upscale_amount):round((y+tile_y)*upscale_amount),round(x*upscale_amount):round((x+tile_x)*upscale_amount)] += mask
                if pbar is not None:
                    pbar.update(1)

        output[b:b+1] = out/out_div
    return output

