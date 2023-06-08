from types import SimpleNamespace

import cv2
import glob
import os

import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

import math
import os

import numpy as np
import torch
from PIL import Image
from qtpy import QtWidgets, QtGui, QtCore

from ..ainodes_backend.model_loader import UpscalerLoader
from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_RESRGAN = get_next_opcode()

models = [
    {"name":"RealESRGAN x4 Plus",
     "code_name":"RealESRGAN_x4plus"},
    {"name":"RealESRNet x4 Plus",
     "code_name":"RealESRNet_x4plus"},
    {"name":"RealESRGAN x4 Plus Anime 6B",
     "code_name":"RealESRGAN_x4plus_anime_6B"},
    {"name":"RealESRGAN x2 Plus",
     "code_name":"RealESRGAN_x2plus"},
    {"name":"RealESRGAN Anime Video v3",
     "code_name":"realesr-animevideov3"},
    {"name":"RealESRGAN General x4 v3",
     "code_name":"realesr-general-x4v3"}
]

class RealESRGWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)

    def create_widgets(self):

        gans = []
        for item in models:
            gans.append(item["name"])

        self.dropdown = self.create_combo_box(gans, "Models")
        self.outscale = self.create_spin_box("Upscale Amount", min_val=1, max_val=16, default_val=8)
        self.denoise_strength = self.create_double_spin_box("Denoise Strength", default_val=0.5)
        self.tiles = self.create_spin_box("Tiles", min_val=0, max_val=16, default_val=0)
        self.pre_pad = self.create_spin_box("Pre Padding", min_val=0, max_val=512, default_val=0)
        self.tile_pad = self.create_spin_box("Tile Padding", min_val=0, max_val=512, default_val=0)
        self.face_enhance = self.create_check_box("Face Enhancement")
        self.fp32 = self.create_check_box("FP32")



class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_RESRGAN)
class REALESRGANNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/realesrgan.png"
    op_code = OP_NODE_RESRGAN
    op_title = "REALESRGan"
    content_label_objname = "realesrgan_node"
    category = "Upscalers"

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])

    def initInnerClasses(self):
        self.content = RealESRGWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 340
        self.grNode.height = 360
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        images = self.getInputData(0)
        return_pixmaps = []

        args = SimpleNamespace()

        i = self.content.dropdown.currentIndex()
        model_name = models[i]["code_name"]

        args.model_name = model_name
        args.outscale = self.content.outscale.value()
        args.denoise_strength = self.content.denoise_strength.value()
        args.tile = self.content.tiles.value()
        args.pre_pad = self.content.pre_pad.value()
        args.tile_pad = self.content.tile_pad.value()
        args.face_enhance = self.content.face_enhance.isChecked()
        args.fp32 = self.content.fp32.isChecked()
        args.alpha_upsampler = "realesrgan"
        args.gpu_id = 0

        if images:
            for image in images:

                img = pixmap_to_pil_image(image).convert("RGB")

                img = upscale_image(img, args)
                if img != None:

                    pixmap = pil_image_to_pixmap(img)
                    return_pixmaps.append(pixmap)

        return return_pixmaps

    def onWorkerFinished(self, result):
        self.busy = False
        if result:
            self.setOutput(0, result)
            self.markDirty(False)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
    def onInputChanged(self, socket=None):
        pass

def upscale_image(image, args):
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    model_path = os.path.join('models/upscalers', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'models/upscalers'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer

        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    img = np.array(image)

    try:
        if args.face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=args.outscale)


        output_img = Image.fromarray(output)
        return output_img
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        return None
