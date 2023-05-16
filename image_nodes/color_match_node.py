import threading
import numpy as np
from skimage.exposure import match_histograms
import cv2

from ..ainodes_backend import common_ksampler, torch_gc, pixmap_to_pil_image, pil_image_to_pixmap

import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import QPixmap

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_COLORMATCH = get_next_opcode()

class MatteWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.create_main_layout()


@register_node(OP_NODE_COLORMATCH)
class ColorMatch(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/color_match_c.png"
    op_code = OP_NODE_COLORMATCH
    op_title = "Color Match"
    content_label_objname = "colormatch_node"
    category = "Image"
    help_text = "Deforum ColorMatch Node\n\n" \
                "" \
                "Please refer to Deforum guide"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])
        pass

        # Create a worker object
    def initInnerClasses(self):
        self.content = MatteWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 140
        self.grNode.width = 256
        self.content.eval_signal.connect(self.evalImplementation)
    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):

        if self.getInput(0) != None and self.getInput(1) != None:
            node, index = self.getInput(0)
            pixmap1 = node.getOutput(index)[0]
            node, index = self.getInput(1)
            pixmap2 = node.getOutput(index)[0]

            pil_image_1 = pixmap_to_pil_image(pixmap1).convert('RGB')
            pil_image_2 = pixmap_to_pil_image(pixmap2).convert('RGB')

            np_image_1 = np.array(pil_image_1)
            np_image_2 = np.array(pil_image_2)

            #np_image_1 = cv2.cvtColor(np_image_1, cv2.COLOR_RGB2BGR)
            #np_image_2 = cv2.cvtColor(np_image_2, cv2.COLOR_RGB2BGR)

            matched_image = maintain_colors(np_image_2, np_image_1, 'LAB')
            #matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(matched_image)
            pixmap = pil_image_to_pixmap(pil_image)


            return pixmap
        else:
            self.markDirty(True)
            return None
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        if result is not None:
            self.setOutput(0, [result])
            self.markDirty(False)
            self.executeChild(output_index=1)

def maintain_colors_old(prev_img, color_match_sample, mode):
    #prev_img = np.float32(prev_img)
    #color_match_sample = np.float32(color_match_sample)
    from skimage.exposure import match_histograms
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, channel_axis=2)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, channel_axis=2)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


def maintain_colors(prev_img, color_match_sample, mode):
    #skimage_version = pkg_resources.get_distribution('scikit-image').version
    #is_skimage_v20_or_higher = pkg_resources.parse_version(skimage_version) >= pkg_resources.parse_version('0.20.0')

    match_histograms_kwargs = {'channel_axis': -1} # if is_skimage_v20_or_higher else {'multichannel': True}

    if mode == 'RGB':
        return match_histograms(prev_img, color_match_sample, **match_histograms_kwargs)
    elif mode == 'HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, **match_histograms_kwargs)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, **match_histograms_kwargs)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)