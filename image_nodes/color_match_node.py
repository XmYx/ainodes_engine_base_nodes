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


@register_node(OP_NODE_COLORMATCH)
class ColorMatch(AiNode):
    icon = "icons/in.png"
    op_code = OP_NODE_COLORMATCH
    op_title = "Color Match"
    content_label_objname = "colormatch_node"
    category = "image"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])
        self.busy = False

        # Create a worker object
    def initInnerClasses(self):
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 140
        self.grNode.width = 256
    def evalImplementation(self, index=0):



        if self.getInput(0) != None and self.getInput(1) != None:
            node, index = self.getInput(0)
            pixmap1 = node.getOutput(index)
            node, index = self.getInput(1)
            pixmap2 = node.getOutput(index)

            pil_image_1 = pixmap_to_pil_image(pixmap1).convert('RGB')
            pil_image_2 = pixmap_to_pil_image(pixmap2).convert('RGB')

            np_image_1 = np.array(pil_image_1)
            np_image_2 = np.array(pil_image_2)

            np_image_1 = cv2.cvtColor(np_image_1, cv2.COLOR_RGB2BGR)
            np_image_2 = cv2.cvtColor(np_image_2, cv2.COLOR_RGB2BGR)

            matched_image = maintain_colors(np_image_2, np_image_1, 'Match Frame 0 HSV')
            matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
            print(matched_image.shape)
            pil_image = Image.fromarray(matched_image)
            print(pil_image)
            pixmap = pil_image_to_pixmap(pil_image)

            self.setOutput(0, pixmap)
            self.markDirty(False)

            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
            return None
        else:
            self.markDirty(True)
            return None


def maintain_colors(prev_img, color_match_sample, mode):
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