from qtpy import QtCore, QtGui

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor2pil, pil2tensor
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
    category = "aiNodes Base/Image"
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
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.height = 140
        self.grNode.width = 256
        self.content.eval_signal.connect(self.evalImplementation)
        import cv2

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):

        if self.getInput(0) != None and self.getInput(1) != None:
            from ..ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap
            from PIL import Image
            import numpy as np
            node, index = self.getInput(0)
            pixmap1 = node.getOutput(index)[0]
            node, index = self.getInput(1)
            pixmap2 = node.getOutput(index)[0]

            pil_image_1 = tensor2pil(pixmap1).convert('RGB')
            pil_image_2 = tensor2pil(pixmap2).convert('RGB')

            np_image_1 = np.array(pil_image_1)
            np_image_2 = np.array(pil_image_2)

            matched_image = maintain_colors(np_image_2, np_image_1, 'LAB')

            pil_image = Image.fromarray(matched_image)
            pixmap = pil2tensor(pil_image)


            return pixmap
        else:
            self.markDirty(True)
            return None
    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if result is not None:
            self.setOutput(0, [result])
            self.markDirty(False)
            self.executeChild(output_index=1)

def maintain_colors_old(prev_img, color_match_sample, mode):
    from skimage.exposure import match_histograms
    import cv2
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
    from skimage.exposure import match_histograms
    import cv2

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