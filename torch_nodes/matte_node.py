import cv2
import numpy as np
from PIL import Image
from qtpy import QtCore, QtGui

from ..ainodes_backend.matte.matte import MatteInference
from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_MATTE = get_next_opcode()
class MatteWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.create_main_layout()



@register_node(OP_NODE_MATTE)
class MatteNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/humanmask.png"
    op_code = OP_NODE_MATTE
    op_title = "Matting"
    content_label_objname = "image_matte_node"
    category = "Image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,5,1])
        #self.eval()
        #self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = MatteWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.output_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]
        self.input_socket_name = ["EXEC", "IMAGE"]

        self.grNode.height = 200
        self.grNode.width = 280
        self.content.eval_signal.connect(self.evalImplementation)

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        self.busy = True

        pixmaps = self.getInputData(0)
        if gs.debug:
            print(type(pixmaps))


        if pixmaps is not None:
            for pixmap1 in pixmaps:

                self.load_matte()
                image = pixmap_to_pil_image(pixmap1)
                np_image = np.array(image)
                bg_mask, fg, fg_alpha, bg_alpha = gs.models["matte"].infer(np_image)
                x = 0
                for i in bg_mask:
                    #print(i)
                    if i[0] > 1:
                        bg_mask[x] = [255]


                bg_mask = cv2.GaussianBlur(bg_mask, (5, 5), 0)
                bg_mask = bg_mask.reshape(*bg_mask.shape, 1)
                #test_np_image = (bg_mask) * np_image
                #print(test_np_image.shape)
                fg_with_alpha = Image.fromarray(fg_alpha)
                bg_with_alpha = Image.fromarray(bg_alpha)
                fg_with_black_bg_image = Image.fromarray(fg)

                np_bg_image = np_image * (1 - bg_mask / 255)
                np_bg_image = np_bg_image.astype(np_image.dtype)
                #print(np_bg_image.shape)

                bg_image = Image.fromarray(np_bg_image)
                bg_pixmap = pil_image_to_pixmap(bg_with_alpha)
                fg_pixmap = pil_image_to_pixmap(fg_with_alpha)
                return([bg_pixmap, fg_pixmap])




        return self.value
    def composite(self, background, foreground, alpha):
        composite = background * (1 - alpha / 255) + foreground * (alpha / 255)
        return composite.astype(background.dtype)
    def feather_mask(self, mask, feather_width):
        feather_mask = np.zeros_like(mask, dtype=np.float)
        for i in range(feather_width):
            feather_mask[i, :] = i / feather_width
            feather_mask[-i - 1, :] = i / feather_width
            feather_mask[:, i] = i / feather_width
            feather_mask[:, -i - 1] = i / feather_width
        feather_mask = np.minimum(feather_mask, mask.astype(np.float) / 255)
        feather_mask = feather_mask[..., np.newaxis]
        return feather_mask
    def shrink_mask(self, mask, kernel_size, iterations):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_1ch = mask.squeeze()
        shrunken_mask_1ch = cv2.erode(mask_1ch, kernel, iterations=iterations)
        shrunken_mask_3ch = cv2.cvtColor(
            shrunken_mask_1ch[..., np.newaxis], cv2.COLOR_GRAY2BGR
        )
        return shrunken_mask_3ch

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        self.setOutput(0, [result[0]])
        self.setOutput(1, [result[1]])

        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)


    def image_op(self, pixmap1, pixmap2, blend):
        # Convert the QPixmap object to a PIL Image object
        image1 = pixmap_to_pil_image(pixmap1).convert("RGBA")
        image2 = pixmap_to_pil_image(pixmap2).convert("RGBA")

        image = Image.blend(image1, image2, blend)
        #print(blend, image)

        # Convert the PIL Image object to a QPixmap object
        pixmap = pil_image_to_pixmap(image)

        return pixmap
    def load_matte(self):
        if "matte" not in gs.models:
            gs.models["matte"] = MatteInference()
        return

