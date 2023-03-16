from PIL import Image
#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap, \
    pixmap_composite_method_list

OP_NODE_IMAGE_BLEND = get_next_opcode()

class BlendWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_layouts()
        self.setLayout(self.main_layout)

    def create_widgets(self):
        self.text_label = QtWidgets.QLabel("Image Operator:")
        self.blend = self.create_double_spin_box("Blend:", 0.00, 1.00, 0.01, 0.00)
        self.composite_method = self.create_combo_box(pixmap_composite_method_list, "Composite Method:")

    def create_layouts(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.composite_method)
        self.main_layout.addWidget(self.blend)


@register_node(OP_NODE_IMAGE_BLEND)
class BlendNode(AiNode):
    icon = "icons/in.png"
    op_code = OP_NODE_IMAGE_BLEND
    op_title = "Image Blend"
    content_label_objname = "image_blend_node"
    category = "image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])
        self.painter = QtGui.QPainter()
        #self.eval()
        #self.content.eval_signal.connect(self.eval)

    def initInnerClasses(self):
        self.content = BlendWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]

        self.grNode.height = 220
    @QtCore.Slot(int)
    def evalImplementation(self, index=0):
        if self.getInput(1) != None:
            node, index = self.getInput(1)
            pixmap1 = node.getOutput(index)
        else:
            pixmap1 = None
        if self.getInput(0) != None:
            node, index = self.getInput(0)
            pixmap2 = node.getOutput(index)
        else:
            pixmap2 = None

        if pixmap1 != None and pixmap2 != None:
            method = self.content.composite_method.currentText()
            if method == 'blend':
                blend = self.content.blend.value()
                value = self.image_op(pixmap1, pixmap2, blend)
                print(f"BLEND NODE: Using both inputs with a blend value: {blend}")
            elif method == "composite":
                # Create a new RGBA image with the same dimensions as the RGB image
                image1 = pixmap_to_pil_image(pixmap1)
                image2 = pixmap_to_pil_image(pixmap2)

                result_image = Image.new("RGBA", image1.size, (0, 0, 0, 0))

                # Use the mask image to composite the RGB image onto the result image
                image = Image.composite(image1, result_image, image2)
                value = pil_image_to_pixmap(image)
            elif method in pixmap_composite_method_list:
                value = self.composite_pixmaps(pixmap1, pixmap2, method)
                #print(self.value)
            self.setOutput(0, value)
            self.markDirty(False)
            self.markInvalid(False)
        elif pixmap2 != None:
            try:
                self.setOutput(0, pixmap2)
                print(f"BLEND NODE: Using only First input")
            except:
                pass
        elif pixmap1 != None:
            try:
                self.setOutput(0, pixmap1)
                print(f"BLEND NODE: Using only Second input")
            except:
                pass
        self.executeChild(output_index=1)
        return None
    def onMarkedDirty(self):
        self.value = None
    def eval(self):
        self.markDirty(True)
        self.evalImplementation()
    def image_op(self, pixmap1, pixmap2, blend):
        # Convert the QPixmap object to a PIL Image object
        image1 = pixmap_to_pil_image(pixmap1).convert("RGBA")
        image2 = pixmap_to_pil_image(pixmap2).convert("RGBA")

        image = Image.blend(image1, image2, blend)
        #print(blend, image)

        # Convert the PIL Image object to a QPixmap object
        pixmap = pil_image_to_pixmap(image)

        return pixmap

    def composite_pixmaps(self, pixmap1, pixmap2, method):
        """
        Composite two pixmaps using a specified compositing method.

        :param pixmap1: First pixmap to composite
        :type pixmap1: QPixmap
        :param pixmap2: Second pixmap to composite
        :type pixmap2: QPixmap
        :param method: Compositing method to use
        :type method: str
        :return: Composite pixmap
        :rtype: QPixmap
        """
        method_valid = True
        #self.result_pixmap = None
        # Create a new pixmap to store the composite
        # Create a QPainter object to draw on the result pixmap
        #self.setOutput(0, self.result_pixmap)
        #self.result_pixmap = QtGui.QPixmap(pixmap1.size())

        self.painter.begin(pixmap2)

        # Set the compositing mode based on the specified method
        if method == 'source_over':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
        elif method == 'destination_over':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationOver)
        elif method == 'clear':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        elif method == 'source':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Source)
        elif method == 'destination':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Destination)
        elif method == 'source_in':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceIn)
        elif method == 'destination_in':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationIn)
        elif method == 'source_out':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOut)
        elif method == 'destination_out':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationOut)
        elif method == 'source_atop':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        elif method == 'destination_atop':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationAtop)
        elif method == 'xor':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Xor)
        elif method == 'overlay':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Overlay)
        elif method == 'screen':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Screen)
        elif method == 'soft_light':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_SoftLight)
        elif method == 'hard_light':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_HardLight)
        elif method == 'color_dodge':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_ColorDodge)
        elif method == 'color_burn':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_ColorBurn)
        elif method == 'darken':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Darken)
        elif method == 'lighten':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Lighten)
        elif method == 'exclusion':
            self.painter.setCompositionMode(QtGui.QPainter.CompositionMode_Exclusion)
        else:
            method_valid = False

        if method_valid == True:

            # Draw the first pixmap onto the result pixmap
            #self.painter.drawPixmap(0, 0, pixmap1)

            # Composite the second pixmap onto the result pixmap using the specified compositing method
            self.painter.drawPixmap(0, 0, pixmap1)

            # End painting
            self.painter.end()

        return pixmap2