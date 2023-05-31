from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap, \
    pixmap_composite_method_list
from ainodes_frontend import singleton as gs
from ..ainodes_backend.face_replacement.fr_model import FaceReplacementModel

OP_NODE_FACE_SWAP = get_next_opcode()

class FaceSwapperWidget(QDMNodeContentWidget):

    def initUI(self):
        self.create_main_layout()



@register_node(OP_NODE_FACE_SWAP)
class FaceSwapperNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/image_blend.png"
    op_code = OP_NODE_FACE_SWAP
    op_title = "Face Swapper"
    content_label_objname = "face_swapper_node"
    category = "Image"
    help_text = "Face Swapper\n\n" \
                ""

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])

    def initInnerClasses(self):
        self.content = FaceSwapperWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.output_socket_name = ["EXEC", "IMAGE"]
        self.input_socket_name = ["EXEC", "IMAGE1", "IMAGE2"]
        self.grNode.height = 220
        self.content.eval_signal.connect(self.evalImplementation)
        #if "faceswap" not in gs.models:
        gs.models["faceswap"] = FaceReplacementModel()
        self.new = None

    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        pixmap1 = self.getInputData(0)
        pixmap2 = self.getInputData(1)

        if pixmap1 and pixmap2:

            print("SWAPPING")

            face_img = pixmap_to_pil_image(pixmap1[0])
            target_img = pixmap_to_pil_image(pixmap2[0])
            result = gs.models["faceswap"](face_img, target_img, self.new)

            pixmap = pil_image_to_pixmap(result)

            if self.new:
                self.new = None

            return [pixmap]
        elif pixmap1:
            return [pixmap1[0]]
        elif pixmap2:
            return [pixmap2[0]]
        else:
            return None
    def onInputChanged(self, socket=None):
        self.new = True

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.setOutput(0, result)
        self.executeChild(output_index=1)
