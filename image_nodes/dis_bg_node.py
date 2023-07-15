from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, \
    pil2tensor

OP_NODE_DIS = get_next_opcode()

class DISBGWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.create_main_layout()

@register_node(OP_NODE_DIS)
class DISBGNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/bg_removal.png"
    op_code = OP_NODE_DIS
    op_title = "Background Separation"
    content_label_objname = "image_disbg_node"
    category = "aiNodes Base/Image"

    custom_output_socket_name = ["FG", "BG", "MASK", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,5,5,1])

    def initInnerClasses(self):
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QImage
        self.content = DISBGWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QImage(self.grNode.icon).scaled(64, 64, Qt.KeepAspectRatio)
        self.grNode.height = 200
        self.grNode.width = 280
        self.content.eval_signal.connect(self.evalImplementation)
        self.model = None

    def evalImplementation_thread(self, index=0):

        if self.model == None:
            from ..ainodes_backend.dis_bg.dis_gb_removal import DISRemoval
            self.model = DISRemoval()
        self.busy = True
        pixmaps = self.getInputData(0)
        if gs.debug:
            print(type(pixmaps))
        if pixmaps is not None:
            for pixmap1 in pixmaps:
                image = tensor2pil(pixmap1)
                img, bg, mask = self.model.inference(image)
                fg_pixmap = pil2tensor(img)
                bg_pixmap = pil2tensor(bg)
                mask_pixmap = pil2tensor(mask)
                return([bg_pixmap, fg_pixmap, mask_pixmap])

    def onWorkerFinished(self, result):
        self.busy = False
        self.setOutput(0, [result[0]])
        self.setOutput(1, [result[1]])
        self.setOutput(2, [result[2]])
        if len(self.getOutputs(3)) > 0:
            self.executeChild(output_index=3)
