from qtpy import QtCore
from ..ainodes_backend.dis_bg.dis_gb_removal import DISRemoval
from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_DIS = get_next_opcode()

class DISBGWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.create_main_layout()

@register_node(OP_NODE_DIS)
class DISBGNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/bg.png"
    op_code = OP_NODE_DIS
    op_title = "Background Separation"
    content_label_objname = "image_disbg_node"
    category = "Image"

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,5,1])

    def initInnerClasses(self):
        self.content = DISBGWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 200
        self.grNode.width = 280
        self.content.eval_signal.connect(self.evalImplementation)
        self.model = None

    def evalImplementation_thread(self, index=0):
        if self.model == None:
            self.model = DISRemoval()
        self.busy = True
        pixmaps = self.getInputData(0)
        if gs.debug:
            print(type(pixmaps))
        if pixmaps is not None:
            for pixmap1 in pixmaps:
                image = pixmap_to_pil_image(pixmap1)
                img, mask = self.model.inference(image)
                fg_pixmap = pil_image_to_pixmap(img)
                bg_pixmap = pil_image_to_pixmap(mask)
                return([bg_pixmap, fg_pixmap])
        return self.value

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.setOutput(0, [result[0]])
        self.setOutput(1, [result[1]])
        if len(self.getOutputs(2)) > 0:
            self.executeChild(output_index=2)
