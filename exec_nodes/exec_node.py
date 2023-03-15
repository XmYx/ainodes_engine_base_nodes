from qtpy import QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_EXEC = get_next_opcode()

class ExecWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.create_button_layout([self.run_button, self.stop_button])

@register_node(OP_NODE_EXEC)
class ExecNode(AiNode):
    icon = "icons/in.png"
    op_code = OP_NODE_EXEC
    op_title = "Execute"
    content_label_objname = "exec_node"
    category = "exec"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.content.run_button.clicked.connect(self.start)
        self.content.stop_button.clicked.connect(self.stop)
        self.interrupt = False
    def initInnerClasses(self):
        self.content = ExecWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 200
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(160)
    def evalImplementation(self, index=0):
        self.markDirty(True)
        self.markInvalid(True)
        if not self.interrupt:
            self.executeChild(0)
        return None
    def onMarkedDirty(self):
        self.value = None

    def stop(self):
        self.interrupt = True
        return
    def start(self):
        self.interrupt = False
        self.evalImplementation(0)










