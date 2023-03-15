import threading

#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

OP_NODE_EXEC = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

class ExecWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_layouts()
        self.setLayout(self.main_layout)

    def create_widgets(self):
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.checkbox = self.create_check_box("Run in thread")

    def create_layouts(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 25)
        self.main_layout.addWidget(self.run_button)
        self.main_layout.addWidget(self.stop_button)
        #self.main_layout.addWidget(self.checkbox)

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
        # Create a worker object
    def initInnerClasses(self):
        self.content = ExecWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 200
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(160)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC"]

        #self.content.setMinimumHeight(400)
        #self.content.setMinimumWidth(256)
        #self.content.image.changeEvent.connect(self.onInputChanged)

    def evalImplementation(self, index=0):
        self.markDirty(True)
        self.markInvalid(True)
        if not self.interrupt:
            if len(self.getOutputs(0)) > 0:
                if self.content.checkbox.isChecked() == True:
                    thread0 = threading.Thread(target=self.executeChild, args=(0,))
                    thread0.start()
                else:
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










