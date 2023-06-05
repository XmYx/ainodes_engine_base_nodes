import neovim
from PyQt6.QtCore import Qt
from qtpy import QtCore, QtGui
from qtpy import QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_VIM = get_next_opcode()

class VimWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.create_button_layout([self.run_button, self.stop_button])

@register_node(OP_NODE_VIM)
class VimNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/exec.png"
    op_code = OP_NODE_VIM
    op_title = "NeoVIM"
    content_label_objname = "exec_node"
    category = "Experimental"
    help_text = "Execution Node\n\n" \
                "Execution chain is essential\n" \
                "in aiNodes. You control the flow\n" \
                "You control the magic. Each value\n" \
                "is created and stored at execution\n" \
                "once a node is validated, you don't\n" \
                "have to run it again in order to get\n" \
                "it's value, just simply connect the\n" \
                "relevant data line. Only execute, if you\n" \
                "want, or have to get a new value."

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.interrupt = False

    def initInnerClasses(self):
        self.content = VimWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.height = 200
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(160)
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.run_button.clicked.connect(self.start)
        self.content.stop_button.clicked.connect(self.stop)


    def evalImplementation_thread(self, index=0, *args, **kwargs):
        return True

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        self.executeChild(0)

    #@QtCore.Slot()
    def stop(self):
        print("Interrupting Execution of Graph")
        gs.should_run = None


    #@QtCore.Slot()
    def start(self):
        gs.should_run = True
        self.content.eval_signal.emit()









class NeovimWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = QtGui.QWindow(self)
        self.container = QtWidgets.QWidget.createWindowContainer(self.window, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.container)

        # Create and attach the Neovim instance to the QWindow
        nvim = neovim.attach('socket', path_to_neovim_socket)
        self.window.create()
        self.window.setFlags(Qt.Widget)
        self.window.setContent(self.container)
        self.window.setTitle('Neovim Editor')
        self.window.setVisible(True)
        self.window.setKeyboardGrabEnabled(True)
        self.window.setMouseGrabEnabled(True)
        self.window.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.window.setFocus(True)
        self.window.resetFocusReason()
