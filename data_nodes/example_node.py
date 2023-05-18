import os
from qtpy import QtCore
from qtpy import QtWidgets

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException
from ainodes_frontend import singleton as gs

"""
Always change the name of this variable when creating a new Node
Also change the names of the Widget and Node classes, as well as it's content_label_objname,
it will be used when saving the graphs.
"""

OP_NODE_EXAMPLE = get_next_opcode()


class ExampleWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()

    def create_widgets(self):
        self.checkbox = self.create_check_box("Example")
        self.line_edit = self.create_line_edit("Example")


@register_node(OP_NODE_EXAMPLE)
class TorchLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/in.png"
    op_code = OP_NODE_EXAMPLE
    op_title = "Example Node"
    content_label_objname = "example_node"
    category = "Data"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    custom_input_socket_name = ["DOG", "CAT", "42"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6,5,4,3,2,1], outputs=[6,5,4,3,2,1])

    def initInnerClasses(self):
        self.content = ExampleWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 340
        self.grNode.height = 500
        self.content.setMinimumWidth(340)
        self.content.setMinimumHeight(500)
        self.content.eval_signal.connect(self.evalImplementation)

    @QtCore.Slot()
    def evalImplementation_thread(self):
        self.busy = True

        value = None

        """
        Do your magic here, to access input nodes data, use self.getInputData(index),
        this is inherently threaded, so returning the value passes it to the onWorkerFinished function,
        it's super call makes sure we clean up and safely return.
        
        Inputs and Outputs are listed from the bottom of the node.
        """

        return value

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.setOutput(index=0, value=result)
        self.executeChild(0)
