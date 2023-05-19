import time

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_EXEC_SPLITTER = get_next_opcode()

class ExecSplitterWidget(QDMNodeContentWidget):
    def initUI(self):
        pass

@register_node(OP_NODE_EXEC_SPLITTER)
class ExecSplitterNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/exec_split.png"
    op_code = OP_NODE_EXEC_SPLITTER
    op_title = "Execute Splitter"
    content_label_objname = "exec_splitter_node"
    category = "Experimental"
    help_text = "Execution Splitter Node\n\n" \
                "You can split your processes into\n" \
                "as many branches as you want\n" \
                "given you have the CPU/GPU\n" \
                "resources."
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1,1])
        pass
        # Create a worker object
    def initInnerClasses(self):
        self.content = ExecSplitterWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 160
        self.grNode.width = 256
        self.output_socket_name = ["EXEC_1", "EXEC_2"]
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0, *args, **kwargs):
        return None

    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.executeChild(1)
        self.executeChild(0)










