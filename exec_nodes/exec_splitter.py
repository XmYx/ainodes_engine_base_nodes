import time

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode

OP_NODE_EXEC_SPLITTER = get_next_opcode()

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
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 160
        self.grNode.width = 256
        self.output_socket_name = ["EXEC_1", "EXEC_2"]
    def evalImplementation(self, index=0, *args, **kwargs):
        self.markDirty(True)
        self.markInvalid(True)
        self.busy = False
        self.executeChild(1)
        self.executeChild(0)
        return None
    def onMarkedDirty(self):
        self.value = None










