from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_UNINODE = get_next_opcode()
OP_NODE_PRIMITIVE = get_next_opcode()
class UniNodeWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)
class PrimitiveNodeWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_UNINODE)
class UniNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Anything - to - Anything"
    op_code = OP_NODE_UNINODE
    op_title = "UniNode"
    content_label_objname = "uninode"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = UniNodeWidget
    custom_input_socket_name = ["ANYTHING", "EXEC"]
    custom_output_socket_name = ["ANYTHING", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[7,1], outputs=[7,1])
        self.input_multi_edged = True

    def evalImplementation_thread(self, index=0):
        return [None]

    def getOutput(self, index=0, origin_index=0):


        check = None
        for edge in self.outputs[index].edges:
            if edge.getOtherSocket(origin_index).index == origin_index:
                check = {
                    "name":edge.getOtherSocket(self.outputs[index]).name.lower(),
                    "index":edge.getOtherSocket(self.outputs[index]).index,
                    "socket_type":edge.getOtherSocket(self.outputs[index]).socket_type,
                }
        print("check", check)
        if check:
            for edge in self.inputs[index].edges:
                if edge.getOtherSocket(origin_index).name.lower() == check["name"]:
                    i = edge.getOtherSocket(origin_index).index
                    node = edge.getOtherSocket(origin_index).node
                    if node != self:
                        data = node.getOutput(i)
                        return data
        return None


@register_node(OP_NODE_PRIMITIVE)
class PrimitiveNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Primitive Node"
    op_code = OP_NODE_PRIMITIVE
    op_title = "PrimitiveNode"
    content_label_objname = "primitivenode"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = PrimitiveNodeWidget

    def __init__(self, scene):
        super().__init__(scene, inputs=[7,1], outputs=[7,1])
        self.input_multi_edged = True

    def evalImplementation_thread(self, index=0):
        return [None]

