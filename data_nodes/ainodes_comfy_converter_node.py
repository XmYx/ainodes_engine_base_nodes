from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_CONVERT = get_next_opcode()
class ConvertWidget(QDMNodeContentWidget):
    def initUI(self):
        self.select_direction = self.create_combo_box(["aiNodes-Comfy", "Comfy-aiNodes"], "direction")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_CONVERT)
class ConvertNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_CONVERT
    op_title = "aiImage-2-ComfyImage"
    content_label_objname = "image_convert_node"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = ConvertWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])

    def evalImplementation_thread(self, index=0):
        images = self.getInputData(0)
        direction = self.content.select_direction.currentText()

        if direction == "aiNodes-Comfy":
            return [images[0]]
        else:
            return [[images]]