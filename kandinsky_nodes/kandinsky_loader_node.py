from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_KANDINSKY_LOADER = get_next_opcode()
class KandinskyLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout()

@register_node(OP_NODE_KANDINSKY_LOADER)
class KandinskyLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_KANDINSKY_LOADER
    op_title = "Kandinsky Loader"
    content_label_objname = "kandinsky_loader_node"
    category = "aiNodes Base/Kandinsky"
    NodeContent_class = KandinskyLoaderWidget
    dim = (340, 180)
    output_data_ports = [0,1]
    exec_port = 2

    custom_output_socket_name = ["PRIOR", "DECODER", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,4,1])
        self.prior = None
        self.decoder = None

    def evalImplementation_thread(self, index=0):
        from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline

        if self.prior == None:

            from diffusers import DiffusionPipeline
            import torch

            # self.prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior",
            #                                                torch_dtype=torch.float16)
            self.prior = KandinskyV22PriorPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
            )
            # self.decoder = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
            #                                              torch_dtype=torch.float16)
            self.decoder = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
                                                        torch_dtype=torch.float16)

        return [self.prior, self.decoder]
