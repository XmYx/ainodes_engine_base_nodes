import torch

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import diffusers_models, diffusers_indexed
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_UNET = get_next_opcode()

class DiffUnetWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_combo_box([item["name"] for item in diffusers_models], "Model", spawn="models")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_UNET)
class DiffUnetNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers Unet Loader"
    op_code = OP_NODE_DIFF_UNET
    op_title = "Diffusers - Unet"
    content_label_objname = "diff_unet_node"
    category = "aiNodes Base/Diffusers/Loaders"
    NodeContent_class = DiffUnetWidget
    dim = (340, 240)
    output_data_ports = [0]
    # exec_port = 4
    custom_output_socket_name = ["UNET", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,1])
        self.model = None
        self.loaded_model = ""
    def evalImplementation_thread(self, index=0):
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]
        if self.model == None or self.loaded_model != model_name:
            from diffusers import UNet2DConditionModel
            self.model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", torch_dtype=torch.float16)
            self.loaded_model = model_name

        return [self.model]
    def remove(self):
        try:
            if self.model != None:
                self.model.to("cpu")
                del self.model
                self.model = None
        except:
            self.model = None
        super().remove()