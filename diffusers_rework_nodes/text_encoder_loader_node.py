import torch

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import diffusers_models, diffusers_indexed
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_TE = get_next_opcode()

class DiffTEWidget(QDMNodeContentWidget):
    def initUI(self):
        self.models = self.create_combo_box([item["name"] for item in diffusers_models], "Model")
        self.create_check_box("XL (ClipVision)", spawn="vision")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_TE)
class DiffTENode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers Unet Loader"
    op_code = OP_NODE_DIFF_TE
    op_title = "Diffusers - Clip Text Model"
    content_label_objname = "diff_te_node"
    category = "aiNodes Base/Diffusers/Loaders"
    NodeContent_class = DiffTEWidget
    dim = (340, 240)
    output_data_ports = [0]
    # exec_port = 4
    custom_output_socket_name = ["TEXT ENCODER", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,1])
        self.model = None
        self.loaded_model = ""
    def evalImplementation_thread(self, index=0):
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]
        vision = self.content.vision.isChecked()
        if self.model == None or self.loaded_model != model_name:
            if not vision:
                from transformers import CLIPTextModel
                self.model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=torch.float16)
            else:
                from transformers import CLIPTextModelWithProjection
                self.model = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=torch.float16)

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