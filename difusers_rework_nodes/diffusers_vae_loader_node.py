import torch

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import diffusers_models, diffusers_indexed
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_VAE = get_next_opcode()

class DiffVAEWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_combo_box([item["name"] for item in diffusers_models], "Model", spawn="models")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_VAE)
class DiffTENode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers VAE Loader"
    op_code = OP_NODE_DIFF_VAE
    op_title = "Diffusers - VAE Loader"
    content_label_objname = "diff_vae_node"
    category = "aiNodes Base/Diffusers/Loaders"
    NodeContent_class = DiffVAEWidget
    dim = (340, 240)
    output_data_ports = [0]
    custom_output_socket_name = ["VAE", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,1])
        self.model = None
        self.loaded_model = ""
    def evalImplementation_thread(self, index=0):
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]
        istiny = False
        variant = f"{str(istiny)}_{model_name}"
        if self.model == None or self.loaded_model != variant:
            from diffusers import AutoencoderKL, AutoencoderTiny
            model_class = AutoencoderKL
            tiny_model = "madebyollin/taesd" if "xl" not in model_name.lower() else "madebyollin/taesdxl"

            print("loading", tiny_model)

            model_name = model_name if not istiny else tiny_model
            subfolder = "vae" if not istiny else None
            self.model = model_class.from_pretrained(model_name, subfolder=subfolder, torch_dtype=torch.float16)
            self.model.enable_slicing()
            self.model.enable_tiling()
            self.loaded_model = variant

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