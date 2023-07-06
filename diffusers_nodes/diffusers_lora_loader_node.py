import os

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from diffusers import StableDiffusionPipeline

#MANDATORY
OP_NODE_DIFF_LOADLORA = get_next_opcode()

#NODE WIDGET
class DiffusersLoraWidget(QDMNodeContentWidget):
    def initUI(self):

        lora_folder = gs.loras
        lora_files = [f for f in os.listdir(lora_folder) if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth'))]
        if lora_files == []:
            self.dropdown.addItem("Please place a lora in models/loras")
            print(f"LORA LOADER NODE: No model file found at {os.getcwd()}/models/loras,")
            print(f"LORA LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.dropdown = self.create_combo_box(lora_files, "Lora")

        self.create_main_layout(grid=1)


#NODE CLASS
@register_node(OP_NODE_DIFF_LOADLORA)
class DiffusersControlNetNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - Lora Loader"
    op_code = OP_NODE_DIFF_LOADLORA
    op_title = "Diffusers - LORA Loader"
    content_label_objname = "diffusers_loraloader_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersLoraWidget
    dim = (340, 340)
    output_data_ports = [0]
    exec_port = 1

    custom_input_socket_name = ["PIPE", "EXEC"]
    custom_output_socket_name = ["PIPE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)

        assert pipe is not None, "No Pipe found"
        assert isinstance(pipe, StableDiffusionPipeline), "Can only work with a diffusers pipeline"

        pipe = self.load_lora(pipe)
        return [pipe]

    def load_lora(self, pipe):
        from .diffusers_lora_loader import install_lora_hook
        install_lora_hook(pipe)
        lora_path = self.content.dropdown.currentText()
        lora1 = pipe.apply_lora(f"models/loras/{lora_path}", alpha=0.8)
        return pipe

    def remove(self):
        super().remove()


















