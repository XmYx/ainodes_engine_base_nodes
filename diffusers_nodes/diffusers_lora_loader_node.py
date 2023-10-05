import os

import torch
from safetensors.torch import load_file

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from diffusers import StableDiffusionPipeline

#MANDATORY
OP_NODE_DIFF_LOADLORA = get_next_opcode()
OP_NODE_DIFF_FUSELORA = get_next_opcode()
OP_NODE_DIFF_UNFUSELORA = get_next_opcode()
OP_NODE_DIFF_UNLOADLORA = get_next_opcode()

from ainodes_frontend import singleton as gs

#NODE WIDGET
class DiffusersLoraWidget(QDMNodeContentWidget):
    def initUI(self):

        lora_folder = gs.prefs.loras
        lora_files = [f for f in os.listdir(lora_folder) if f.endswith(('.safetensors', '.ckpt', '.pt', '.bin', '.pth'))]
        if lora_files == []:
            self.dropdown.addItem("Please place a lora in models/loras")
            print(f"LORA LOADER NODE: No model file found at {os.getcwd()}/models/loras,")
            print(f"LORA LOADER NODE: please download your favorite ckpt before Evaluating this node.")
        self.dropdown = self.create_combo_box(lora_files, "Lora")

        self.create_main_layout(grid=1)
class DiffusersEmptyWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)
class DiffusersEmptyWidget2(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)


#NODE CLASS
@register_node(OP_NODE_DIFF_LOADLORA)
class DiffusersLoraLoaderNode(AiNode):
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
    make_dirty = True

    custom_input_socket_name = ["PIPE", "EXEC"]
    custom_output_socket_name = ["PIPE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)

        assert pipe is not None, "No Pipe found"

        lora_path = self.content.dropdown.currentText()
        pipe.load_lora_weights("models/loras", weight_name=lora_path, local_files_only=True)
        return [pipe]

    def load_lora(self, pipe):
        from .diffusers_lora_loader import install_lora_hook
        install_lora_hook(pipe)
        lora_path = self.content.dropdown.currentText()
        lora1 = pipe.apply_lora(f"models/loras/{lora_path}", alpha=0.8)
        return pipe

    def remove(self):
        super().remove()

@register_node(OP_NODE_DIFF_FUSELORA)
class DiffusersFuseLoraNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - Fuse Lora"
    op_code = OP_NODE_DIFF_FUSELORA
    op_title = "Diffusers - Fuse LORA"
    content_label_objname = "diffusers_fuselora_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersEmptyWidget
    dim = (340, 340)
    output_data_ports = [0]
    exec_port = 1
    make_dirty = True

    custom_input_socket_name = ["PIPE", "EXEC"]
    custom_output_socket_name = ["PIPE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)

        try:
            pipe.fuse_lora()
        except:
            pass
        return [pipe]

    def remove(self):
        super().remove()


@register_node(OP_NODE_DIFF_UNFUSELORA)
class DiffusersUnFuseLoraNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - Unfuse Lora"
    op_code = OP_NODE_DIFF_UNFUSELORA
    op_title = "Diffusers - Unfuse LORA"
    content_label_objname = "diffusers_unfuselora_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersEmptyWidget2
    dim = (340, 340)
    output_data_ports = [0]
    exec_port = 1

    make_dirty = True

    custom_input_socket_name = ["PIPE", "EXEC"]
    custom_output_socket_name = ["PIPE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)

        try:
            pipe.unfuse_lora()
        except:
            pass
        return [pipe]

    def remove(self):
        super().remove()

@register_node(OP_NODE_DIFF_UNLOADLORA)
class DiffusersUnloadLoraNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - Unload Lora"
    op_code = OP_NODE_DIFF_UNLOADLORA
    op_title = "Diffusers - Unload LORA"
    content_label_objname = "diffusers_unfuselora_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersEmptyWidget2
    dim = (340, 340)
    output_data_ports = [0]
    exec_port = 1

    make_dirty = True

    custom_input_socket_name = ["PIPE", "EXEC"]
    custom_output_socket_name = ["PIPE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,1], outputs=[4,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        pipe = self.getInputData(0)

        try:
            pipe.unload_lora_weights()
        except:
            pass
        return [pipe]

    def remove(self):
        super().remove()

def load_lora_weights_(pipeline, checkpoint_path):
    # load base model
    pipeline.to("cuda")
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    alpha = 0.75
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device="cuda")
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline
















