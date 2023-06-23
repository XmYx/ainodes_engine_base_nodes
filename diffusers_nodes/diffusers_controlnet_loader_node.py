from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, torch_gc
from diffusers import StableUnCLIPImg2ImgPipeline, UnCLIPImageVariationPipeline
from diffusers.utils import load_image
import torch


#Function imports
import qrcode
from PIL import Image

#MANDATORY
OP_NODE_DIFF_CONTROLNET = get_next_opcode()

controlnets_15 = {
    "qrControl": "DionTimmer/controlnet_qrcode-control_v1p_sd15",
    "Inpaint": "lllyasviel/control_v11p_sd15_inpaint",
    "Ip2p": "lllyasviel/control_v11e_sd15_ip2p",
    "Tile": "lllyasviel/control_v11f1e_sd15_tile",
    "Shuffle": "lllyasviel/control_v11e_sd15_shuffle",
    "Softedge": "lllyasviel/control_v11p_sd15_softedge",
    "Scribble (v11P)": "lllyasviel/control_v11p_sd15_scribble",
    "Lineart Anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "Lineart (v11P)": "lllyasviel/control_v11p_sd15_lineart",
    "Seg": "lllyasviel/control_v11p_sd15_seg",
    "Normalbae (v11P)": "lllyasviel/control_v11p_sd15_normalbae",
    "Depth (F1P)": "lllyasviel/control_v11f1p_sd15_depth",
    "Mlsd": "lllyasviel/control_v11p_sd15_mlsd",
    "Canny (v11P)": "lllyasviel/control_v11p_sd15_canny",
    "Openpose (v11P)": "lllyasviel/control_v11p_sd15_openpose",
    "Depth (v11P)": "lllyasviel/control_v11p_sd15_depth",
}

controlnets_21 = {
    "Normalbae": "thibaud/controlnet-sd21-normalbae-diffusers",
    "Lineart": "thibaud/controlnet-sd21-lineart-diffusers",
    "Ade20k": "thibaud/controlnet-sd21-ade20k-diffusers",
    "Openposev2": "thibaud/controlnet-sd21-openposev2-diffusers",
    "Zoedepth": "thibaud/controlnet-sd21-zoedepth-diffusers",
    "Color": "thibaud/controlnet-sd21-color-diffusers",
    "Scribble": "thibaud/controlnet-sd21-scribble-diffusers",
    "Openpose": "thibaud/controlnet-sd21-openpose-diffusers",
    "Depth": "thibaud/controlnet-sd21-depth-diffusers",
    "Hed": "thibaud/controlnet-sd21-hed-diffusers",
    "Canny": "thibaud/controlnet-sd21-canny-diffusers",
}

#NODE WIDGET
class DiffusersControlNetWidget(QDMNodeContentWidget):
    def initUI(self):

        self.version_select = self.create_combo_box(["1.5", "2.0/2.1"], "Version")

        self.controlnet_name = self.create_combo_box(self.get_control_list(), "ControlNet")
        self.scale = self.create_double_spin_box(label_text="Control Strength", min_val=0.0, max_val=10.0, default_val=1.0, step=0.01)
        self.start_value = self.create_spin_box("Start", min_val=0, max_val=100, default_val=0)
        self.stop_value = self.create_spin_box("Stop", min_val=0, max_val=100, default_val=100)
        self.version_select.currentIndexChanged.connect(self.update_combo_box)

        self.create_main_layout(grid=1)

    def update_combo_box(self):
        self.controlnet_name.clear()
        self.controlnet_name.addItems(self.get_control_list())

    def get_control_list(self):
        ver = self.version_select.currentText()
        if ver == "1.5":
            return [str(key) for key, value in controlnets_15.items()]
        else:
            return [str(key) for key, value in controlnets_21.items()]

#NODE CLASS
@register_node(OP_NODE_DIFF_CONTROLNET)
class DiffusersControlNetNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - "
    op_code = OP_NODE_DIFF_CONTROLNET
    op_title = "Diffusers - ControlNet"
    content_label_objname = "diffusers_controlnet_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersControlNetWidget
    dim = (340, 340)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[6,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):

        controlnet_name = self.content.controlnet_name.currentText()
        ver = self.content.version_select.currentText()
        controlnet_dict = controlnets_15 if ver == "1.5" else controlnets_21

        controlnet_repo = controlnet_dict[controlnet_name]

        images = self.getInputData(0)
        print(images)

        data = self.getInputData(1)

        image = pixmap_to_tensor(images[0])
        scale = self.content.scale.value()
        start = self.content.start_value.value()
        stop = self.content.stop_value.value()


        to_insert = {
                    "name":controlnet_repo,
                    "image":image,
                    "scale":scale,
                    "start":start,
                    "stop":stop
                }

        if data:
            if "control_diff" in data:
                data["control_diff"].append(to_insert)
            else:
                data["control_diff"] = [to_insert]
        else:
            data = {"control_diff":[to_insert]}

        print(data)
        return [data]


    def remove(self):
        super().remove()


















