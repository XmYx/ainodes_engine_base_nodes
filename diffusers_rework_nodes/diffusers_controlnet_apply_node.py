from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, torch_gc, \
    tensor2pil
from diffusers import StableUnCLIPImg2ImgPipeline, UnCLIPImageVariationPipeline
from diffusers.utils import load_image
import torch


#Function imports
import qrcode
from PIL import Image

#MANDATORY
OP_NODE_DIFF_CONTROLNET_APPLY = get_next_opcode()


#NODE WIDGET
class DiffusersControlNetApplyWidget(QDMNodeContentWidget):
    def initUI(self):

        self.scale_value = self.create_double_spin_box(label_text="Control Strength", min_val=0.0, max_val=10.0, default_val=1.0, step=0.01)
        self.start_value = self.create_double_spin_box("Start", min_val=0, max_val=1.0, default_val=0.0)
        self.stop_value = self.create_double_spin_box("Stop", min_val=0, max_val=1.0, default_val=1.0)

        self.create_main_layout(grid=1)


#NODE CLASS
@register_node(OP_NODE_DIFF_CONTROLNET_APPLY)
class DiffusersControlNetApplyNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - "
    op_code = OP_NODE_DIFF_CONTROLNET_APPLY
    op_title = "Diffusers - ControlNet Apply"
    content_label_objname = "diffusers_controlnet_apply_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersControlNetApplyWidget
    dim = (340, 340)
    output_data_ports = [0]
    exec_port = 1

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[6,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):


        image = self.getInputData(0)
        data = self.getInputData(1)

        if image is not None:
            image = tensor2pil(image)
        scale = self.content.scale_value.value()
        start = self.content.start_value.value()
        stop = self.content.stop_value.value()

        if data is not None:
            if "controlnet_conditioning_scale" in data:
                if data["controlnet_conditioning_scale"] is not None:
                    data["controlnet_conditioning_scale"].append(scale)
                    data["guess_mode"].append(False)
                    data["control_guidance_start"].append(start)
                    data["control_guidance_end"].append(stop)
                else:
                    data["controlnet_conditioning_scale"] = [scale]
                    data["guess_mode"] = [False]
                    data["control_guidance_start"] = [start]
                    data["control_guidance_end"] = [stop]
            if "image" in data:
                if isinstance(data["image"], list):
                    data["image"].append(image)
                else:
                    data["image"] = [image]
            else:
                data["image"] = [image]

            if len(data["controlnet_conditioning_scale"]) == 1:
                data["image"] = [image]

        else:
            data = {
                "image":[image],
                "controlnet_conditioning_scale": [scale],
                "guess_mode": [False],
                "control_guidance_start": [start],
                "control_guidance_end": [stop]
            }

        return [data]




    def remove(self):
        super().remove()


















