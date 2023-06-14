from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap, pixmap_to_pil_image, torch_gc, \
    get_torch_device
from diffusers import StableUnCLIPImg2ImgPipeline, UnCLIPImageVariationPipeline
from diffusers.utils import load_image
import torch


#Function imports
import qrcode
from PIL import Image

#MANDATORY
OP_NODE_DIFF_KARLO_UNCLIP = get_next_opcode()

#NODE WIDGET
class DiffusersKarloUnclipWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

#NODE CLASS
@register_node(OP_NODE_DIFF_KARLO_UNCLIP)
class DiffusersKarloUnclipNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - "
    op_code = OP_NODE_DIFF_KARLO_UNCLIP
    op_title = "Diffusers - Karlo UnClip Node"
    content_label_objname = "diffusers_karlo_unclip_node"
    category = "Diffusers"
    NodeContent_class = DiffusersKarloUnclipWidget
    dim = (340, 260)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        self.pipe = None

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):
        device = get_torch_device()
        images = self.getInputData(0)
        return_pixmaps = []
        for image in images:
            pil_image = pixmap_to_pil_image(image)
            if not self.pipe:
                self.pipe = UnCLIPImageVariationPipeline.from_pretrained("fusing/karlo-image-variations-diffusers")
                self.pipe = self.pipe.to(device)
            generator = torch.Generator(device).manual_seed(420)


            image = self.pipe(pil_image, decoder_num_inference_steps=30, super_res_num_inference_steps=25, generator=generator).images[0]
            return_pixmaps.append(pil_image_to_pixmap(image))
        return [return_pixmaps]

    def remove(self):
        print("REMOVING", self)
        if self.pipe:
            self.pipe.to("cpu")
            del self.pipe
            torch_gc()

        super().remove()


















