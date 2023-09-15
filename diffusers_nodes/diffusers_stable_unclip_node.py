from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, torch_gc, \
    tensor2pil, pil2tensor
from diffusers import StableUnCLIPImg2ImgPipeline
import torch

#MANDATORY
OP_NODE_DIFF_UNCLIP = get_next_opcode()

#NODE WIDGET
class DiffusersUnclipWidget(QDMNodeContentWidget):
    def initUI(self):

        self.prompt = self.create_text_edit("Prompt", placeholder="Optional prompt field")
        self.eta = self.create_double_spin_box("ETA", min_val=0.0, max_val=1.0, default_val=0.0, step=0.01)
        self.noise_level = self.create_spin_box("NOISE LEVEL", min_val=0, max_val=100, default_val=0)
        self.steps = self.create_spin_box("STEPS", min_val=0, max_val=1000, default_val=25)
        self.create_main_layout(grid=1)

#NODE CLASS
@register_node(OP_NODE_DIFF_UNCLIP)
class DiffusersUnclipNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers - "
    op_code = OP_NODE_DIFF_UNCLIP
    op_title = "Diffusers - Stable UnClip Node"
    content_label_objname = "diffusers_stable_unclip_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersUnclipWidget
    dim = (340, 500)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        self.pipe = None
        self.grNode.height = 600
        self.update_all_sockets()

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):
        images = self.getInputData(0)
        return_pixmaps = []
        for image in images:
            pil_image = tensor2pil(image)
            if not self.pipe:
                self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
                )
                self.pipe = self.pipe.to("cuda")
            generator = torch.Generator("cuda").manual_seed(420)

            prompt = self.content.prompt.toPlainText()
            height = pil_image.size[0]
            width = pil_image.size[1]
            eta = self.content.eta.value()
            noise_level = self.content.noise_level.value()
            steps = self.content.steps.value()

            image = self.pipe(pil_image,
                              prompt=prompt,
                              height=height,
                              width=width,
                              num_inference_steps=steps,
                              eta=eta,
                              callback=self.callback,
                              noise_level=noise_level,
                              generator=generator
                              ).images[0]
            return_pixmaps.append(pil2tensor(image))
        return [return_pixmaps]
    def callback(self, i, j, tensor):
        print(i, j)
    def remove(self):
        if self.pipe:
            self.pipe.to("cpu")
            del self.pipe
            torch_gc()
        super().remove()














