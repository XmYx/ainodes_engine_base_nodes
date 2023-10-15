import os
import secrets
import torch
from PIL import Image

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.showone_model import VideoGenerator
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_SHOWONE = get_next_opcode()


class ShowOneWidget(QDMNodeContentWidget):
    def initUI(self):
        self.prompt = self.create_text_edit("Prompt", placeholder="Prompt or Negative Prompt (use 2x Conditioning Nodes for Stable Diffusion),\n"
                                                                  "and connect them to a K Sampler.\n"
                                                                  "If you want to control your resolution,\n"
                                                                  "or use an init image, use an Empty Latent Node.")
        self.frames = self.create_spin_box("Frames", min_val=1, max_val=4096, default_val=8)
        self.width_value = self.create_spin_box("Width", min_val=256, max_val=4096, default_val=512, step=8)
        self.height_value = self.create_spin_box("Height", min_val=256, max_val=4096, default_val=384, step=8)
        self.create_main_layout(grid=1)

@register_node(OP_NODE_SHOWONE)
class ShowOneNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Show-1 Sampler"
    op_code = OP_NODE_SHOWONE
    op_title = "Show-1 Sampler"
    content_label_objname = "showone_node"
    category = "aiNodes Base/Show-1"
    NodeContent_class = ShowOneWidget
    dim = (340, 460)
    output_data_ports = [0]
    custom_input_socket_name = ["IMAGE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])
        self.pipe = None
    def evalImplementation_thread(self, index=0):

        if self.pipe == None:
            self.pipe = VideoGenerator()

        video = self.pipe(prompt=self.content.prompt.toPlainText(),
                          frames=self.content.frames.value(),
                          width=safe_divide_by_8(self.content.width_value.value // 8),
                          height=safe_divide_by_8(self.content.height_value.value // 8),
                          )

        images = tensor2vid(video)
        images = [Image.fromarray(image) for image in images]
        images = torch.stack([pil2tensor(image)[0] for image in images], dim=0)


        return [images]


    def remove(self):
        super().remove()


def safe_divide_by_8(n):
    return int((n // 8) * 8)