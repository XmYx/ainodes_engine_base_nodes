import math

import numpy as np
import torch

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_pil_image

OP_NODE_ENCODE_INPAINT = get_next_opcode()
class InpaintEncodeWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout()

@register_node(OP_NODE_ENCODE_INPAINT)
class InpaintEncodeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_ENCODE_INPAINT
    op_title = "Encode for Inpaint"
    content_label_objname = "inpaint_encode_node"
    category = "Experimental"
    NodeContent_class = InpaintEncodeWidget
    dim = (340, 180)
    output_data_ports = [0, 1]
    exec_port = 2

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[2,6,1])

    def evalImplementation_thread(self, index=0):

        masks = self.getInputData(0)
        images = self.getInputData(1)

        image = pixmap_to_pil_image(images[0])
        mask = pixmap_to_pil_image(masks[0])

        latent, noise_mask = self.encode(image, mask)

        data = {"noise_mask":noise_mask}

        return [[latent], data]



    def encode(self, pixels, mask, grow_mask_by=6):

        pixels = torch.from_numpy(np.array(pixels.convert("RGB"))[None].astype(np.uint8)).to(dtype=torch.float32)

        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8

        mask = torch.from_numpy(np.array(mask.convert("RGB")).astype(np.uint8)).to(dtype=torch.float32)

        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        # grow mask by a few pixels to keep things seamless in latent space
        if grow_mask_by == 0:
            mask_erosion = mask
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)

        print(m[None].shape)
        print(pixels.shape)

        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m[-1]
            pixels[:, :, :, i] += 0.5
        t = gs.models["vae"].encode(pixels)

        return t, (mask_erosion[:, :, :x, :y].round())