import math

import torch

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, \
    pil2tensor
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_MASK_CONVERT = get_next_opcode()
OP_NODE_MASK_ENCODE = get_next_opcode()
class MaskConvertWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_MASK_CONVERT)
class MaskConvertNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_MASK_CONVERT
    op_title = "Convert Mask/Image"
    content_label_objname = "mask_convert_node"
    category = "aiNodes Base/Image"
    NodeContent_class = MaskConvertWidget
    dim = (340, 180)
    make_dirty = True
    custom_input_socket_name = ["MASK", "IMAGE", "EXEC"]
    custom_output_socket_name = ["MASK", "IMAGE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,5,5,1], outputs=[5,5,1])

    def evalImplementation_thread(self, index=0):

        mask = self.getInputData(0)
        image = self.getInputData(1)

        if mask is not None:
            mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        return [image, mask]

@register_node(OP_NODE_MASK_ENCODE)
class MaskEncodeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_MASK_ENCODE
    op_title = "Encode Mask/Image"
    content_label_objname = "mask_encode_node"
    category = "aiNodes Base/Image"
    NodeContent_class = MaskConvertWidget
    dim = (340, 180)
    make_dirty = True
    custom_input_socket_name = ["VAE","MASK", "IMAGE", "EXEC"]
    custom_output_socket_name = ["MASK", "IMAGE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[4,5,5,1], outputs=[2,1])

    def evalImplementation_thread(self, index=0):
        vae = self.getInputData(0)
        mask = self.getInputData(1)
        image = self.getInputData(2)

        def process_pixels_and_mask(pixels, mask, grow_mask_by=0):

            print(pixels.shape)
            print(mask.shape)
            mask_erosion = None
            # Ensure mask has the right dimensions
            mask = mask.unsqueeze(-1)  # Adding channel dimension to the mask

            # Adjust the dimensions of 'pixels' and 'mask' if they are not multiples of 8
            x = (pixels.shape[1] // 8) * 8
            y = (pixels.shape[2] // 8) * 8

            pixels = pixels.clone()
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, x_offset:x + x_offset, y_offset:y + y_offset, :]

            # Grow the mask by a few pixels to keep things seamless in latent space
            if grow_mask_by != 0:
                kernel_tensor = torch.ones((1, grow_mask_by, grow_mask_by, 1))
                padding = math.ceil((grow_mask_by - 1) / 2)
                mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.permute(0, 3, 1, 2).round(),
                                                                      kernel_tensor.permute(3, 2, 0, 1),
                                                                      padding=padding), 0, 1)
                m = (1.0 - mask_erosion.round()).squeeze(1)
            else:
                m = (1.0 - mask.round()).squeeze(-1)

            # Process the pixels tensor using the mask
            for i in range(3):  # Assuming 3 channels (RGB)
                pixels[..., i] -= 0.5
                pixels[..., i] *= m
                pixels[..., i] += 0.5

            return pixels, mask, mask_erosion, x, y
        if mask is not None and image is not None:
            pixels, mask, mask_erosion, x, y = process_pixels_and_mask(image, mask)
            t = vae.encode(pixels)
            return [{"samples":t, "noise_mask": mask}]



