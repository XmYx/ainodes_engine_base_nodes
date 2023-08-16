import numpy as np
from PIL import Image

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, \
    pil2tensor
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_IMAGE_PREP_OUTPAINT = get_next_opcode()
class ImagePrepOutpaintWidget(QDMNodeContentWidget):
    def initUI(self):
        self.width_value = self.create_spin_box("Width", min_val=256, max_val=4096, default_val=1024, step=8)
        self.height_value = self.create_spin_box("Height", min_val=256, max_val=4096, default_val=1024, step=8)
        self.offset_x = self.create_spin_box("Offset X", min_val=0, max_val=4096, default_val=0, step=1)
        self.offset_y = self.create_spin_box("Offset Y", min_val=0, max_val=4096, default_val=0, step=1)
        self.scale_value = self.create_double_spin_box("Scale", min_val=0.1, max_val=10.0, step=0.01, default_val=0.5)
        self.create_main_layout(grid=1)


@register_node(OP_NODE_IMAGE_PREP_OUTPAINT)
class ImagePasteNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_IMAGE_PREP_OUTPAINT
    op_title = "Prepare Outpaint Image"
    content_label_objname = "image_prep_outpaint_node"
    category = "aiNodes Base/Image"
    NodeContent_class = ImagePrepOutpaintWidget
    dim = (340, 180)
    output_data_ports = [0, 1]
    exec_port = 2

    #custom_input_socket_name = ["LOGO_IMAGE", "TARGET_IMAGE"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,5,1])

    def evalImplementation_thread(self, index=0):
        result = None
        tensor = self.getInputData(0)
        if tensor is not None:
            image = tensor2pil(tensor[0])

            w = self.content.width_value.value()
            h = self.content.height_value.value()
            target_size = (w, h)
            scale = self.content.scale_value.value()
            offset_x = self.content.offset_x.value()
            offset_y = self.content.offset_y.value()

            new_img, mask = scale_and_paste(target_size, image, scale, offset_x, offset_y)
            img_tensor = pil2tensor(new_img)
            mask_tensor = pil2tensor(mask)

            return [[img_tensor], [mask_tensor]]
        else:
            return [None, None]


def scale_and_paste(target_size, source_img, scale, offset_x, offset_y, expand=8):
    # Resize the source image
    source_img = source_img.resize((int(source_img.width * scale), int(source_img.height * scale)))

    # Create a new image of the target size
    new_img = Image.new('RGB', (target_size[0], target_size[1]))

    # Paste the scaled source image onto the new image
    new_img.paste(source_img, (offset_x, offset_y))

    # Create a mask of the empty area
    mask = np.full((new_img.height, new_img.width), 255)
    mask[offset_y+expand:(source_img.height-expand+offset_y), offset_x+expand:(source_img.width-expand+offset_x)] = 0
    mask = Image.fromarray(mask).convert("RGB")

    print(new_img.size, mask.size)

    return new_img, mask
