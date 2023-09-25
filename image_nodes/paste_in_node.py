from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, \
    pil2tensor
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_IMAGE_PASTE = get_next_opcode()
OP_NODE_IMAGE_CROP = get_next_opcode()



class ImagePasteWidget(QDMNodeContentWidget):
    def initUI(self):
        self.scale_value = self.create_double_spin_box("Scale", min_val=0.01, max_val=10.0, default_val=1.0, step=0.1)
        self.create_main_layout(grid=1)
class ImageCropWidget(QDMNodeContentWidget):
    def initUI(self):
        self.top = self.create_spin_box("Top", min_val=0, max_val=4096, default_val=0, step=1)
        self.left = self.create_spin_box("Left", min_val=0, max_val=4096, default_val=0, step=1)
        self.bottom = self.create_spin_box("Bottom", min_val=0, max_val=4096, default_val=0, step=1)
        self.right = self.create_spin_box("Right", min_val=0, max_val=4096, default_val=0, step=1)
        self.create_main_layout(grid=1)


@register_node(OP_NODE_IMAGE_PASTE)
class ImagePasteNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_IMAGE_PASTE
    op_title = "Paste Image"
    content_label_objname = "imagepaste_node"
    category = "aiNodes Base/Image"
    NodeContent_class = ImagePasteWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1

    #custom_input_socket_name = ["LOGO_IMAGE", "TARGET_IMAGE"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])

    def evalImplementation_thread(self, index=0):

        result = [None]
        self.busy = True


        pixmap1 = self.getInputData(0)
        pixmap2 = self.getInputData(1)

        if pixmap1 and pixmap2:
            img1 = tensor2pil(pixmap1[0])
            img2 = tensor2pil(pixmap2[0])
            width, height = img2.size
            width = int(width * self.content.scale_value.value())
            height = int(height * self.content.scale_value.value())
            result = paste_image_center(img1, img2, width, height)
            print(result)
            result = [pil2tensor(result)]

        return [result]

@register_node(OP_NODE_IMAGE_CROP)
class ImageCropNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_IMAGE_CROP
    op_title = "Crop Image"
    content_label_objname = "imagecrop_node"
    category = "aiNodes Base/Image"
    NodeContent_class = ImageCropWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1

    #custom_input_socket_name = ["LOGO_IMAGE", "TARGET_IMAGE"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])

    def evalImplementation_thread(self, index=0):

        result = [None]
        self.busy = True


        pixmap1 = self.getInputData(0)

        if pixmap1:
            img1 = tensor2pil(pixmap1[0])

            top = self.content.top.value()
            left = self.content.left.value()
            bottom = self.content.bottom.value()
            right = self.content.right.value()

            result = crop_image(img1, top, left, bottom, right)
            print(result)
            result = [pil2tensor(result)]

        return [result]



def paste_image_center(img1, img2, width, height):
    # Resize second image
    img2_resized = img2.resize((width, height))

    # Calculate the position to paste, which is the center of img1
    paste_position = ((img1.width - img2_resized.width) // 2, (img1.height - img2_resized.height) // 2)

    # Paste img2_resized into img1 at the calculated position
    img1.paste(img2_resized, paste_position)

    # Return the result image
    return img1

def crop_image(img, upper_crop, left_crop, lower_crop, right_crop):
    """Crop a PIL image by a certain amount from each side.

    Args:
        img (PIL Image): Image to be cropped.
        left_crop (int): Amount of pixels to crop from the left side.
        upper_crop (int): Amount of pixels to crop from the upper side.
        right_crop (int): Amount of pixels to crop from the right side.
        lower_crop (int): Amount of pixels to crop from the lower side.

    Returns:
        PIL Image: Cropped image.
    """
    width, height = img.size
    left = left_crop
    upper = upper_crop
    right = width - right_crop
    lower = height - lower_crop

    cropped_image = img.crop((left, upper, right, lower))
    return cropped_image