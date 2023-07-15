from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, \
    pil2tensor
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_IMAGE_PASTE = get_next_opcode()
class ImagePasteWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_IMAGE_PASTE)
class DataMergeNode(AiNode):
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

            result = paste_image_center(img1, img2, 192, 192)
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