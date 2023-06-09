import PIL.Image
import numpy as np
import torch
from PIL import ImageOps

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_pil_image
from ainodes_frontend import singleton as gs

OP_NODE_COND_MASK = get_next_opcode()
class CondMaskWidget(QDMNodeContentWidget):
    def initUI(self):
        self.strength = self.create_double_spin_box("Strength")
        self.create_main_layout()

@register_node(OP_NODE_COND_MASK)
class CondMaskNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/cond_masking.png"
    op_code = OP_NODE_COND_MASK
    op_title = "Conditioning Mask"
    content_label_objname = "cond_apply_mask"
    category = "Conditioning"
    NodeContent_class = CondMaskWidget
    #dim = (340, 160)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,3,5,1], outputs=[3,1])

    def initInnerClasses(self):
        super().initInnerClasses()
        width = 340
        height = 260
        self.grNode.width = width
        self.grNode.height = height
        self.content.setMinimumHeight(height)
        self.content.setMinimumWidth(width)


    def evalImplementation_thread(self, index=0):

        data = self.getInputData(0)
        conds = self.getInputData(1)
        images = self.getInputData(2)

        if images:
            i = pixmap_to_pil_image(images[0])
            i = i.resize((i.size[0] // 8, i.size[1] // 8), resample=PIL.Image.Resampling.LANCZOS)
            i = ImageOps.exif_transpose(i)

            if i.mode == 'RGBA':
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                print("ALPHAMASK", mask.shape)
            elif i.mode == 'RGB' or i.mode == 'P':
                # if the image is RGB or P, convert it to greyscale and use as alpha channel
                i = i.convert('L')  # convert image to greyscale
                mask = np.array(i).astype(np.float32) / 255.0  # normalize the image data to 0 - 1
                print("GREYSCALE", mask.shape)

            else:
                raise ValueError("Unsupported image mode")
            mask = 1. - torch.from_numpy(mask).to("cuda")
            set_area_to_bounds = True
            strength = self.content.strength.value()
            c = self.append(conds[0], mask, set_area_to_bounds, strength)
            uc = {}


            return [[c]]
        else:
            print("No valid input image found")
            return [[conds]]


    def append(self, conditioning, mask, set_area_to_bounds, strength):
        c = []
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            _, h, w = mask.shape
            n[1]['mask'] = mask
            n[1]['set_area_to_bounds'] = set_area_to_bounds
            n[1]['mask_strength'] = strength
            c.append(n)
        return c