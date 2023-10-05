import os, sys
from types import SimpleNamespace

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap, tensor2pil, \
    pil2tensor
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.CodeFormer.model import run_codeformer

OP_NODE_CODEFORMER = get_next_opcode()

class CodeFormerWidget(QDMNodeContentWidget):
    def initUI(self):

        self.fidelity = self.create_double_spin_box("Fidelity Weight:", min_val=0.0, max_val=1.0, default_val=0.5, step=0.1)
        self.upscale = self.create_spin_box("Upscale x:", min_val=1, max_val=16, default_val=2)
        self.bg_upsample = self.create_check_box("Bg Upsampling")
        self.face_upsample = self.create_check_box("Face Upsample")

        self.create_main_layout(grid=1)

@register_node(OP_NODE_CODEFORMER)
class CodeFormerNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/codeformers.png"
    op_code = OP_NODE_CODEFORMER
    op_title = "Codeformer"
    content_label_objname = "codeformer_node"
    category = "aiNodes Base/Upscalers"
    NodeContent_class = CodeFormerWidget
    dim = (340, 260)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])


    def evalImplementation_thread(self, index=0):
        self.busy = True
        image = self.getInputData(0)
        pil_img = tensor2pil(image[0])
        bg_up = self.content.bg_upsample.isChecked()
        bg_up = 'realesrgan' if bg_up else 'None'
        print(bg_up)
        args = SimpleNamespace(
            input_path='./inputs/whole_imgs',
            output_path=None,
            fidelity_weight=self.content.fidelity.value(),
            upscale=self.content.upscale.value(),
            has_aligned=False,
            only_center_face=False,
            draw_box=False,
            detection_model='retinaface_resnet50',
            bg_upsampler=bg_up,
            face_upsample=self.content.face_upsample.isChecked(),
            bg_tile=400,
            suffix=None,
            save_video_fps=None
        )
        #print(args)
        result = run_codeformer(args, [pil_img])
        tensor = pil2tensor(result[0])
        return [tensor]

