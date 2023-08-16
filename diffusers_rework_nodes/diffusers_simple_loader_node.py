import torch

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import diffusers_models, diffusers_indexed, \
    scheduler_type_values, SchedulerType, get_scheduler_class
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_SIMPLE_PIPE = get_next_opcode()
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline,
                       StableDiffusionXLImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionXLControlNetPipeline, StableDiffusionXLInpaintPipeline)

class DiffSDSimplePipelineWidget(QDMNodeContentWidget):
    def initUI(self):

        self.create_combo_box([item["name"] for item in diffusers_models], "Model", spawn="models")
        self.create_check_box("XL", spawn="xl")
        self.create_check_box("TinyVAE", spawn="tinyvae")
        #self.create_combo_box(["txt2img", "img2img", "txt2img_cnet", "img2img_cnet", "txt2img_xl", "txt2img_xl_cnet", "img2img_xl"], "pipeline", spawn="pipe_select")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_SIMPLE_PIPE)
class DiffSDPipelineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers SDPipeline"
    op_code = OP_NODE_DIFF_SIMPLE_PIPE
    op_title = "Diffusers - Simple Pipeline"
    content_label_objname = "diff_simple_pipeline_node"
    category = "aiNodes Base/Diffusers/Loaders"
    NodeContent_class = DiffSDSimplePipelineWidget
    dim = (340, 460)
    output_data_ports = [0]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[4,1])
        self.pipe = None
    def evalImplementation_thread(self, index=0):


        isxl = self.content.xl.isChecked()


        # scheduler_name = self.content.schedulers.currentText()
        # scheduler = SchedulerType(scheduler_name)
        # scheduler_class = get_scheduler_class(scheduler)
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]
        # scheduler = scheduler_class.from_pretrained(model_name, subfolder="scheduler")



        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16)

        tinyvae = self.content.tinyvae.isChecked()

        if tinyvae:
            tiny_model = "madebyollin/taesd" if not isxl else "madebyollin/taesdxl"
            from diffusers import AutoencoderTiny
            self.pipe.vae = AutoencoderTiny.from_pretrained(tiny_model, torch_dtype=torch.float16)
        return [self.pipe]
    def remove(self):
        try:
            if self.pipe != None:
                self.pipe.to("cpu")
                del self.pipe
                self.pipe = None
        except:
            self.pipe = None
        super().remove()