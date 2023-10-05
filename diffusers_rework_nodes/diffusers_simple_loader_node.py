import torch
import os

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
from ainodes_frontend import singleton as gs

class DiffSDSimplePipelineWidget(QDMNodeContentWidget):
    def initUI(self):

        self.create_combo_box([item["name"] for item in diffusers_models], "Model", spawn="models")
        self.create_check_box("XL", spawn="xl")
        self.create_check_box("TinyVAE", spawn="tinyvae")

        self.create_check_box("Use Local Model", spawn="use_local_models")
        checkpoint_folder = gs.prefs.checkpoints

        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith(('.ckpt', '.pt', '.bin', '.pth', '.safetensors'))]
        self.local_model = self.create_combo_box(checkpoint_files, "Model:")
        if checkpoint_files == []:
            self.local_model.addItem("Please place a model in models/checkpoints_xl")
            print(f"TORCH LOADER NODE: No model file found at {os.getcwd()}/models/checkpoints,")
            print(f"TORCH LOADER NODE: please download your favorite ckpt before Evaluating this node.")


        self.create_combo_box(["txt2img", "img2img", "txt2img_cnet", "img2img_cnet", "txt2img_xl", "txt2img_xl_cnet", "img2img_xl"], "pipeline", spawn="pipe_select")
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

        pipe_select = self.content.pipe_select.currentText()

        pipes = {"txt2img_xl":StableDiffusionXLPipeline,
                 "img2img_xl":StableDiffusionXLImg2ImgPipeline}

        pipe_class = pipes.get(pipe_select, StableDiffusionXLPipeline)

        if not self.content.use_local_models.isChecked():
            self.pipe = pipe_class.from_pretrained(model_name, torch_dtype=torch.float16)
        else:
            model_name = f"{gs.prefs.checkpoints}/{self.content.local_model.currentText()}"
            self.pipe = pipe_class.from_single_file(model_name, torch_dtype=torch.float16)

        if isinstance(self.pipe, StableDiffusionXLPipeline):

            print("Adding custom Generate call to Diffusers Stable Diffusion XL Pipeline")

            from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.diffusers_xl_call import new_call

            def replace_call(pipe, new_call):
                def call_with_self(*args, **kwargs):
                    return new_call(pipe, *args, **kwargs)

                return call_with_self

            self.pipe.generate = replace_call(self.pipe, new_call)

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