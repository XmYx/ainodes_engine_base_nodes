import torch

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import diffusers_models, diffusers_indexed, \
    scheduler_type_values, SchedulerType, get_scheduler_class
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DIFF_SDPIPE = get_next_opcode()
from diffusers import (StableDiffusionPipeline, StableDiffusionXLPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline,
                       StableDiffusionXLImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionXLControlNetPipeline)

pipes = {"txt2img":StableDiffusionPipeline,
         "img2img":StableDiffusionImg2ImgPipeline,
         "txt2img_xl":StableDiffusionXLPipeline,
         "img2img_xl":StableDiffusionXLImg2ImgPipeline,
         "txt2img_cnet_xl":StableDiffusionXLControlNetPipeline}

class DiffSDPipelineWidget(QDMNodeContentWidget):
    def initUI(self):

        self.create_combo_box([item["name"] for item in diffusers_models], "Model", spawn="models")
        self.create_combo_box(scheduler_type_values, "Scheduler", spawn="schedulers")
        self.create_check_box("XL", spawn="xl")
        self.create_check_box("TinyVAE", spawn="tinyvae")
        self.create_combo_box(["txt2img", "img2img", "txt2img_cnet", "img2img_cnet", "txt2img_xl", "txt2img_xl_cnet", "img2img_xl"], "pipeline", spawn="pipe_select")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_SDPIPE)
class DiffSDPipelineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Diffusers SDPipeline"
    op_code = OP_NODE_DIFF_SDPIPE
    op_title = "Diffusers - SDPipeline"
    content_label_objname = "diff_pipeline_node"
    category = "aiNodes Base/Diffusers/Loaders"
    NodeContent_class = DiffSDPipelineWidget
    dim = (340, 460)
    output_data_ports = [0]
    custom_input_socket_name = ["VAE", "TOKENIZER 2", "TOKENIZER", "TEXT ENCODER 2", "TEXT ENCODER", "UNET", "CONTROLNET", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[4,4,4,4,4,4,4,1], outputs=[4,1])
        self.pipe = None
    def evalImplementation_thread(self, index=0):

        vae = self.getInputData(0)
        tokenizer_2 = self.getInputData(1)
        tokenizer = self.getInputData(2)
        text_encoder_2 = self.getInputData(3)
        text_encoder = self.getInputData(4)
        unet = self.getInputData(5)
        cnets = self.getInputData(6)

        isxl = self.content.xl.isChecked()

        pipe_select = self.content.pipe_select.currentText()
        add = "_xl" if isxl else ""
        pipe_select = f"{pipe_select}{add}"
        pipe_class = pipes[pipe_select]

        scheduler_name = self.content.schedulers.currentText()
        scheduler = SchedulerType(scheduler_name)
        scheduler_class = get_scheduler_class(scheduler)
        model_key = self.content.models.currentIndex()
        model_name = diffusers_indexed[model_key]
        scheduler = scheduler_class.from_pretrained(model_name, subfolder="scheduler")

        args = {"text_encoder":text_encoder,
                "unet":unet,
                "vae":vae,
                "tokenizer":tokenizer,
                "scheduler":scheduler}
        if isxl:
            args["text_encoder_2"] = text_encoder_2
            args["tokenizer_2"] = tokenizer_2

        if not isxl:
            args["safety_checker"] = None
            args["feature_extractor"] = None
            args["requires_safety_checker"] = False

        if "cnet" in pipe_select:
            args["controlnet"] = cnets["controlnets"][0]

        self.pipe = pipe_class.from_pretrained(model_name, **args)

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