import secrets

import torch

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, torch_gc, \
    get_torch_device, pil2tensor, tensor2pil
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, \
    StableDiffusionImageVariationPipeline

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import multiForward, \
    scheduler_type_values, SchedulerType, get_scheduler
from ainodes_frontend import singleton as gs

#MANDATORY
OP_NODE_DIFF_VAR_PIPELINE = get_next_opcode()

#NODE WIDGET
class DiffusersVarPipeLineWidget(QDMNodeContentWidget):
    def initUI(self):
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")

        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.eta = self.create_double_spin_box("Eta", min_val=0.00, max_val=1.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")
        self.create_main_layout(grid=1)

#NODE CLASS
@register_node(OP_NODE_DIFF_VAR_PIPELINE)
class DiffusersVarPipeLineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/variations.png"
    help_text = "Diffusers - Variations"
    op_code = OP_NODE_DIFF_VAR_PIPELINE
    op_title = "Diffusers - Variations"
    content_label_objname = "diffusers_variations_node"
    category = "aiNodes Base/Diffusers"
    NodeContent_class = DiffusersVarPipeLineWidget
    dim = (340, 300)
    output_data_ports = [0,1]
    exec_port = 2

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,4,1], outputs=[5,4,1])
        self.pipe = None

    def evalImplementation_thread(self, index=0):
        images = self.getInputData(0)
        image = tensor2pil(images[0])

        pipe = self.getInputData(1)

        print("PIPE", pipe)
        print("IMG", images)

        if pipe:
            self.pipe = pipe

        if self.pipe == None:
            self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
              "lambdalabs/sd-image-variations-diffusers",
              revision="v2.0",
              safety_checker=None,
              ).to(gs.device)
        else:
            self.pipe.to(gs.device)

        scheduler_name = self.content.scheduler_name.currentText()
        scheduler_enum = SchedulerType(scheduler_name)
        self.pipe = get_scheduler(self.pipe, scheduler_enum)

        #self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        guidance_scale = self.content.scale.value()
        steps = self.content.steps.value()
        eta = self.content.eta.value()

        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        generator = torch.Generator(gs.device).manual_seed(seed)
        latents = None

        image = self.pipe(image,
                          width=image.size[0],
                          height=image.size[1],
                          num_inference_steps = steps,
                          guidance_scale=guidance_scale,
                          generator=generator,
                          eta=eta).images[0]

        torch_gc()
        return [[pil2tensor(image)], self.pipe]


    def remove(self):
        print("REMOVING", self)
        if self.pipe:
            self.pipe.to("cpu")
            del self.pipe
            torch_gc()

        super().remove()

