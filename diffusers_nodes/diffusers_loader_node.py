import diffusers as df
import torch

from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import scheduler_type_values, diffusers_models
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_DIFF_LOADER = get_next_opcode()


pipes = {

    "StableDiffusion":df.StableDiffusionPipeline,
    "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    "StableDiffusionControlNet":df.StableDiffusionControlNetPipeline,
    "StableDiffusionImg2ImgControlNet":df.StableDiffusionControlNetImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,
    # "StableDiffusionImg2Img":df.StableDiffusionImg2ImgPipeline,

}

class DiffusersLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.models = self.create_combo_box([item["name"] for item in diffusers_models], "Model")
        self.pipe_type = self.create_combo_box(pipes.keys(), "Pipeline Type")
        self.reload = self.create_check_box("Reload")
        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DIFF_LOADER)
class DataMergeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_DIFF_LOADER
    op_title = "DataMerger"
    content_label_objname = "datamerge_node"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = DiffusersLoaderWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1

    custom_output_socket_name = ["PIPE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[4,1])

        self.pipe_type = None

    def evalImplementation_thread(self, index=0):


        reload = self.content.reload.isChecked()
        pipe_type = self.content.pipe_type.currentText()
        pipe_class = pipes[pipe_type]
        model_name = self.content.models.currentText()
        data = self.getInputData(0)


        options = {"model_name":model_name,
                   "torch_dtype":torch.float16,
                   "safety_checker":None}

        type_string = f"{pipe_type}_{model_name}"

        if not self.pipe_type or reload:

            if data:
                if "control_diff" in data:
                    control_params = []
                    x = 0

            self.pipe = pipe_class.from_pretrained(**options)
            self.pipe_type = type_string




            if "control_diff" in data:
                control_params = []
                x = 0

                if len(data["control_diff"]) != self.control_params:
                    reload = True

                for control in data["control_diff"]:
                    control_params.append(control["name"])
                    # if len(control_params) >= x:
                    #     if not reload and control_params[x] != self.control_params[x]:
                    #         reload = True
                    # else:
                    #     reload = True
                    if reload:
                        cnet = ControlNetModel.from_pretrained(control["name"], torch_dtype=torch.float16).to(get_torch_device())
                    else:
                        cnet = self.pipe.controlnet.nets[x]
                    cnet.start_control = control["start"]
                    cnet.stop_control = control["stop"]
                    control_images.append(control["image"])
                    cnet_scales.append(control["scale"])
                    if reload:
                        controlnets.append(cnet)

                    x += 1
                self.control_params = control_params
