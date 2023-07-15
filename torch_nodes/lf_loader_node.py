from qtpy.QtWidgets import QLineEdit
from qtpy import QtCore, QtGui
from qtpy import QtWidgets

from ..ainodes_backend.model_loader import ModelLoader

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs


import torch

OP_NODE_DEEPFLOYD_LOADER = get_next_opcode()
class TorchLoaderWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)

    def create_widgets(self):
        checkpoint_files = ["Stage 1", "Stage 2", "Stage 3"]
        self.dropdown = self.create_combo_box(checkpoint_files, "Models")
        self.token = self.create_line_edit("Token:")
        self.token.setEchoMode(QLineEdit.Password)



class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_DEEPFLOYD_LOADER)
class DeepFloydLoaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/deep_floyd.png"
    op_code = OP_NODE_DEEPFLOYD_LOADER
    op_title = "DeepFloyd Loader"
    content_label_objname = "deepfloyd_loader_node"
    category = "aiNodes Base/Model Loading"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[6,1])
        self.loader = ModelLoader()

    def initInnerClasses(self):
        self.content = TorchLoaderWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.width = 340
        self.grNode.height = 180
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        #pass
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        #pass
        model = ""
        if gs.token == "":
            token = self.content.token.text()
            gs.token = token
        else:
            token = gs.token

        try:
            from diffusers import DiffusionPipeline
            from diffusers.utils import pt_to_pil
            model_name = self.content.dropdown.currentText()
            if model_name == "Stage 1":
                if "deepfloyd_1" not in gs.models:
                    # stage 1
                    gs.models["deepfloyd_1"] = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16, use_auth_token=token)
                    #gs.models["deepfloyd_1"].enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
                    gs.models["deepfloyd_1"].enable_model_cpu_offload()
                model = "deepfloyd_1"
            elif model_name == "Stage 2":
                if "deepfloyd_2" not in gs.models:
                    # stage 2
                    gs.models["deepfloyd_2"] = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16, use_auth_token=token)
                    #gs.models["deepfloyd_2"].enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
                    gs.models["deepfloyd_2"].enable_model_cpu_offload()
                model = "deepfloyd_2"

            elif model_name == "Stage 3":
                if "deepfloyd_3" not in gs.models:
                    # stage 3
                    safety_modules = {"feature_extractor": None,
                                      "safety_checker": None, "watermarker": None}
                    gs.models["deepfloyd_3"] = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler",
                                                                **safety_modules, torch_dtype=torch.float16)
                    #gs.models["deepfloyd_3"].enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
                    gs.models["deepfloyd_3"].enable_model_cpu_offload()
                model = "deepfloyd_3"
            return model
        except Exception as e:
            print("Could not load DeepFloyd because of ", e)

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)

        self.content.token.setText(gs.token)
        self.markDirty(False)
        self.content.update()
        data = {"model":result}
        self.setOutput(0, data)
        #pass
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
    def onInputChanged(self, socket=None):
        pass




