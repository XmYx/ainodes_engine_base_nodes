import threading

from qtpy import QtWidgets, QtCore

from ..torch_nodes.torch_loader import TorchLoaderNode

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

from ainodes_frontend import singleton as gs

OP_NODE_CONDITIONING = get_next_opcode()
class ConditioningWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
    def create_widgets(self):
        self.prompt = self.create_text_edit("Prompt")
        self.button = QtWidgets.QPushButton("Get Conditioning")
        self.create_button_layout([self.button])

@register_node(OP_NODE_CONDITIONING)
class ConditioningNode(AiNode):
    icon = "icons/in.png"
    op_code = OP_NODE_CONDITIONING
    op_title = "Conditioning"
    content_label_objname = "cond_node"
    category = "sampling"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[3,1])
        #self.eval()
        self.content.eval_signal.connect(self.evalImplementation)
        # Create a worker object
    def initInnerClasses(self):
        self.content = ConditioningWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 256
        self.grNode.width = 320
        self.content.setMinimumHeight(200)
        self.content.setMinimumWidth(320)
        self.busy = False
        self.content.button.clicked.connect(self.evalImplementation)
        self.input_socket_name = ["EXEC"]
        self.output_socket_name = ["EXEC", "COND"]

        #self.content.image.changeEvent.connect(self.onInputChanged)
    def evalImplementation(self, index=0):
        if self.busy == False:
            self.busy = True
            thread0 = threading.Thread(target=self.evalImplementation_thread)
            thread0.start()
            return None
        else:
            self.markDirty(False)
            self.markInvalid(False)
            return None

    def evalImplementation_thread(self, index=0):
        try:
            self.markDirty(True)
            print(f"CONDITIONING NODE: Applying conditioning with prompt: {self.content.prompt.toPlainText()}")
            result = self.get_conditioning()
            self.setOutput(0, result)
            # print(result)

            self.markDirty(False)
            self.markInvalid(False)

            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
            self.busy = False
            return None
        except:
            self.busy = False
            return None

    def onMarkedDirty(self):
        self.value = None
    def get_conditioning(self, progress_callback=None):
        #print("Getting Conditioning on ", id(self))
        prompt = self.content.prompt.toPlainText()

        """if gs.loaded_models["loaded"] == []:
            for node in self.scene.nodes:
                if isinstance(node, TorchLoaderNode):
                    node.evalImplementation()
                    #print("Node found")"""

        c = gs.models["sd"].model.cond_stage_model.encode([prompt])
        uc = {}
        return [[c, uc]]
    def onWorkerFinished(self, result):
        # Update the node value and mark it as dirty
        self.value = result
        #self.scene.queue.task_finished.disconnect(self.onWorkerFinished)
        #self.worker.autoDelete()
        #result = None
        self.setOutput(0, result)
        #print(result)
        self.markDirty(False)
        self.markInvalid(False)
        self.busy = False
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return
        #self.markDescendantsDirty()
        #self.evalChildren()
    def onInputChanged(self, socket=None):
        pass

    def exec(self):
        self.markDirty(True)
        self.markInvalid(True)
        self.value = None
        self.eval(0)

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]
