import secrets

from ..ainodes_backend.inpaint import run_inpaint
from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap

from qtpy import QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_INPAINT = get_next_opcode()
class InpaintWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        #self.text_label = QtWidgets.QLabel("K Sampler")


        self.seed_layout = QtWidgets.QHBoxLayout()
        self.seed_label = QtWidgets.QLabel("Seed:")
        self.seed = QtWidgets.QLineEdit()
        self.seed_layout.addWidget(self.seed_label)
        self.seed_layout.addWidget(self.seed)

        self.prompt = QtWidgets.QTextEdit()

        self.steps_layout = QtWidgets.QHBoxLayout()
        self.steps_label = QtWidgets.QLabel("Steps:")
        self.steps = QtWidgets.QSpinBox()
        self.steps.setMinimum(1)
        self.steps.setMaximum(1000)
        self.steps.setValue(10)
        self.steps_layout.addWidget(self.steps_label)
        self.steps_layout.addWidget(self.steps)


        self.guidance_scale_layout = QtWidgets.QHBoxLayout()
        self.guidance_scale_label = QtWidgets.QLabel("Guidance Scale:")
        self.guidance_scale = QtWidgets.QDoubleSpinBox()
        self.guidance_scale.setMinimum(1.01)
        self.guidance_scale.setMaximum(100.00)
        self.guidance_scale.setSingleStep(0.01)
        self.guidance_scale.setValue(7.50)
        self.guidance_scale_layout.addWidget(self.guidance_scale_label)
        self.guidance_scale_layout.addWidget(self.guidance_scale)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button = QtWidgets.QPushButton("Run")
        self.fix_seed_button = QtWidgets.QPushButton("Fix Seed")
        self.button_layout.addWidget(self.button)
        self.button_layout.addWidget(self.fix_seed_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        layout.addLayout(self.seed_layout)
        layout.addWidget(self.prompt)
        layout.addLayout(self.steps_layout)
        layout.addLayout(self.guidance_scale_layout)
        layout.addLayout(self.button_layout)

        self.setLayout(layout)



@register_node(OP_NODE_INPAINT)
class InpaintNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_INPAINT
    op_title = "InPaint Alpha"
    content_label_objname = "inpaint_sampling_node"
    category = "Sampling"
    custom_input_socket_names = ["MASK", "IMAGE", "EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[5,1])

    def initInnerClasses(self):
        self.content = InpaintWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 500
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.seed = ""
        self.content.fix_seed_button.clicked.connect(self.setSeed)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self):
        pixmap = None
        try:
            image_input_node, index = self.getInput(1)
            image_pixmap = image_input_node.getOutput(index)
        except Exception as e:
            print(e)
        try:
            mask_input_node, index = self.getInput(0)
            mask_pixmap = mask_input_node.getOutput(index)
        except Exception as e:
            print(e)
        if image_pixmap is not None and mask_pixmap is not None:
            init_image = pixmap_to_pil_image(image_pixmap[0])
            mask_image = pixmap_to_pil_image(mask_pixmap[0])

            prompt = self.content.prompt.toPlainText()
            try:
                seed = self.content.seed.text()
                seed = int(seed)
            except:
                seed = secrets.randbelow(99999999)
            scale = self.content.guidance_scale.value()
            steps = self.content.steps.value()
            blend_mask = 5
            mask_blur = 5
            recons_blur = 5

            result = run_inpaint(init_image, mask_image, prompt, seed, scale, steps, blend_mask, mask_blur, recons_blur)

            print("RESULT", result)
            pixmap = pil_image_to_pixmap(result)
            return pixmap

    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        if result is not None:
            self.setOutput(0, [result])
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)
        else:
            self.markDirty(True)

    def setSeed(self):
        self.content.seed.setText(str(self.seed))

