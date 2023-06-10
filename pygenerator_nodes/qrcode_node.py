from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap

#Function imports
import qrcode
from PIL import Image

#MANDATORY
OP_NODE_QRCODE = get_next_opcode()

#NODE WIDGET
class QRWidget(QDMNodeContentWidget):
    def initUI(self):
        self.textedit = self.create_line_edit("Qr Data", placeholder="Enter QR Data here")
        self.qr_version = self.create_spin_box(label_text="QR Code Version", min_val=1, max_val=40, default_val=1)
        self.qr_box = self.create_spin_box(label_text="QR Box Size", min_val=1, max_val=40, default_val=10)
        self.qr_border = self.create_spin_box(label_text="QR Border Size", min_val=0, max_val=40, default_val=4)
        self.qr_fit = self.create_check_box("Fit to minimum QR Version")
        self.create_main_layout(grid=1)

#NODE CLASS
@register_node(OP_NODE_QRCODE)
class QRNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/qr_gen.png"
    help_text = "QR Code Generator Node."
    op_code = OP_NODE_QRCODE
    op_title = "QR Code Generator"
    content_label_objname = "qrcode_node"
    category = "Image"
    NodeContent_class = QRWidget
    dim = (340, 260)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):
        self.busy = True
        data = self.content.textedit.text()

        version = self.content.qr_version.value()
        box = self.content.qr_box.value()
        border = self.content.qr_border.value()
        fit = self.content.qr_fit.isChecked()
        qr_image = create_qr_code(data, version, box, border, fit)
        pixmap = pil_image_to_pixmap(qr_image)
        return [[pixmap]]


def create_qr_code(data, version, box, border, fit):
    # Create qr code instance
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box,
        border=border,
    )

    # Add data to qr code
    qr.add_data(data)

    qr.make(fit=fit)

    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white")

    # Resize the image to 512x512
    resized_img = img.resize((512,512), Image.ANTIALIAS)

    return resized_img