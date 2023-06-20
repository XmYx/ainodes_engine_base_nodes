


from PIL import Image
from qtpy import QtCore

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap, pixmap_to_pil_image

#Function imports

#MANDATORY
OP_NODE_QRCODE_READER = get_next_opcode()

#NODE WIDGET
class QRReaderWidget(QDMNodeContentWidget):
    label_signal = QtCore.Signal(str)
    def initUI(self):
        self.label = self.create_label("")
        self.create_main_layout(grid=1)



#NODE CLASS
@register_node(OP_NODE_QRCODE_READER)
class QRReaderNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/qr_val.png"
    help_text = "QR Code Generator Node."
    op_code = OP_NODE_QRCODE_READER
    op_title = "QR Code Validator"
    content_label_objname = "qrcode_validator_node"
    category = "Image"
    NodeContent_class = QRReaderWidget
    dim = (340, 260)
    output_data_ports = [0]
    exec_port = 0

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[1])

    def initInnerClasses(self):
        super().initInnerClasses()
        self.content.label_signal.connect(self.set_label)

    def set_label(self, text:str):
        self.content.label.setText(text)

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):
        pixmaps = self.getInputData(0)

        for pixmap in pixmaps:
            image = pixmap_to_pil_image(pixmap)
            result = read_qr(image)
            self.content.label_signal.emit(result)
            print(result)
        return [None]



def read_qr(pil_image):
    from pyzbar.pyzbar import decode

    pil_image = pil_image.convert("L")
    result = decode(pil_image)

    if result:
        # If the QR code is readable, return the decoded result
        return result[0].data.decode('utf-8')
    else:
        # If the QR code is not readable, return a specific message
        return "The image could not be read as a QR code."