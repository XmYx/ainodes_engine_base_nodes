import datetime
import os

from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui, QtCore

from ..ainodes_backend import pixmap_to_pil_image, pil_image_to_pixmap

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton
gs = singleton.Singleton.instance()
from PIL import Image

OP_NODE_IMG_SCRIBBLE = get_next_opcode()
class DrawingWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage(512, 512, QtGui.QImage.Format_RGB32)
        self.image.fill(Qt.black)
        self.drawing = False
        self.last_point = None
        self.brush_size = 10
        self.setCursor(self.createBrushCursor())
        self.dec_button = QtWidgets.QPushButton()

    def get_image(self):
        pixmap = QtGui.QPixmap.fromImage(self.image)
        return pixmap

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)

    def createBrushCursor(self, color=None):
        cursor_pixmap = QtGui.QPixmap(self.brush_size * 2 + 2, self.brush_size * 2 + 2)
        cursor_pixmap.fill(Qt.transparent)
        cursor_painter = QtGui.QPainter(cursor_pixmap)
        if color == None:
            color = Qt.white
        # Draw the outer circle
        cursor_painter.setPen(QtGui.QPen(color, 1, Qt.SolidLine))
        cursor_painter.drawEllipse(1, 1, self.brush_size * 2 - 1, self.brush_size * 2 - 1)

        # Fill the inner circle
        cursor_painter.setBrush(color)
        cursor_painter.setPen(QtGui.QPen(Qt.transparent))
        cursor_painter.drawEllipse(2, 2, self.brush_size * 2 - 2, self.brush_size * 2 - 2)

        cursor_painter.end()
        return QtGui.QCursor(cursor_pixmap)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_point is not None:
            painter = QtGui.QPainter(self.image)
            pen = QtGui.QPen()
            if event.modifiers() & Qt.ShiftModifier:
                pen.setColor(Qt.black)
            else:
                pen.setColor(Qt.white)

            pen.setWidth(self.brush_size)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)



            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setCursor(self.createBrushCursor(Qt.white))
            self.drawing = False
            self.last_point = None
    def dec_brush(self):
        self.brush_size -= 3
        if self.brush_size < 1:
            self.brush_size = 1
        self.setCursor(self.createBrushCursor())
    def inc_brush(self):
        self.brush_size += 3
        self.setCursor(self.createBrushCursor())


class ScribbleWidget(QDMNodeContentWidget):
    preview_signal = QtCore.Signal(object)
    def initUI(self):
        self.image = DrawingWidget(self)
        self.image.setObjectName(self.node.content_label_objname)
        self.checkbox = QtWidgets.QCheckBox("Autosave")
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
        self.checkbox.setPalette(palette)
        self.dec_button = QtWidgets.QPushButton("Smaller")
        self.inc_button = QtWidgets.QPushButton("Larger")
        self.new_image = QtWidgets.QPushButton("Resize")
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.dec_button)
        self.button_layout.addWidget(self.inc_button)
        self.button_layout.addWidget(self.new_image)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 35)
        layout.addWidget(self.image)
        layout.addLayout(self.button_layout)
        self.setLayout(layout)

@register_node(OP_NODE_IMG_SCRIBBLE)
class ScribbleNode(AiNode):
    icon = "icons/out.png"
    op_code = OP_NODE_IMG_SCRIBBLE
    op_title = "Scribble"
    content_label_objname = "image_scribble_node"
    category = "image"


    def __init__(self, scene):
        super().__init__(scene, inputs=[5,6,1], outputs=[5,6,1])

    def initInnerClasses(self):
        self.content = ScribbleWidget(self)
        self.content.setMinimumWidth(512)
        self.content.setMinimumHeight(512)
        self.content.setGeometry(QtCore.QRect(15,15,512,512))
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 640
        self.grNode.width = 512
        self.images = []
        self.index = 0

        self.content.dec_button.clicked.connect(self.content.image.dec_brush)
        self.content.inc_button.clicked.connect(self.content.image.inc_brush)
        self.content.new_image.clicked.connect(self.new_image)

    def evalImplementation(self, index=0):

        pixmap = self.content.image.get_image()
        self.markDirty(False)
        self.setOutput(0, pixmap)
        self.executeChild(2)
    def new_image(self):
        image = get_custom_size_image()
        if image is not None:
            self.content.image.image = image
        self.resize()

    def onMarkedDirty(self):
        #
        pass
    def onMarkedInvalid(self):
        self.content.image.image.fill(Qt.black)
    def onInputChanged(self, socket=None):

        pass
    def eval(self):
        self.evalImplementation(0)

    def resize(self):
        self.grNode.setToolTip("")

        width = self.content.image.image.size().width()
        height = self.content.image.image.size().height()
        self.grNode.height = height + 120
        self.grNode.width = width + 30
        self.content.setMinimumWidth(width)
        self.content.setMinimumHeight(height)
        self.content.button_layout.setGeometry(QtCore.QRect(15,height - 25,width,25))
        self.update_all_sockets()
        QtCore.QRect()

def get_custom_size_image():
    def create_spin_box():
        spin_box = QtWidgets.QSpinBox()
        spin_box.setRange(64, 1024)
        spin_box.setSingleStep(64)
        spin_box.setValue(512)
        spin_box.setSuffix(" px")
        return spin_box

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Select Image Size")

    layout = QtWidgets.QVBoxLayout()

    width_label = QLabel("Width:")
    layout.addWidget(width_label)
    width_spin_box = create_spin_box()
    layout.addWidget(width_spin_box)

    height_label = QLabel("Height:")
    layout.addWidget(height_label)
    height_spin_box = create_spin_box()
    layout.addWidget(height_spin_box)

    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    layout.addWidget(button_box)

    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)

    dialog.setLayout(layout)

    result = dialog.exec()
    if result == QtWidgets.QDialog.Accepted:
        width = width_spin_box.value()
        height = height_spin_box.value()
        image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        image.fill(Qt.black)
        return image
    else:
        return None