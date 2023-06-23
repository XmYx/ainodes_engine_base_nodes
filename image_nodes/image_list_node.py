import datetime
import os

from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui, QtCore

from ..ainodes_backend import pixmap_to_tensor, tensor_image_to_pixmap

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from PIL import Image

OP_NODE_IMG_LIST = get_next_opcode()
import os
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QListWidget, QListWidgetItem

class ImageListWidget(QWidget):
    pixmap_selected = Signal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QtCore.QSize(256, 256))
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.currentItemChanged.connect(self.on_item_clicked)
        self.list_widget.setViewMode(QtWidgets.QListView.IconMode)

        self.load_button = QPushButton("Load Images")
        self.load_button.clicked.connect(self.load_images)

        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(self.load_button)
        self.setLayout(layout)

    def load_images(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    pixmap = QPixmap(os.path.join(folder_path, file))
                    if not pixmap.isNull():
                        #pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        item = QListWidgetItem()
                        item.setIcon(pixmap)
                        item.setData(Qt.UserRole, pixmap)
                        self.list_widget.addItem(item)

    def on_item_clicked(self, item, prev_item=None):
        pixmap = item.data(Qt.UserRole)
        self.pixmap_selected.emit(pixmap)


class ScribbleWidget(QDMNodeContentWidget):
    preview_signal = QtCore.Signal(object)
    def initUI(self):
        self.image = ImageListWidget(self)
        self.image.setObjectName(self.node.content_label_objname)
        self.checkbox = QtWidgets.QCheckBox("Autosave")
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor("black"))
        self.checkbox.setPalette(palette)
        self.dec_button = QtWidgets.QPushButton("Save Image")
        self.inc_button = QtWidgets.QPushButton("Show Next")
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.dec_button)
        button_layout.addWidget(self.inc_button)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(15, 30, 15, 35)
        layout.addWidget(self.image)

        self.setLayout(layout)
        #self.toggleHelp()

@register_node(OP_NODE_IMG_LIST)
class ImageListNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_list.png"
    op_code = OP_NODE_IMG_LIST
    op_title = "Image List"
    content_label_objname = "image_list_node"
    category = "aiNodes Base/Image"


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
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.images = []
        self.index = 0
        self.content.image.pixmap_selected.connect(self.set_pixmap)
        self.content.eval_signal.connect(self.evalImplementation)

    #@QtCore.Slot(object)
    def set_pixmap(self, pixmap):
        print("PIXMAP SELECTED", pixmap)
        self.pixmap = pixmap

    def select_next_item(self):
        current_row = self.content.image.list_widget.currentRow()
        num_items = self.content.image.list_widget.count()
        if current_row == num_items - 1:
            self.content.image.list_widget.setCurrentRow(0)
        else:
            self.content.image.list_widget.setCurrentRow(current_row + 1)

    def evalImplementation_thread(self, index=0):
        self.select_next_item()
        pixmap = self.pixmap
        print(pixmap)
        return pixmap

    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        self.setOutput(0, [result])
        self.executeChild(2)

    def resize(self):
        self.grNode.setToolTip("")
        self.grNode.height = self.content.image.size().height() + 155
        self.grNode.width = self.content.image.size().width() + 32
        self.content.setGeometry(0, 0, self.content.image.size().width(),
                                 self.content.image.size().height())
        self.update_all_sockets()
