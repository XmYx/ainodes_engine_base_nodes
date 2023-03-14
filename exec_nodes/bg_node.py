#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from ainodes_frontend.node_engine.node_graphics_node import QDMGraphicsBGNode
from qtpy import QtWidgets, QtGui

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import CalcNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

OP_NODE_BG = get_next_opcode()

@register_node(OP_NODE_BG)
class BGNode(CalcNode):
    icon = "icons/in.png"
    op_code = OP_NODE_BG
    op_title = "Bg Node"
    content_label_objname = "bg_node"
    category = "exec"

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[])
        #self.content.button.clicked.connect(self.evalImplementation)
        self.busy = False
        # Create a worker object
    def initInnerClasses(self):
        self.grNode = QDMGraphicsBGNode(self)
        self.grNode.setZValue(-2)

    def evalImplementation(self, index=0):
        return None

    def onMarkedDirty(self):
        self.value = None

    def serialize(self):
        res = super().serialize()
        res['color'] = self.grNode._brush_background.color().name(QtGui.QColor.NameFormat.HexArgb)
        res['width'] = self.grNode.width
        res['height'] = self.grNode.height
        return res

    def deserialize(self, data, hashmap={}, restore_id=False):
        res = super().deserialize(data, hashmap)
        try:
            deserialized_color = QtGui.QColor(data['color'])
            deserialized_brush = QtGui.QBrush(deserialized_color)
            self.grNode._brush_background = deserialized_brush
            self.grNode.width = (data['width'])
            self.grNode.height = (data['height'])
            #self.grNode._min_size = self.grNode.width, self.grNode.height
            self.grNode._sizer.set_pos([self.grNode.width, self.grNode.height])
            return True & res
        except Exception as e:
            dumpException(e)
        return res








