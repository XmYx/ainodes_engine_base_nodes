#from qtpy.QtWidgets import QLineEdit, QLabel, QPushButton, QFileDialog, QVBoxLayout
from qtpy import QtWidgets, QtCore
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException


OP_NODE_DATA = get_next_opcode()
class DataWidget(QDMNodeContentWidget):
    resize_signal = QtCore.Signal()
    def initUI(self):

        self.node_types_list = ["KSampler", "Warp3D", "Debug"]
        self.node_data_types = {
            "KSampler":[("steps", "int"), ("scale", "float"), ("seed", "text")],
            "Warp3D":[("translation_x", "int"),("translation_y", "int"),("translation_z", "int"),("rotation_3d_x", "int"),("rotation_3d_y", "int"),("rotation_3d_z", "int")],
            "Debug":[("debug", "text")]
        }
        self.add_button = QtWidgets.QPushButton("Add more")
        self.print_button = QtWidgets.QPushButton("Print")
        self.print_button.clicked.connect(self.get_widget_values)
        self.node_types = QtWidgets.QComboBox()
        self.node_types.addItems(self.node_types_list)
        self.node_types.currentIndexChanged.connect(self.update_data_types)
        self.data_types = QtWidgets.QComboBox()
        self.update_data_types()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,25)
        layout.addWidget(self.add_button)
        layout.addWidget(self.print_button)
        layout.addWidget(self.node_types)
        layout.addWidget(self.data_types)
        self.setLayout(layout)
        self.add_button.clicked.connect(self.add_widget)

    def add_widget(self):
        node_type = self.node_types.currentText()
        data_type = self.data_types.currentText()
        name = f"{node_type}_{data_type}"
        data_types = [dt for dt, _ in self.node_data_types[node_type]]
        index = data_types.index(data_type)
        _, data_type_class = self.node_data_types[node_type][index]
        widget = None
        if data_type_class == "int":
            widget = QtWidgets.QSpinBox()
        elif data_type_class == "float":
            widget = QtWidgets.QDoubleSpinBox()
        elif data_type_class == "text":
            widget = QtWidgets.QLineEdit()
        if widget is not None:
            label = QtWidgets.QLabel(data_type)
            widget.setAccessibleName(name)
            widget.setObjectName(name)
            # Check if a widget with the same AccessibleName already exists
            for i in range(self.layout().count()):
                item = self.layout().itemAt(i)
                if isinstance(item, QtWidgets.QLayout):
                    for j in range(item.count()):
                        existing_widget = item.itemAt(j).widget()
                        if existing_widget and existing_widget.accessibleName() == name:
                            return
            delete_button = QtWidgets.QPushButton("Delete")
            delete_button.clicked.connect(lambda: self.layout().removeWidget(delete_button))
            delete_button.clicked.connect(lambda: self.layout().removeWidget(widget))
            delete_button.clicked.connect(lambda: self.layout().removeWidget(label))
            delete_button.clicked.connect(widget.deleteLater)
            delete_button.clicked.connect(delete_button.deleteLater)
            delete_button.clicked.connect(label.deleteLater)
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(label)
            hbox.addWidget(widget)
            hbox.addWidget(delete_button)
            self.layout().addLayout(hbox)
        self.node.resize()
    def get_widget_values(self):
        widget_values = {}
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            #print(item)
            if isinstance(item, QtWidgets.QHBoxLayout):
                for j in range(item.count()):
                    sub_item = item.itemAt(j)
                    #print(sub_item)
                    if isinstance(sub_item, QtWidgets.QWidgetItem):
                        widget = sub_item.widget()
                        accessible_name = widget.accessibleName()
                        #print(accessible_name)
                        if accessible_name:
                            node_type, data_type = accessible_name.split("_", 1)
                            if isinstance(widget, QtWidgets.QLineEdit):
                                widget_values[(node_type, data_type)] = widget.text()
                            elif isinstance(widget, QtWidgets.QSpinBox):
                                widget_values[(node_type, data_type)] = widget.value()
                            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                                widget_values[(node_type, data_type)] = widget.value()
        #print(widget_values)
        return widget_values

    def update_data_types(self):
        node_type = self.node_types.currentText()
        self.data_types.clear()
        for data_type, _ in self.node_data_types[node_type]:
            self.data_types.addItem(data_type)



@register_node(OP_NODE_DATA)
class DataNode(AiNode):
    icon = "ainodes_frontend/icons/dot.png"
    op_code = OP_NODE_DATA
    op_title = "Data"
    content_label_objname = "data_node"
    category = "data"
    help_text = "Data Node\n" \
                "Currently, it can be used to create\n" \
                "Deforum Warp values, just connect the data\n" \
                "line, and press Eval on the node to set the value"
    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,1])
        self.interrupt = False
        self.resize()
        # Create a worker object
    def initInnerClasses(self):
        self.content = DataWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 600
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(160)
        self.input_socket_name = ["EXEC", "DATA"]
        self.output_socket_name = ["EXEC", "DATA"]
    @QtCore.Slot()
    def resize(self):
        y = 300
        for i in range(self.content.layout().count()):
            item = self.content.layout().itemAt(i)
            if isinstance(item, QtWidgets.QLayout):
                for j in range(item.count()):
                    y += 15
        self.grNode.height = y + 20
        self.content.setGeometry(0,0,240,y)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.content.setSizePolicy(size_policy)
        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def evalImplementation(self, index=0):
        self.resize()
        self.markDirty(True)
        data = self.getInputData(0)
        values = self.content.get_widget_values()
        if data != None:
            data = merge_dicts(data, values)
        else:
            data = values
        self.setOutput(0, data)
        self.executeChild(1)
        self.markDirty(False)
        self.markInvalid(False)
        return None
    def onMarkedDirty(self):
        self.value = None

    def stop(self):
        self.interrupt = True
        return
    def start(self):
        self.interrupt = False
        self.evalImplementation(0)

def merge_dicts(dict1, dict2):
    result_dict = dict1.copy()
    for key, value in dict2.items():
        if key in result_dict:
            result_dict[key] = value
        else:
            result_dict[key] = value
    return result_dict





