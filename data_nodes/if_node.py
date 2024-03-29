from qtpy import QtCore
from qtpy import QtWidgets

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs


OP_NODE_DATA_IF = get_next_opcode()
class DataIfWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)


    def createUI(self):
        self.data_name = self.create_line_edit("Data Name")
        self.data_equal = self.create_line_edit("Equal")


class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_DATA_IF)
class DataIfNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    op_code = OP_NODE_DATA_IF
    op_title = "Data IF Node"
    content_label_objname = "data_if_node"
    category = "aiNodes Base/WIP Experimental"
    #input_socket_name = ["EXEC"]
    #output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6,1], outputs=[6,1])

    def initInnerClasses(self):
        self.content = DataIfWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 340
        self.grNode.height = 180
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)
        pass
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        self.busy = True

        data = self.getInputData(0)
        if gs.debug:
            print("DATA1", data, "DATA2", data)


        new_data = {}

        if data is not None:
            data_name = self.content.data_name.text()
            data_value = self.content.data_equal.text()



    #@QtCore.Slot(object)
    def onWorkerFinished(self, result, exec=True):
        self.busy = False
        self.markDirty(False)
        self.setOutput(0, result)
        pass
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
    def onInputChanged(self, socket=None):
        pass




