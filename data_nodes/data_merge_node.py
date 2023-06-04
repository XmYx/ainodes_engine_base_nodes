from qtpy import QtCore
from qtpy import QtWidgets

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs


OP_NODE_DATA_MERGE = get_next_opcode()
class DataMergeWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout()



class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_DATA_MERGE)
class DataMergeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_DATA_MERGE
    op_title = "DataMerger"
    content_label_objname = "datamerge_node"
    category = "Data"
    #input_socket_name = ["EXEC"]
    #output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6,6,1], outputs=[6,1])

    def initInnerClasses(self):
        self.content = DataMergeWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 340
        self.grNode.height = 180
        self.content.setMinimumHeight(140)
        self.content.setMinimumWidth(340)



    def evalImplementation_thread(self, index=0):
        self.busy = True

        data1 = self.getInputData(1)
        data2 = self.getInputData(0)
        if gs.debug:
            print("DATA1", data1, "DATA2", data2)


        new_data = {}
        try:
            if data1 and data2:
                for key, value in data1.items():
                    new_data[key] = value
                for key, value in data2.items():
                    new_data[key] = value
            elif data1 and not data2:
                new_data = data1
            elif data2 and not data1:
                new_data = data2
            if gs.debug:
                print("NEW DATA", new_data)
            return new_data
        except Exception as e:
            deno = handle_ainodes_exception()
            return new_data


    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.markDirty(False)
        self.setOutput(0, result)
        pass
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
    def onInputChanged(self, socket=None):
        pass




