from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_DATA_MERGE = get_next_opcode()
class DataMergeWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout()

@register_node(OP_NODE_DATA_MERGE)
class DataMergeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_DATA_MERGE
    op_title = "DataMerger"
    content_label_objname = "datamerge_node"
    category = "Experimental"
    NodeContent_class = DataMergeWidget
    dim = (340, 180)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,6,1], outputs=[6,1])

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
            return [new_data]
        except Exception as e:
            done = handle_ainodes_exception()
            return [new_data]

