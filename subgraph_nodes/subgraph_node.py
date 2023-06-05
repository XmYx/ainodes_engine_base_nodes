from qtpy import QtCore
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_SUBGRAPH = get_next_opcode()
OP_NODE_SUBGRAPH_INPUT = get_next_opcode()
OP_NODE_SUBGRAPH_OUTPUT = get_next_opcode()

class SubgraphNodeWidget(QDMNodeContentWidget):
    def initUI(self):
        self.label = self.create_label(str(f"ID: {self.node.name}"))
        self.edit_label = self.create_line_edit("Name")
        self.description = self.create_text_edit("Description")
        self.create_main_layout(grid=1)


class SubgraphNodePathwayWidget(QDMNodeContentWidget):
    done_signal = QtCore.Signal()
    def initUI(self):
        pass

@register_node(OP_NODE_SUBGRAPH)
class SubgraphNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/subgraph.png"
    op_code = OP_NODE_SUBGRAPH
    op_title = "Subgraph"
    content_label_objname = "subgraph_node"
    category = "Subgraph Nodes"
    help_text = "Execution Node\n\n" \
                "Execution chain is essential\n" \
                "in aiNodes. You control the flow\n" \
                "You control the magic. Each value\n" \
                "is created and stored at execution\n" \
                "once a node is validated, you don't\n" \
                "have to run it again in order to get\n" \
                "it's value, just simply connect the\n" \
                "relevant data line. Only execute, if you\n" \
                "want, or have to get a new value."
    load_graph = True
    init_done = None
    graph_json = None
    graph_window = None

    def __init__(self, scene, graph_json=None):
        super().__init__(scene, inputs=[2,3,5,6,1], outputs=[2,3,5,6,1])
        self.graph_json = graph_json

        if self.graph_json:
            self.scene.getView().parent().window().json_open_signal.emit(self)

    def initInnerClasses(self):
        self.name = f"{self.getID(0)}_Subgraph"
        self.content = SubgraphNodeWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.height = 400
        self.grNode.width = 300
        self.content.setMinimumHeight(260)
        self.content.setMinimumWidth(300)
        self.content.eval_signal.connect(self.evalImplementation)
        self.init_done = True

    def force_init(self):
        self.scene.getView().parent().window().json_open_signal.emit(self)

    def evalImplementation_thread(self, index=0, *args, **kwargs):

        nodes = self.graph_window.widget().scene.nodes
        for node in nodes:
            if isinstance(node, SubGraphOutputNode):
                node.true_parent = self
                continue
        for node in nodes:
            if isinstance(node, SubGraphInputNode):

                node.data = self.getInputData(3)
                node.images = self.getInputData(2)
                node.latents = self.getInputData(0)
                node.conds = self.getInputData(1)

                node.content.eval_signal.emit()
                return None

        return None

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if result:
            self.setOutput(0, result[0])
            self.setOutput(1, result[1])
            self.setOutput(2, result[2])
            self.setOutput(3, result[3])
            self.executeChild(4)

    def onDoubleClicked(self, event=None):
        if self.graph_window:
            self.scene.getView().parent().window().mdiArea.setActiveSubWindow(self.graph_window)
        else:
            if self.graph_json:
                self.scene.getView().parent().window().json_open_signal.emit(self)
            else:
                self.scene.getView().parent().window().file_new_signal.emit(self)
                self.graph_window.widget().filename = f"{self.getID(0)}_Subgraph"

    def serialize(self):
        """
        Serialize the node's data into a dictionary.

        Returns:
            dict: The serialized data of the node.
        """
        #if self.graph_window:
        #    self.graph_json = self.graph_window.widget().scene.serialize()

        res = super().serialize()
        res['op_code'] = self.__class__.op_code
        res['content_label_objname'] = self.__class__.content_label_objname
        if self.graph_window:
            res['node_graph'] = self.graph_window.widget().scene.serialize()
            self.graph_json = res['node_graph']
        return res

    def deserialize(self, data, hashmap={}, restore_id=True):
        """
        Deserialize the node's data from a dictionary.

        Args:
            data (dict): The serialized data of the node.
            hashmap (dict): A dictionary of node IDs and their corresponding objects.
            restore_id (bool): Whether to restore the original node ID. Defaults to True.

        Returns:
            bool: True if deserialization is successful, False otherwise.
        """
        #print(data)
        if 'node_graph' in data:
            self.graph_json = data['node_graph']
        res = super().deserialize(data, hashmap, restore_id)
        return res


@register_node(OP_NODE_SUBGRAPH_INPUT)
class SubGraphInputNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/subgraph_in.png"
    op_code = OP_NODE_SUBGRAPH_INPUT
    op_title = "Subgraph Inputs"
    content_label_objname = "subgraph_input_node"
    category = "Subgraph Nodes"
    help_text = "Execution Node\n\n" \
                "Execution chain is essential\n" \
                "in aiNodes. You control the flow\n" \
                "You control the magic. Each value\n" \
                "is created and stored at execution\n" \
                "once a node is validated, you don't\n" \
                "have to run it again in order to get\n" \
                "it's value, just simply connect the\n" \
                "relevant data line. Only execute, if you\n" \
                "want, or have to get a new value."

    def __init__(self, scene):
        super().__init__(scene, inputs=[], outputs=[2, 3, 5, 6, 1])

    def initInnerClasses(self):
        self.content = SubgraphNodePathwayWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.height = 180
        self.content.eval_signal.connect(self.evalImplementation)
        self.latents = None
        self.conds = None
        self.images = None
        self.data = None

    def evalImplementation_thread(self, index=0, *args, **kwargs):

        self.setOutput(0, self.latents)
        self.setOutput(1, self.conds)
        self.setOutput(2, self.images)
        self.setOutput(3, self.data)
        return True

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #print("REACHED SUBGRAPH INPUT NODE END")
        #super().onWorkerFinished(None)
        self.executeChild(4)


@register_node(OP_NODE_SUBGRAPH_OUTPUT)
class SubGraphOutputNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/subgraph_out.png"
    op_code = OP_NODE_SUBGRAPH_OUTPUT
    op_title = "Subgraph Outputs"
    content_label_objname = "subgraph_output_node"
    category = "Subgraph Nodes"
    help_text = "Execution Node\n\n" \
                "Execution chain is essential\n" \
                "in aiNodes. You control the flow\n" \
                "You control the magic. Each value\n" \
                "is created and stored at execution\n" \
                "once a node is validated, you don't\n" \
                "have to run it again in order to get\n" \
                "it's value, just simply connect the\n" \
                "relevant data line. Only execute, if you\n" \
                "want, or have to get a new value."

    def __init__(self, scene):
        super().__init__(scene, inputs=[2, 3, 5, 6, 1], outputs=[])

    def initInnerClasses(self):
        self.content = SubgraphNodePathwayWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.height = 180
        self.content.eval_signal.connect(self.evalImplementation)
        self.true_parent = None

    def run(self, data, images, latens, conds):
        pass


    def evalImplementation_thread(self, index=0, *args, **kwargs):

        latents = self.getInputData(0)
        conds = self.getInputData(1)
        images = self.getInputData(2)
        data = self.getInputData(3)

        return[latents, conds, images, data]


    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)

        self.setOutput(0, result[0])
        self.setOutput(1, result[1])
        self.setOutput(2, result[2])
        self.setOutput(3, result[3])
        if self.true_parent:
            self.true_parent.onWorkerFinished(result)








