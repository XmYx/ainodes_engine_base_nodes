import torch
from qtpy import QtWidgets, QtCore

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

from ainodes_frontend import singleton as gs


OP_NODE_CONDITIONING_COMBINE = get_next_opcode()
OP_NODE_CONDITIONING_SET_AREA = get_next_opcode()

#from singleton import Singleton
#gs = Singleton()

class ConditioningSetAreaWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()
        # Create a label to display the image
        #self.text_label = QtWidgets.QLabel("Diffusers:")
    def create_widgets(self):
        self.width = self.create_spin_box("Width", 64, 4096, 512, 64)
        self.height = self.create_spin_box("Height", 64, 4096, 512, 64)
        self.x_spinbox = self.create_spin_box("X", 0, 4096, 0, 1)
        self.y_spinbox = self.create_spin_box("Y", 0, 4096, 0, 1)
        self.strength = self.create_double_spin_box("strength", 0.01, 10.00, 0.01, 1.00)

class ConditioningCombineWidget(QDMNodeContentWidget):

    def initUI(self):
        self.create_widgets()
        self.create_main_layout()

    def create_widgets(self):
        self.strength = self.create_double_spin_box("Strength", 0.00, 10.00, 0.01, 0.00)




@register_node(OP_NODE_CONDITIONING_COMBINE)
class ConditioningCombineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/cond_comb.png"
    op_code = OP_NODE_CONDITIONING_COMBINE
    op_title = "Combine Conditioning"
    content_label_objname = "cond_combine_node"
    category = "Conditioning"


    def __init__(self, scene):
        super().__init__(scene, inputs=[3,3,1], outputs=[3,1])
    def initInnerClasses(self):
        self.content = ConditioningCombineWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 128
        self.grNode.width = 320
        #self.content.setMinimumHeight(200)
        #self.content.setMinimumWidth(320)
        self.input_socket_name = ["EXEC", "COND", "COND2"]
        self.output_socket_name = ["EXEC", "COND"]
        self.content.eval_signal.connect(self.evalImplementation)


    def evalImplementation_thread(self, index=0):
        cond = self.combine_conditioning()

        return cond

    def onMarkedDirty(self):
        self.value = None
    def combine_conditioning(self, progress_callback=None):
        try:

            cond1_list = self.getInputData(1)
            cond2_list = self.getInputData(0)
            strength = self.content.strength.value()

            if strength > 0:
                c = self.addWeighted(cond1_list[0], cond2_list[0], strength)
                if gs.logging:
                    print("COND COMBINE NODE: Conditionings weighted.")
                return [c]
            else:
                c = cond1_list[0] + cond2_list[0]
                if gs.logging:
                    print("COND COMBINE NODE: Conditionings combined.")
                return [c]


        except Exception as e:
            print(f"COND COMBINE NODE: \nFailed: {e}")
            return None

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)
        return out

    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)

        # Update the node value and mark it as dirty
        self.setOutput(0, result)
        self.markDirty(False)
        self.markInvalid(False)
        self.executeChild(1)

    def onInputChanged(self, socket=None):
        pass
    def combine(self, conditioning_1, conditioning_2):
        return [conditioning_1 + conditioning_2]


@register_node(OP_NODE_CONDITIONING_SET_AREA)
class ConditioningAreaNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/cond_comb.png"
    op_code = OP_NODE_CONDITIONING_SET_AREA
    op_title = "Set Conditioning Area"
    content_label_objname = "cond_area_node"
    category = "Conditioning"

    def __init__(self, scene):
        super().__init__(scene, inputs=[3,1], outputs=[3,1])
    def initInnerClasses(self):
        self.content = ConditioningSetAreaWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.height = 256
        self.grNode.width = 320
        self.content.setMinimumHeight(200)
        self.content.setMinimumWidth(320)
        #self.content.button.clicked.connect(self.exec)
        self.input_socket_name = ["EXEC", "COND"]
        self.output_socket_name = ["EXEC", "COND"]
        self.content.eval_signal.connect(self.evalImplementation)


    @QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        #try:
        cond = self.append_conditioning()

        return cond
        #except:
        #    print("COND AREA NODE: Failed, please make sure that the conditioning is valid.")
        #    self.markDirty(True)
        #    return None
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)

        self.setOutput(0, result)
        if gs.logging:
            print("COND AREA NODE: Conditionings Area Set.")
        self.markDirty(False)
        self.executeChild(1)



    def append_conditioning(self, progress_callback=None, min_sigma=0.0, max_sigma=99.0):
        cond_node, index = self.getInput(0)
        conditioning_list = cond_node.getOutput(index)
        return_list = []
        for conditioning in conditioning_list:
            width = self.content.width.value()
            height = self.content.height.value()
            x = self.content.x_spinbox.value()
            y = self.content.y_spinbox.value()
            strength = self.content.strength.value()
            c = []
            for t in conditioning:
                n = [t[0], t[1].copy()]
                n[1]['area'] = (height // 8, width // 8, y // 8, x // 8)
                n[1]['strength'] = strength
                n[1]['min_sigma'] = min_sigma
                n[1]['max_sigma'] = max_sigma
                c.append(n)
            return_list.append(c)
        print(return_list)
        return return_list

