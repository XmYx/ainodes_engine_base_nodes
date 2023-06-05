from qtpy import QtCore
from qtpy import QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

import rtmidi

OP_NODE_MIDI = get_next_opcode()


class MidiWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()

    def create_widgets(self):
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.create_button_layout([self.run_button, self.stop_button])


@register_node(OP_NODE_MIDI)
class MidiNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/exec.png"
    op_code = OP_NODE_MIDI
    op_title = "Midi In"
    content_label_objname = "midi_node"
    category = "Experimental"
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
        super().__init__(scene, inputs=[1], outputs=[1])
        self.interrupt = False
        self.midi_in = None

    def initInnerClasses(self):
        self.content = MidiWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.height = 200
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(160)
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.run_button.clicked.connect(self.evalImplementation_thread)
        self.content.stop_button.clicked.connect(self.stop)


    def evalImplementation_thread(self, index=0, *args, **kwargs):

        # Initialize MIDI input
        self.midi_in = rtmidi.MidiIn()

        # Get the number of available ports
        port_count = self.midi_in.get_port_count()

        # Iterate over the ports to find a USB MIDI device
        for port_index in range(port_count):
            port_name = self.midi_in.get_port_name(port_index)
            if "USB MIDI" in port_name:
                print("Found USB MIDI device:", port_name)
                self.midi_in.open_port(port_index)
                self.midi_in.set_callback(self.midi_callback)
                break
        else:
            print("No USB MIDI device found.")


        #return True

    def onWorkerFinished(self, result):
        self.busy = False
        self.executeChild(0)

    def stop(self):
        print("Interrupting Execution of Graph")
        gs.should_run = None

        # Close MIDI input port
        if self.midi_in.is_port_open():
            self.midi_in.close_port()

    def midi_callback(self, message, timestamp):
        # Process the MIDI event
        # This function will be called when a MIDI event occurs
        print("MIDI Message:", message)

        # Extract MIDI event information
        status_byte = message[0] & 0xF0
        data_byte1 = message[1]
        data_byte2 = message[2]

        # TODO: Handle MIDI events and update your GUI controls accordingly

        # Example: Print note-on events
        if status_byte == 0x90:  # Note On message
            note_number = data_byte1
            velocity = data_byte2
            print("Note On - Note:", note_number, "Velocity:", velocity)

        # Example: Print control change events
        if status_byte == 0xB0:  # Control Change message
            control_number = data_byte1
            control_value = data_byte2
            print("Control Change - Control Number:", control_number, "Control Value:", control_value)
