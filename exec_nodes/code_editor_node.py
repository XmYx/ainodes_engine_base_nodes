from qtpy import QtCore, QtGui
from qtpy import QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from PyQt6.Qsci import QsciScintilla, QsciLexerPython

from pyflakes.reporter import Reporter
import ast
from textwrap import dedent
class CustomReporter(Reporter):
    def __init__(self):
        super().__init__(None, None)
        self.errors = []

    def unexpectedError(self, filename, msg):
        self.errors.append(msg)

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.errors.append((filename, msg, lineno, offset, text))

    def flake(self, message):
        self.errors.append(message)

default_fn = """def customFunction(self):
    print("This is a susccesful test")
    return [None, None, None, None]"""

class PythonCodeEditor(QsciScintilla):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setUtf8(True)
        self.setIndentationsUseTabs(False)
        self.setIndentationWidth(4)
        self.setTabWidth(4)
        self.setIndentationGuides(True)
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setAutoIndent(True)
        # Setup syntax highlighter
        lexer = QsciLexerPython(self)
        self.setLexer(lexer)
        self.setMarginType(0,QsciScintilla.MarginType.NumberMargin)
        self.setMarginWidth(0, "0000")
        self.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle)

OP_NODE_VIM = get_next_opcode()

class VimWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
        self.grid_layout.addWidget(self.editor)
    def create_widgets(self):
        self.editor = PythonCodeEditor(parent=self)
        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.create_button_layout([self.run_button, self.stop_button])

    def serialize(self) -> dict:
        res = super().serialize()
        res["code"] = str(self.editor.text())
        return res

    def deserialize(self, data, hashmap={}, restore_id:bool=True) -> bool:
        if "code" in data:
            self.editor.setText(data["code"])
        else:
            self.editor.setText(default_fn)
        super().deserialize(data, hashmap, restore_id)
        return True


@register_node(OP_NODE_VIM)
class VimNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    op_code = OP_NODE_VIM
    op_title = "CodeEditor Node"
    content_label_objname = "code_editor_node"
    category = "Experimental"
    help_text = "Code Editor Node\n\n" \

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,3,2,6,1], outputs=[5,3,2,6,1])
        self.interrupt = False

    def initInnerClasses(self):
        self.content = VimWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)
        self.grNode.height = 600
        self.grNode.width = 1024
        self.content.setMinimumWidth(1024)
        self.content.setMinimumHeight(450)
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.run_button.clicked.connect(self.start)
        self.content.stop_button.clicked.connect(self.stop)


    def evalImplementation_thread(self, index=0, *args, **kwargs):
        function_string = dedent(self.content.editor.text())  # Get function string from the editor

        # Parse the function string into a Python function object
        function_definition = ast.parse(function_string, mode='exec')
        globals_ = {}
        exec(compile(function_definition, filename="<ast>", mode="exec"), globals_)
        self.origFunction = globals_['customFunction']  # new_function is assumed to be the name of your function

        # Call the new function
        return self.origFunction(self, *args, **kwargs)

    def origFunction(self):
        return True

    def onWorkerFinished(self, result):
        self.busy = False
        assert isinstance(result, list), "Result is not a list"
        assert len(result) == 4, "Please make sure to return a list of all 4 elements [data:dict, conditionings:List[Torch.tensor]], images:List[QPixmap], latents:List[Torch.tensor], even if they are None."
        self.setOutput(0, result[0])
        self.executeChild(4)

    def stop(self):
        print("Interrupting Execution of Graph")
        gs.should_run = None

    def start(self):
        gs.should_run = True
        self.content.eval_signal.emit()
