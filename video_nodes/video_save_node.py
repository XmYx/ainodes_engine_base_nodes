import datetime
import os
import subprocess
import time

import cv2
import imageio
import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QVBoxLayout

from ..ainodes_backend import pixmap_to_pil_image

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

OP_NODE_VIDEO_SAVE = get_next_opcode()

class VideoOutputWidget(QDMNodeContentWidget):
    def initUI(self):
        self.video = GifRecorder()
        self.current_frame = 0
        self.type_select = QtWidgets.QComboBox()
        self.type_select.addItems(["GIF", "mp4_ffmpeg", "mp4_fourcc"])
        self.save_button = QPushButton("Save buffer to GIF", self)
        #self.new_button = QPushButton("New Video", self)
        #self.save_button.clicked.connect(self.loadVideo)

        self.width_value = QtWidgets.QSpinBox()
        self.width_value.setMinimum(64)
        self.width_value.setSingleStep(64)
        self.width_value.setMaximum(4096)
        self.width_value.setValue(512)

        self.height_value = QtWidgets.QSpinBox()
        self.height_value.setMinimum(64)
        self.height_value.setSingleStep(64)
        self.height_value.setMaximum(4096)
        self.height_value.setValue(512)

        self.fps = QtWidgets.QSpinBox()
        self.fps.setMinimum(1)
        self.fps.setSingleStep(1)
        self.fps.setMaximum(4096)
        self.fps.setValue(24)

        self.dump_at = self.create_spin_box("Dump at every:", 0, 20000, 0, 1)


        layout = QVBoxLayout()
        layout.addWidget(self.type_select)
        layout.addWidget(self.save_button)
        #layout.addWidget(self.width_value)
        #layout.addWidget(self.height_value)
        layout.addWidget(self.fps)
        layout.addWidget(self.dump_at)

        self.setLayout(layout)



@register_node(OP_NODE_VIDEO_SAVE)
class VideoOutputNode(AiNode):
    icon = "ainodes_frontend/icons/in.png"
    op_code = OP_NODE_VIDEO_SAVE
    op_title = "Video Save"
    content_label_objname = "video_output_node"
    category = "video"
    input_socket_name = ["EXEC", "IMAGE"]
    output_socket_name = ["EXEC", "IMAGE"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,1])
        self.filename = ""
        self.content.eval_signal.connect(self.evalImplementation)
        self.busy = False
    def initInnerClasses(self):
        self.content = VideoOutputWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.content.save_button.clicked.connect(self.start_new_video)
        self.grNode.height = 300
        self.grNode.width = 260

        self.content.setGeometry(0, 0, 260, 230)
        self.markInvalid(True)
    def evalImplementation(self, index=0):
        self.busy = True
        if self.getInput(0) is not None:
            input_node, other_index = self.getInput(0)
            if not input_node:
                self.grNode.setToolTip("Input is not connected")
                self.markInvalid()
                return

            val = input_node.getOutput(other_index)
            image = pixmap_to_pil_image(val)
            frame = np.array(image)
            self.markInvalid(False)
            self.markDirty(True)
            self.content.video.add_frame(frame, dump=self.content.dump_at.value())
            self.setOutput(0, val)
            if len(self.getOutputs(1)) > 0:
                node = self.getOutputs(1)[0]
                node.eval()
                while node.busy == True:
                    time.sleep(0.1)
                    self.busy = node.busy
        self.busy = False
        return None
    def close(self):
        self.content.video.close(self.filename)
        self.markDirty(False)
        self.markInvalid(False)
        self.start_new_video()
    def resize(self):
        self.content.setMinimumHeight(self.content.label.pixmap().size().height())
        self.content.setMinimumWidth(self.content.label.pixmap().size().width())
        self.grNode.height = self.content.label.pixmap().size().height() + 96
        self.grNode.width = self.content.label.pixmap().size().width() + 64

        for socket in self.outputs + self.inputs:
            socket.setSocketPosition()
        self.updateConnectedEdges()

    def eval(self):
        self.markDirty(True)
        self.content.eval_signal.emit()

    def start_new_video(self):
        self.markDirty(True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = f"output/gifs/{timestamp}.gif"
        fps = self.content.fps.value()
        type = self.content.type_select.currentText()
        self.content.video.close(timestamp, fps, type)
        print(f"VIDEO SAVE NODE: Done. The frame buffer is now empty.")

class VideoRecorder:

    def __init__(self):
        pass
    def start_recording(self, filename, fps, width, height):
        #fourcc = cv2.VideoWriter_fourcc(*'GIF')
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        #self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def add_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)
        print("VIDEO SAVE NODE: New frame added to video stream")

    def close(self, filename=""):
        self.video_writer.release()
        #os.rename("test.mp4", filename)
        print(f"VIDEO SAVE NODE: Video saved as {filename}")


class GifRecorder:

    def __init__(self):
        self.frames = []

    def start_recording(self, filename, fps, width, height):
        self.filename = filename
        self.fps = fps

    def add_frame(self, frame, dump):
        self.frames.append(frame)
        if len(self.frames) >= dump and dump != 0:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            fps = 24
            type = 'mp4_ffmpeg'
            self.close(timestamp, fps, type, True)

        print(f"VIDEO SAVE NODE: Image added to frame buffer, current frames: {len(self.frames)}")

    def close(self, timestamp, fps, type='GIF', dump=False):
        if type == 'GIF':
            os.makedirs("output/gifs", exist_ok=True)
            filename = f"output/gifs/{timestamp}.gif"
            self.filename = filename
            self.fps = fps
            print(f"VIDEO SAVE NODE: Video saving {len(self.frames)} frames at {self.fps}fps as {self.filename}")
            imageio.mimsave(self.filename, self.frames, fps=self.fps)
        elif type == 'mp4_ffmpeg':
            os.makedirs("output/mp4s", exist_ok=True)
            filename = f"output/mp4s/{timestamp}.mp4"
            width = self.frames[0].shape[1]
            height = self.frames[0].shape[0]

            print(width, height)
            cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt',
                   'rgb24', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-an',
                   filename]
            video_writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in self.frames:
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.stdin.write(frame.tobytes())
            video_writer.communicate()
        elif type == 'mp4_fourcc':
            os.makedirs("output/mp4s", exist_ok=True)
            filename = f"output/mp4s/{timestamp}.mp4"
            width = self.frames[0].shape[0]
            height = self.frames[0].shape[1]
            video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for frame in self.frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            video_writer.release()
        if dump == True:
            self.frames = []
