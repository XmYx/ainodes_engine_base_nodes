import datetime
import os

from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QAudioOutput
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QSlider
from audiocraft.models import MusicGen
from qtpy.QtCore import QUrl
from qtpy.QtMultimedia import QMediaPlayer

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
import torch
from audiocraft.data.audio import audio_write

from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import torch_gc

#MANDATORY
OP_NODE_AUDIOCRAFT = get_next_opcode()


class MusicPlayer(QWidget):
    def __init__(self):
        super(MusicPlayer, self).__init__()

        self.player = QMediaPlayer(self)
        self.audioOutput = QAudioOutput()
        self.audioOutput.setVolume(50)
        self.player.setAudioOutput(self.audioOutput)
        layout = QVBoxLayout()

        self.play_button = QPushButton('Play')
        self.pause_button = QPushButton('Pause')
        self.stop_button = QPushButton('Stop')

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.sliderMoved.connect(self.seek_position)

        self.play_button.clicked.connect(self.play_music)
        self.pause_button.clicked.connect(self.pause_music)
        self.stop_button.clicked.connect(self.stop_music)

        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)

        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.timeline)
        self.setLayout(layout)

    def set_media(self, path):
        print("SOURCE SET TO ", path)
        self.player.setSource(QUrl.fromLocalFile(path))

    def play_music(self):
        print("PLAY TRIGGERED")
        self.player.play()

    def pause_music(self):
        self.player.pause()

    def stop_music(self):
        self.player.stop()

    def seek_position(self, position):
        self.player.setPosition(position)

    def position_changed(self, position):
        self.timeline.setValue(position)

    def duration_changed(self, duration):
        self.timeline.setRange(0, duration)
class AudiocraftWidget(QDMNodeContentWidget):
    def initUI(self):
        self.player = MusicPlayer()
        self.prompt = self.create_line_edit("Prompt", placeholder="Prompt")
        self.create_main_layout(grid=1)
        self.main_layout.addWidget(self.player)




#NODE CLASS
@register_node(OP_NODE_AUDIOCRAFT)
class DiffusersKarloUnclipNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Audiocraft - Music"
    op_code = OP_NODE_AUDIOCRAFT
    op_title = "Audiocraft Music Node"
    content_label_objname = "audiocraft_node"
    category = "Sampling"
    NodeContent_class = AudiocraftWidget
    dim = (340, 260)
    output_data_ports = [0]
    exec_port = 0

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[1])
        self.model = None

    #MAIN NODE FUNCTION
    def evalImplementation_thread(self, index=0):
        if not self.model:
            self.model = MusicGen.get_pretrained("melody")

        prompt = self.content.prompt.text()

        audio_path = self.predict(text=prompt, melody=None, duration=25, topk=250, topp=0, temperature=1.0, cfg_coef=3.0)
        print("DONE", audio_path)
        self.content.player.set_media(audio_path)
        return None

    def onWorkerFinished(self, result):
        self.busy = False
        pass

    def remove(self):
        print("REMOVING", self)
        del self.model
        torch_gc()
        super().remove()


    def predict(self, text, melody, duration, topk, topp, temperature, cfg_coef):

    
        if duration > self.model.lm.cfg.dataset.segment_duration:
            raise ValueError("MusicGen currently supports durations of up to 30 seconds!")
        self.model.set_generation_params(
            use_sampling=True,
            top_k=topk,
            top_p=topp,
            temperature=temperature,
            cfg_coef=cfg_coef,
            duration=duration,
        )
    
        if melody:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(self.model.device).float().t().unsqueeze(0)
            print(melody.shape)
            if melody.dim() == 2:
                melody = melody[None]
            melody = melody[..., :int(sr * self.model.lm.cfg.dataset.segment_duration)]
            output = self.model.generate_with_chroma(
                descriptions=[text],
                melody_wavs=melody,
                melody_sample_rate=sr,
                progress=False
            )
        else:
            output = self.model.generate(descriptions=[text], progress=False)

        output = output.detach().cpu().float()[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = "output/WAVs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{timestamp}.wav")
        print(output)
        #with open(output_path, "wb") as file:
        audio_write(output_path, output, self.model.sample_rate, strategy="loudness", add_suffix=False)

        return output_path