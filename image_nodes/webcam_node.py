import secrets

import cv2
import numpy as np
import torch
from PyQt6.QtCore import QThreadPool
from PyQt6.QtGui import QImage, QPixmap
from qtpy import QtWidgets, QtGui, QtCore
from torch import autocast

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.opt_ip2p_pipeline import StableDiffusionInstructPix2PixPipeline
from ainodes_frontend.base import register_node, get_next_opcode, Worker
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from PIL import Image

OP_NODE_WEBCAM = get_next_opcode()

class WebcamPreviewWidget(QDMNodeContentWidget):
    preview_signal = QtCore.Signal(object)
    decodevae = QtCore.Signal(object)

    def initUI(self):

        self.prompt = self.create_text_edit("Linguistic Prompt", placeholder="Linguistic Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=1024, step=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=1024, step=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=25, step=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.image_scale = self.create_double_spin_box("Image Guidance Scale", min_val=0.01, max_val=25.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")

        self.image = self.create_label("")
        self.available_cameras = self.get_available_cameras()
        self.dropdown_camera = self.create_combo_box(self.available_cameras, "Camera")

        # Button to start the webcam feed
        self.start_webcam_button = QtWidgets.QPushButton("Start Webcam")
        self.start_diffusion = QtWidgets.QPushButton("Start Diffusion")
        self.stop_diffusion = QtWidgets.QPushButton("Stop Diffusion")
        self.create_button_layout([self.start_webcam_button, self.start_diffusion, self.stop_diffusion])

        self.create_main_layout(grid=1)

    def get_available_cameras(self):
        # Use OpenCV to detect available cameras
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(f"Camera {index}")
            cap.release()
            index += 1
        return arr


@register_node(OP_NODE_WEBCAM)
class WebcamPreviewNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/image_preview.png"
    op_code = OP_NODE_WEBCAM
    op_title = "Webcam Preview"
    content_label_objname = "webcam_output_node"
    category = "aiNodes Base/Image"
    dims = (800, 600)
    NodeContent_class = WebcamPreviewWidget

    output_data_ports = [0]
    exec_port = 1

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])

        self.pipe = None
        self.run = False
        self.threadpool = QThreadPool()
        self.update_all_sockets()

        self.generator = torch.Generator(gs.device.type)


    def initInnerClasses(self):
        super().initInnerClasses()

        self.content.preview_signal.connect(self.show_image)
        self.content.start_webcam_button.clicked.connect(self.start_webcam_feed)
        self.content.start_diffusion.clicked.connect(self.start)
        self.content.stop_diffusion.clicked.connect(self.stop)
        self.content.decodevae.connect(self.decode_vae)

        self.grNode.height = self.dims[0]
        self.grNode.width = self.dims[1]
        self.content.setGeometry(0, 25, self.dims[1], self.dims[0])

    def start(self):

        worker = Worker(self.run_live)
        self.threadpool.start(worker)

        self.run = True
    def stop(self):
        self.run = False

    def get_params(self, frame):

        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        print("Using Seed", seed)
        return {"image":frame,
                "prompt":self.content.prompt.toPlainText(),
                "width":self.content.width_val.value(),
                "height":self.content.height_val.value(),
                "guidance_scale":self.content.scale.value(),
                "image_guidance_scale":self.content.image_scale.value(),
                "num_inference_steps":self.content.steps.value(),
                "generator":self.generator.manual_seed(seed)}

    def run_live(self, progress_callback=None):

        if self.pipe == None:
            model_id = "timbrooks/instruct-pix2pix"
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16, safety_checker=None
            ).to("cuda")

        while self.run:
            with torch.inference_mode():
                frame = self.frame_rgb
                seed = 0
                generator = torch.Generator(gs.device.type).manual_seed(seed)
                if frame is not None:
                    params = self.get_params(frame)

                    params["image"] = cv2.resize(frame, (params["width"], params["height"]),
                                           interpolation=cv2.INTER_LANCZOS4)


                    latents = self.pipe(**params)
                    self.content.decodevae.emit(latents)

    def decode_vae(self, latent):
        with autocast("cuda"):
            # 1. Scales the latent variable.
            latents = 1 / 0.18215 * latent

            # 2. Decodes the latent variable to get an image.
            image = self.pipe.vae.decode(latents).sample

            # 3. Clamps and scales the image values.
            image = (image / 2 + 0.5).clamp(0, 1)

            # 4. Convert tensor values to float32
            image =  image.permute(0, 2, 3, 1).float().detach().cpu()

            # 5. Add batch dimension (if it's removed in prior steps)
            if image.dim() == 3:
                image = image.unsqueeze(0)
        preview = self.getOutputs(0)
        if len(preview) > 0:
            for node in preview:
                from ai_nodes.ainodes_engine_base_nodes.image_nodes.image_preview_node import ImagePreviewNode
                if isinstance(node, ImagePreviewNode):
                    from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap
                    pixmap = tensor_image_to_pixmap(image)
                    node.content.preview_signal.emit(pixmap)

        #print(preview)
        self.init_avail = True
    def start_webcam_feed(self):
        cam_index = self.content.dropdown_camera.currentIndex()
        self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        self.webcam_timer = QtCore.QTimer()
        self.webcam_timer.timeout.connect(self.update_frame)
        self.webcam_timer.start(30)  # Update frame every 30 ms

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert to PIL Image and then to tensor
            self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # pil_img = Image.fromarray(frame_rgb)
            # tensor_frame = pil2tensor(pil_img)

            # Display on UI
            q_img = QImage(self.frame_rgb.data, self.frame_rgb.shape[1], self.frame_rgb.shape[0],
                           QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.resize(pixmap)
            self.content.preview_signal.emit(pixmap)


            # # Set tensor as output
            # self.setOutput(0, tensor_frame)

    def evalImplementation_thread(self):
        return [pil2tensor(Image.fromarray(self.frame_rgb.copy()))]


    def show_image(self, pixmap):
        self.content.image.setPixmap(pixmap)

    def resize(self, pixmap):
        dims = (pixmap.size().height() + 420, pixmap.size().width() + 30)
        if self.dims != dims:
            self.grNode.setToolTip("")
            self.grNode.height = dims[0]
            self.grNode.width = dims[1]
            self.content.setGeometry(0, 25, pixmap.size().width(), pixmap.size().height() + 150)
            self.update_all_sockets()
            self.dims = dims

    def remove(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'webcam_timer'):
            self.webcam_timer.stop()
        super().remove()
