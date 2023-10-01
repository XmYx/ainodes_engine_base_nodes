import secrets
import time
from typing import Union, List

import cv2
import numpy as np
import torch
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QThreadPool
from PyQt6.QtGui import QImage, QPixmap
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionControlNetPipeline, \
    ControlNetModel
from qtpy import QtWidgets, QtGui, QtCore
from torch import autocast

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.interpolation.hybrid import optical_flow_cadence
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.interpolation.linear_interpolation import interpolate_linear
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.ip_adapter import IPAdapterPlus
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.opt_ip2p_pipeline import StableDiffusionInstructPix2PixPipeline
from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import get_scheduler, scheduler_type_values, \
    SchedulerType
from ainodes_frontend.base import register_node, get_next_opcode, Worker
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from PIL import Image

OP_NODE_WEBCAM = get_next_opcode()

class WebcamPreviewWidget(QDMNodeContentWidget):
    preview_signal = QtCore.Signal(object)
    decodevae = QtCore.Signal(object)
    update_image_signal = QtCore.Signal(object)
    fps_readout_signal = QtCore.Signal(str)
    cadence_fps_readout_signal = QtCore.Signal(str)

    def initUI(self):
        self.pipeline_select = self.create_combo_box(["instruct", "img2img", "txt2img", "txt2img_cnet"], "Select Pipeline:")
        self.enable_ipadapter = self.create_check_box("Enable IP Adapter")
        self.prompt = self.create_text_edit("Linguistic Prompt", placeholder="Linguistic Prompt")
        self.height_val = self.create_spin_box("Height", min_val=64, max_val=4096, default_val=384, step=64)
        self.width_val = self.create_spin_box("Width", min_val=64, max_val=4096, default_val=384, step=64)
        self.steps = self.create_spin_box("Steps", min_val=1, max_val=4096, default_val=10, step=1)
        self.scale = self.create_double_spin_box("Scale", min_val=0.01, max_val=25.00, default_val=7.5, step=0.01)
        self.image_scale = self.create_double_spin_box("Image Guidance Scale", min_val=0.01, max_val=25.00, default_val=1.0, step=0.01)
        self.seed = self.create_line_edit("Seed")
        self.blendimgs = self.create_double_spin_box("Blend A", min_val=0.00, max_val=10.00, default_val=1.0, step=0.01)
        self.blendimgs_2 = self.create_double_spin_box("Blend B", min_val=0.00, max_val=10.00, default_val=0.0, step=0.01)

        self.scheduler_name = self.create_combo_box(scheduler_type_values, "Scheduler")
        self.cadence_type = self.create_combo_box(["simple", "reallybigname"], "Cadence Type")
        self.cadence_quality = self.create_combo_box(["DIS Medium", "DIS Fast", "DIS UltraFast", "DenseRLOF", "SF", "Farneback Fine", "Normal"], "Cadence Quality")
        self.cadence = self.create_spin_box("Cadence", min_val=0, max_val=200, default_val=5, step=1)

        self.image = self.create_label("")
        self.cad_image = self.create_label("")
        self.fps_label = self.create_label("")
        self.cadence_fps_label = self.create_label("")
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

    output_data_ports = [0, 1]
    exec_port = 1

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,5,1])

        self.pipe = None
        self.run = False
        self.threadpool = QThreadPool()
        self.update_all_sockets()

        self.generator = torch.Generator(gs.device.type)
        self.scheduler = ""
        self.init = None
        self.cap = None
        self.images = []
        self.morphed_images = []
        self.loaded_pipeline = ""

        self.content.update_image_signal.connect(self.update_image)
        self.content.fps_readout_signal.connect(self.update_fps)
        self.content.cadence_fps_readout_signal.connect(self.update_cadence_fps)

    def initInnerClasses(self):
        super().initInnerClasses()

        self.content.preview_signal.connect(self.show_image)
        self.content.update_image_signal.connect(self.show_cadence_image)
        self.content.start_webcam_button.clicked.connect(self.start_webcam_feed)
        self.content.start_diffusion.clicked.connect(self.start)
        self.content.stop_diffusion.clicked.connect(self.stop)
        self.content.decodevae.connect(self.decode_vae)

        self.grNode.height = self.dims[0]
        self.grNode.width = self.dims[1]
        self.content.setGeometry(0, 25, self.dims[1], self.dims[0])

    def start(self):

        self.stop()
        self.frame_count = 0
        self.last_fps_update_time = time.time()

        self.avg_frame_count = 0
        self.last_avg_fps_update_time = time.time()
        self.avg_start_time = time.time()

        self.images = []
        self.morphed_images = []
        self.prev_flow = None

        worker = Worker(self.run_live)

        self.run = True
        self.init_avail = False

        self.threadpool.start(worker)

        self.iterator()

    def iterator(self, progress_callback=None):

        try:
            if hasattr(self, "timer"):
                self.timer.stop()
        except:
            pass

        self.index = 0
        if not hasattr(self, "timer"):
            self.timer = QtCore.QTimer()
            # Set the timer interval
            self.timer.setInterval(41)

            # Connect the timer's timeout signal to the update_image slot
            self.timer.timeout.connect(self.update_image)

        # Start the timer
        self.timer.start()

    def update_image(self):
        if self.run:
            self.index += 1
            donotdraw = None
            if self.index > len(self.morphed_images) - 1:
                self.index = len(self.morphed_images) - 1
                donotdraw = True
            if self.morphed_images:
                if not donotdraw:
                    array = self.morphed_images[self.index]
                    height, width, channel = array.shape
                    bytesPerLine = 3 * width
                    buffer = array.tobytes()
                    qImg = QImage(buffer, width, height, bytesPerLine, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qImg)
                    self.content.preview_signal.emit(pixmap)
                    self.resize(pixmap)

            # Update the frame count
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.last_fps_update_time

            # If more than 1 second has passed, update the FPS display
            if elapsed_time > 1.0:
                fps = self.frame_count / elapsed_time
                self.content.cadence_fps_readout_signal.emit(f"{fps:.2f} fps")
                self.frame_count = 0
                self.last_fps_update_time = current_time
        else:
            self.index = 0

    def stop(self):
        self.run = False
        self.init = None
        # Stop any existing workers
        if hasattr(self, 'threadpool') and self.threadpool:
            self.threadpool.clear()
            self.threadpool.waitForDone()

        # Stop any existing timers
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        try:
            self.cap.release()
        except:
            pass

        self.images = []
        self.morphed_images = []


    def get_params(self, frame):

        seed = secrets.randbelow(9999999999) if self.content.seed.text() == "" else int(self.content.seed.text())
        self.generator.manual_seed(seed)
        if not self.content.enable_ipadapter.isChecked():
            if isinstance(self.pipe, StableDiffusionInstructPix2PixPipeline):
                return {"image":frame,
                        "prompt":self.content.prompt.toPlainText(),
                        "width":self.content.width_val.value(),
                        "height":self.content.height_val.value(),
                        "guidance_scale":self.content.scale.value(),
                        "image_guidance_scale":self.content.image_scale.value(),
                        "num_inference_steps":self.content.steps.value(),
                        "generator":self.generator,
                        "scheduler_name":self.content.scheduler_name.currentText()}
            elif isinstance(self.pipe, StableDiffusionImg2ImgPipeline):

                return {"image":cv2.resize(frame, (self.content.width_val.value(), self.content.height_val.value()),
                                                  interpolation=cv2.INTER_LANCZOS4),
                        "prompt":self.content.prompt.toPlainText(),
                        "guidance_scale":self.content.scale.value(),
                        "strength":self.content.image_scale.value(),
                        "num_inference_steps":self.content.steps.value(),
                        "generator":self.generator,
                        "scheduler_name":self.content.scheduler_name.currentText(),
                        "clip_skip":2,
                        "output_type":"latent",
                        "return_dict":True}
            elif isinstance(self.pipe, StableDiffusionPipeline):
                return {"prompt":self.content.prompt.toPlainText(),
                        "guidance_scale":self.content.scale.value(),
                        "width": self.content.width_val.value(),
                        "height": self.content.height_val.value(),

                        "num_inference_steps":self.content.steps.value(),
                        "generator":self.generator,
                        "scheduler_name":self.content.scheduler_name.currentText(),
                        "clip_skip":2,
                        "output_type":"latent",
                        "return_dict":True}
            elif isinstance(self.pipe, StableDiffusionControlNetPipeline):

                image = np.array(frame)
                image = cv2.Canny(image, 100, 200,
                                  L2gradient=True)
                from ai_nodes.ainodes_engine_base_nodes.image_nodes.image_op_node import HWC3
                image = HWC3(image)
                pil_image = Image.fromarray(image)
                print("RETURNING CNET VALUES", pil_image)
                return {"prompt":str(self.content.prompt.toPlainText()),
                        "image":pil_image,
                        "guidance_scale":self.content.scale.value(),
                        # "width": self.content.width_val.value(),
                        # "height": self.content.height_val.value(),

                        "num_inference_steps":self.content.steps.value(),
                        "generator":self.generator,
                        "scheduler_name":self.content.scheduler_name.currentText(),
                        "clip_skip":2,
                        "output_type":"latent",
                        "return_dict":True}
        else:

            if isinstance(self.ip_pipe, IPAdapterPlus):
                if isinstance(self.pipe, StableDiffusionImg2ImgPipeline):
                    return {"pil_image":Image.fromarray(cv2.resize(frame, (self.content.width_val.value(), self.content.height_val.value()),
                                                      interpolation=cv2.INTER_LANCZOS4)),
                            "scale":self.content.image_scale.value(),
                            "prompt":self.content.prompt.toPlainText(),
                            "guidance_scale":self.content.scale.value(),
                            "strength":self.content.image_scale.value(),
                            "num_inference_steps":self.content.steps.value(),
                            "generator":self.generator,
                            "scheduler_name":self.content.scheduler_name.currentText(),
                            "clip_skip":2,
                            "output_type":"latent"
                            }
                elif isinstance(self.pipe, StableDiffusionPipeline):
                    return {"pil_image":Image.fromarray(cv2.resize(frame, (self.content.width_val.value(), self.content.height_val.value()),
                                                      interpolation=cv2.INTER_LANCZOS4)),
                            "scale":self.content.image_scale.value(),
                            "width": self.content.width_val.value(),
                            "height": self.content.height_val.value(),

                            "prompt":self.content.prompt.toPlainText(),
                            "guidance_scale":self.content.scale.value(),
                            "num_inference_steps":self.content.steps.value(),
                            "generator":self.generator,
                            "scheduler_name":self.content.scheduler_name.currentText(),
                            "clip_skip":2,
                            "output_type":"latent"
                            }
                elif isinstance(self.pipe, StableDiffusionControlNetPipeline):

                    image = np.array(frame)
                    image = cv2.Canny(image, 100, 200,
                                      L2gradient=True)
                    from ai_nodes.ainodes_engine_base_nodes.image_nodes.image_op_node import HWC3
                    image = HWC3(image)
                    pil_image = Image.fromarray(image)
                    pil_image.save("cannt_test.png", "PNG")
                    image = Image.fromarray(cv2.resize(frame, (self.content.width_val.value(), self.content.height_val.value()),
                                                      interpolation=cv2.INTER_LANCZOS4))
                    image.save("input_image.png", "PNG")

                    return {"pil_image":image,
                            "prompt": self.content.prompt.toPlainText(),
                            "image": pil_image,
                            "guidance_scale": self.content.scale.value(),
                            "width": self.content.width_val.value(),
                            "height": self.content.height_val.value(),

                            "num_inference_steps": self.content.steps.value(),
                            "generator": self.generator,
                            "scheduler_name": self.content.scheduler_name.currentText(),
                            "clip_skip": 2,
                            "output_type": "latent",
                            "return_dict": True}

    def run_live(self, progress_callback=None):
        self.start_webcam_feed()

        self.baseline = 12
        # Initialize the start time
        start_time = time.time()
        selected_pipeline = self.content.pipeline_select.currentText()
        addition = "ip_" if self.content.enable_ipadapter.isChecked() else ""
        selected_pipeline = f"{addition}{selected_pipeline}"

        if self.pipe == None or self.loaded_pipeline != selected_pipeline:

            self.loaded_pipeline = selected_pipeline


            if selected_pipeline == f"{addition}instruct":

                model_id = "timbrooks/instruct-pix2pix"
                self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16, safety_checker=None
                ).to("cuda")

            elif selected_pipeline == f"{addition}img2img":
                model_id = "stablediffusionapi/realistic-vision"
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16, safety_checker=None
                ).to("cuda")
            elif selected_pipeline == f"{addition}txt2img":
                model_id = "stablediffusionapi/realistic-vision"
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16, safety_checker=None
                ).to("cuda")
            elif selected_pipeline == f"{addition}txt2img_cnet":
                model_id = "stablediffusionapi/realistic-vision"
                self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                                  torch_dtype=torch.float16)
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    model_id, torch_dtype=torch.float16, safety_checker=None, controlnet=self.controlnet
                ).to("cuda")
            if self.content.enable_ipadapter.isChecked():
                from ai_nodes.ainodes_engine_base_nodes.diffusers_rework_nodes.diffusers_ip_adapter_node import \
                    download_ip_adapter_xl
                download_ip_adapter_xl(version="1.5")
                image_encoder_path = "models/ip_adapter/models/image_encoder"
                ip_ckpt = "models/ip_adapter/models/ip-adapter-plus_sd15.bin"
                device = gs.device
                from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.ip_adapter import IPAdapterPlus
                self.ip_pipe = IPAdapterPlus(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
                # if self.ip_pipe.device.type != "cuda":
                #     self.ip_pipe.to("cuda")
                #self.loaded_pipeline = f"ip_{self.loaded_pipeline}"

            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            # print(self.pipe.unet.conv_out.state_dict()["weight"].stride())  # (2880, 9, 3, 1)
            self.pipe.unet.to(memory_format=torch.channels_last)  # in-place operation
            # print(
            #     self.pipe.unet.conv_out.state_dict()["weight"].stride()
            # )  # (2880, 1, 960, 320) having a stride of 1 for the 2nd dimension proves that it works
            self.pipe.unet.eval()
            # print("Torch optimizations enabled.")

        self.run_inference()

    def run_inference(self):
        max_flow = 5  # Maximum motion allowed in pixels. Adjust as necessary.

        if self.run:
            frame = self.update_frame()
            if frame is not None:
                params = self.get_params(frame)

                if self.scheduler != params["scheduler_name"]:
                    scheduler_enum = SchedulerType(params["scheduler_name"])
                    self.pipe = get_scheduler(self.pipe, scheduler_enum)
                    self.scheduler = params["scheduler_name"]
                if "image" in params:
                    params["image"] = cv2.resize(np.array(params["image"]).astype(np.uint8), (self.content.width_val.value(), self.content.height_val.value()),
                                           interpolation=cv2.INTER_LANCZOS4)

                if self.init_avail:
                    lastinit = cv2.resize(self.init, (self.content.width_val.value(), self.content.height_val.value()),
                                          interpolation=cv2.INTER_LANCZOS4)
                    frame = cv2.resize(frame, (self.content.width_val.value(), self.content.height_val.value()),
                                          interpolation=cv2.INTER_LANCZOS4)

                    # Calculate Optical Flow


                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cur_gray = cv2.cvtColor(lastinit, cv2.COLOR_BGR2GRAY)

                    # Calculate the difference between current and previous frames
                    diff = np.mean(np.abs(cur_gray - prev_gray))
                    flow_scale = np.clip(1.0 - diff / 255.0, 0.5, 1.0)  # Feedback mechanism

                    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow *= flow_scale

                    # Limit Flow Magnitudes
                    flow = np.clip(flow, -max_flow, max_flow)

                    # Flow Regularization
                    flow = cv2.GaussianBlur(flow, (5, 5), 0)

                    #print("flow", flow.shape)

                    # Blend Flow Maps
                    if hasattr(self, 'prev_flow') and self.prev_flow is not None:
                        magnitude = np.linalg.norm(flow, axis=2)
                        confidence = np.exp(-magnitude)
                        blend_ratio = 0.5 * confidence
                        flow = blend_ratio[..., np.newaxis] * self.prev_flow + (1 - blend_ratio[..., np.newaxis]) * flow

                    # Warp the Current Frame using the Flow
                    h, w = flow.shape[:2]
                    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
                    displaced = flow_map + flow.reshape(-1, 2)
                    remapped = cv2.remap(frame, displaced[:, 1].reshape(h, w).astype(np.float32),
                                         displaced[:, 0].reshape(h, w).astype(np.float32),
                                         interpolation=cv2.INTER_LINEAR)

                    # Error Compensation
                    alpha = self.content.blendimgs.value()
                    remapped = alpha * remapped + (1 - alpha) * frame

                    print(remapped.shape)
                    self.prev_flow = flow

                    blend_value = self.content.blendimgs.value()  # Assuming this is the blend value you want to use

                    # Blending
                    beta = self.content.blendimgs_2.value()
                    if "image" in params:
                        params["image"] = beta * lastinit + (1 - beta) * remapped

                    # if "image" in params:
                    #     params["image"] = cv2.addWeighted(remapped, blend_value, lastinit, 1 - blend_value, 0)
                if isinstance(self.pipe, StableDiffusionControlNetPipeline):
                    params["image"] = Image.fromarray(params["image"])

                with torch.inference_mode():
                    from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.ip_adapter import IPAdapterPlus
                    if not self.content.enable_ipadapter.isChecked():
                        if isinstance(self.pipe, StableDiffusionInstructPix2PixPipeline):
                            latents = self.pipe(**params)
                        elif isinstance(self.pipe, StableDiffusionImg2ImgPipeline):
                            del params["scheduler_name"]
                            latents = self.pipe(**params)[0]
                        elif isinstance(self.pipe, StableDiffusionPipeline):
                            del params["scheduler_name"]
                            #del params["image"]
                            latents = self.pipe(**params)[0]
                        elif isinstance(self.pipe, StableDiffusionControlNetPipeline):
                            del params["scheduler_name"]

                            print("USING CNET PARAMS\n", params)
                            latents = self.pipe(**params)[0]
                    else:
                        if isinstance(self.ip_pipe, IPAdapterPlus):
                            del params["scheduler_name"]
                            if not isinstance(self.pipe, StableDiffusionControlNetPipeline):
                                del params["image"]
                            latents = self.ip_pipe.generate(**params)


                self.content.decodevae.emit(latents)
                # Calculate FPS
                # Update the average frame count
                self.avg_frame_count += 1
                current_time = time.time()
                elapsed_time_since_last_avg_fps_update = current_time - self.last_avg_fps_update_time

                # If more than 1 second has passed, update the FPS display
                if elapsed_time_since_last_avg_fps_update > 1.0:
                    avg_fps = self.avg_frame_count / elapsed_time_since_last_avg_fps_update
                    self.content.fps_readout_signal.emit(f"{avg_fps:.2f} fps")
                    self.avg_frame_count = 0
                    self.last_avg_fps_update_time = current_time

                self.run_inference()

    @torch.no_grad()
    def decode_vae(self, latent):
        with autocast("cuda"):
            latent = 1 / 0.18215 * latent
            image = self.pipe.vae.decode(latent).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.permute(0, 2, 3, 1).float().cpu().numpy()
            image = (image * 255).astype("uint8")[0]
            #image = image.astype("uint8")[0]

            # if hasattr(self, "prev_flow"):
            #     h, w = self.prev_flow.shape[:2]
            #     flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
            #     displaced = flow_map + self.prev_flow.reshape(-1, 2)
            #     image = cv2.remap(image, displaced[:, 1].reshape(h, w).astype(np.float32),
            #                       displaced[:, 0].reshape(h, w).astype(np.float32), interpolation=cv2.INTER_LINEAR)

            self.images.append(image)
            self.init = image
            self.init_avail = True

        preview = self.getOutputs(0)
        if len(preview) > 0:
            for node in preview:
                from ai_nodes.ainodes_engine_base_nodes.image_nodes.image_preview_node import ImagePreviewNode
                if isinstance(node, ImagePreviewNode):
                    from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap
                    pixmap = tensor_image_to_pixmap(image)
                    node.content.preview_signal.emit(pixmap)

        if len(self.images) >= 2:
            self.try_for_morph()
    def try_for_morph(self):
        # print(type(self.images_2))
        img1 = self.images[len(self.images) - 1]
        img2 = self.images[len(self.images) - 2]

        with autocast("cuda"):
            if self.content.cadence_type.currentText() == 'simple':
                self.morphed_images = self.add_frames_linear_interp([img2, img1], nmb_frames_target=self.content.cadence.value())
            else:
                self.morphed_images = optical_flow_cadence(img2, img1, self.content.cadence.value(),
                                                           method=self.content.cadence_quality.currentText())
        self.morphed_images.append(img1)
        self.index = 1
        self.images = [self.images[1]]

    def start_webcam_feed(self):
        cam_index = self.content.dropdown_camera.currentIndex()
        self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert to PIL Image and then to tensor
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            return None

    def evalImplementation_thread(self):

        if self.cap == None:
            self.start_webcam_feed()
        frame = self.update_frame()
        result = pil2tensor(Image.fromarray(frame))

        print("webcamnode", type(result))

        return [result, None]


    def add_frames_linear_interp(
            self,
            list_imgs: List[np.ndarray],
            fps_target: Union[float, int] = None,
            duration_target: Union[float, int] = None,
            nmb_frames_target: int = None,
    ):
        r"""
        Helper function to cheaply increase the number of frames given a list of images,
        by virtue of standard linear interpolation.
        The number of inserted frames will be automatically adjusted so that the total of number
        of frames can be fixed precisely, using a random shuffling technique.
        The function allows 1:1 comparisons between transitions as videos.

        Args:
            list_imgs: List[np.ndarray)
                List of images, between each image new frames will be inserted via linear interpolation.
            fps_target:
                OptionA: specify here the desired frames per second.
            duration_target:
                OptionA: specify here the desired duration of the transition in seconds.
            nmb_frames_target:
                OptionB: directly fix the total number of frames of the output.
        """

        # Sanity
        if nmb_frames_target is not None and fps_target is not None:
            raise ValueError("You cannot specify both fps_target and nmb_frames_target")
        if fps_target is None:
            assert nmb_frames_target is not None, "Either specify nmb_frames_target or nmb_frames_target"
        if nmb_frames_target is None:
            assert fps_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
            assert duration_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
            nmb_frames_target = fps_target * duration_target

        # Get number of frames that are missing
        nmb_frames_diff = len(list_imgs) - 1
        nmb_frames_missing = nmb_frames_target - nmb_frames_diff - 1

        if nmb_frames_missing < 1:
            return list_imgs

        list_imgs_float = [img.astype(np.float32) for img in list_imgs]

        # Distribute missing frames, append nmb_frames_to_insert(i) frames for each frame
        mean_nmb_frames_insert = nmb_frames_missing / nmb_frames_diff
        constfact = np.floor(mean_nmb_frames_insert)
        remainder_x = 1 - (mean_nmb_frames_insert - constfact)

        nmb_iter = 0
        while True:
            nmb_frames_to_insert = np.random.rand(nmb_frames_diff)
            nmb_frames_to_insert[nmb_frames_to_insert <= remainder_x] = 0
            nmb_frames_to_insert[nmb_frames_to_insert > remainder_x] = 1
            nmb_frames_to_insert += constfact
            if np.sum(nmb_frames_to_insert) == nmb_frames_missing:
                break
            nmb_iter += 1
            if nmb_iter > 100000:
                print("add_frames_linear_interp: issue with inserting the right number of frames")
                break

        nmb_frames_to_insert = nmb_frames_to_insert.astype(np.int32)
        list_imgs_interp = []
        for i in range(len(list_imgs_float) - 1):  # , desc="STAGE linear interp"):
            img0 = list_imgs_float[i]
            img1 = list_imgs_float[i + 1]
            list_imgs_interp.append(img0.astype(np.uint8))
            list_fracts_linblend = np.linspace(0, 1, nmb_frames_to_insert[i] + 2)[1:-1]
            for fract_linblend in list_fracts_linblend:
                img_blend = interpolate_linear(img0, img1, fract_linblend).astype(np.uint8)
                list_imgs_interp.append(img_blend.astype(np.uint8))

            if i == len(list_imgs_float) - 2:
                list_imgs_interp.append(img1.astype(np.uint8))

        return list_imgs_interp



    def show_image(self, pixmap):
        self.content.image.setPixmap(pixmap)
    def show_cadence_image(self, pixmap):
        self.content.cad_image.setPixmap(pixmap)

    def update_fps(self, text):
        self.content.fps_label.setText(text)
    def update_cadence_fps(self, text):
        self.content.cadence_fps_label.setText(text)

    def resize(self, pixmap):
        dims = (pixmap.size().height() + 700, pixmap.size().width() + 30)
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
