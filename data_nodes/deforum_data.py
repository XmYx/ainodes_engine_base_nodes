import json
import os
import re
import shutil
import sys
import time
from functools import partial
from types import SimpleNamespace

import numpy as np
from qtpy import QtCore
from qtpy import QtWidgets

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap
from custom_nodes.ainodes_engine_base_nodes.image_nodes.output import ImagePreviewNode
from custom_nodes.ainodes_engine_base_nodes.video_nodes.video_save_node import VideoOutputNode

OP_NODE_DEFORUM_DATA = get_next_opcode()
OP_NODE_DEFORUM_ARGS_DATA = get_next_opcode()
OP_NODE_DEFORUM_PROMPT = get_next_opcode()

def get_os():
    import platform
    return {"Windows": "Windows", "Linux": "Linux", "Darwin": "Mac"}.get(platform.system(), "Unknown")

def custom_placeholder_format(value_dict, placeholder_match):
    key = placeholder_match.group(1).lower()
    value = value_dict.get(key, key) or "_"
    if isinstance(value, dict) and value:
        first_key = list(value.keys())[0]
        value = str(value[first_key][0]) if isinstance(value[first_key], list) and value[first_key] else str(value[first_key])
    return str(value)[:50]

def test_long_path_support(base_folder_path):
    long_folder_name = 'A' * 300
    long_path = os.path.join(base_folder_path, long_folder_name)
    try:
        os.makedirs(long_path)
        shutil.rmtree(long_path)
        return True
    except OSError:
        return False
def get_max_path_length(base_folder_path):
    if get_os() == 'Windows':
        return (32767 if test_long_path_support(base_folder_path) else 260) - len(base_folder_path) - 1
    return 4096 - len(base_folder_path) - 1

def substitute_placeholders(template, arg_list, base_folder_path):
    import re
    # Find and update timestring values if resume_from_timestring is True
    resume_from_timestring = next((arg_obj.resume_from_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_from_timestring')), False)
    resume_timestring = next((arg_obj.resume_timestring for arg_obj in arg_list if hasattr(arg_obj, 'resume_timestring')), None)

    if resume_from_timestring and resume_timestring:
        for arg_obj in arg_list:
            if hasattr(arg_obj, 'timestring'):
                arg_obj.timestring = resume_timestring

    max_length = get_max_path_length(base_folder_path)
    values = {attr.lower(): getattr(arg_obj, attr)
              for arg_obj in arg_list
              for attr in dir(arg_obj) if not callable(getattr(arg_obj, attr)) and not attr.startswith('__')}
    formatted_string = re.sub(r"{(\w+)}", lambda m: custom_placeholder_format(values, m), template)
    formatted_string = re.sub(r'[<>:"/\\|?*\s,]', '_', formatted_string)
    return formatted_string[:max_length]

deforum_args_layout = {
                "animation_mode": {
                    "type": "dropdown",
                    "choices": ["None", "2D", "3D", "Video Input", "Interpolation"]
                },
                "max_frames": {
                    "type": "spinbox",
                    "min": 1,
                    "max": 2048,
                    "default": 120,
                    "step": 1
                },
                "border": {
                    "type": "dropdown",
                    "choices": ["wrap", "replicate"]
                },
                "angle": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "zoom": {
                    "type": "lineedit",
                    "default": "0:(1.0025+0.002*sin(1.25*3.14*t/30))"
                },
                "translation_x": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "translation_y": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "translation_z": {
                    "type": "lineedit",
                    "default": "0:(1.75)"
                },
                "transform_center_x": {
                    "type": "lineedit",
                    "default": "0:(0.5)"
                },
                "transform_center_y": {
                    "type": "lineedit",
                    "default": "0:(0.5)"
                },
                "rotation_3d_x": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "rotation_3d_y": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "rotation_3d_z": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "enable_perspective_flip": {
                    "type": "checkbox",
                    "default": False
                },
                "perspective_flip_theta": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "perspective_flip_phi": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "perspective_flip_gamma": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "perspective_flip_fv": {
                    "type": "lineedit",
                    "default": "0:(53)"
                },
                "noise_schedule": {
                    "type": "lineedit",
                    "default": "0: (0.065)"
                },
                "strength_schedule": {
                    "type": "lineedit",
                    "default": "0: (0.65)"
                },
                "contrast_schedule": {
                    "type": "lineedit",
                    "default": "0: (1.0)"
                },
                "cfg_scale_schedule": {
                    "type": "lineedit",
                    "default": "0: (7)"
                },
                "enable_steps_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "steps_schedule": {
                    "type": "lineedit",
                    "default": "0: (25)"
                },
                "fov_schedule": {
                    "type": "lineedit",
                    "default": "0: (70)"
                },
                "aspect_ratio_schedule": {
                    "type": "lineedit",
                    "default": "0: (1)"
                },
                "aspect_ratio_use_old_formula": {
                    "type": "checkbox",
                    "default": False
                },
                "near_schedule": {
                    "type": "lineedit",
                    "default": "0: (200)"
                },
                "far_schedule": {
                    "type": "lineedit",
                    "default": "0: (10000)"
                },
                "seed_schedule": {
                    "type": "lineedit",
                    "default": "0:(s), 1:(-1), \"max_f-2\":(-1), \"max_f-1\":(s)"
                },
                "pix2pix_img_cfg_scale": {
                    "type": "doublespinbox",
                    "min": 0.0,
                    "max": 10.0,
                    "default": 1.5,
                    "step": 0.1
                },
                "pix2pix_img_cfg_scale_schedule": {
                    "type": "lineedit",
                    "default": "0:(1.5)"
                },
                "enable_subseed_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "subseed_schedule": {
                    "type": "lineedit",
                    "default": "0:(1)"
                },
                "subseed_strength_schedule": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "enable_sampler_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "sampler_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"Euler a\")"
                },
                "use_noise_mask": {
                    "type": "checkbox",
                    "default": False
                },
                "mask_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"{video_mask}\")"
                },
                "noise_mask_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"{video_mask}\")"
                },
                "enable_checkpoint_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "checkpoint_schedule": {
                    "type": "lineedit",
                    "default": "0: (\"model1.ckpt\"), 100: (\"model2.safetensors\")"
                },
                "enable_clipskip_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "clipskip_schedule": {
                    "type": "lineedit",
                    "default": "0: (2)"
                },
                "enable_noise_multiplier_scheduling": {
                    "type": "checkbox",
                    "default": True
                },
                "noise_multiplier_schedule": {
                    "type": "lineedit",
                    "default": "0: (1.05)"
                },
                "amount_schedule": {
                    "type": "lineedit",
                    "default": "0: (0.1)"
                },
                "kernel_schedule": {
                    "type": "lineedit",
                    "default": "0: (5)"
                },
                "sigma_schedule": {
                    "type": "lineedit",
                    "default": "0: (1.0)"
                },
                "threshold_schedule": {
                    "type": "lineedit",
                    "default": "0: (0.0)"
                },
                "hybrid_comp_alpha_schedule": {
                    "type": "lineedit",
                    "default": "0:(0.5)"
                },
                "hybrid_comp_mask_blend_alpha_schedule": {
                    "type": "lineedit",
                    "default": "0:(0.5)"
                },
                "hybrid_comp_mask_contrast_schedule": {
                    "type": "lineedit",
                    "default": "0:(1)"
                },
                "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {
                    "type": "lineedit",
                    "default": "0:(100)"
                },
                "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "hybrid_flow_factor_schedule": {
                    "type": "lineedit",
                    "default": "0:(1)"
                },
                "color_coherence": {
                    "type": "dropdown",
                    "choices": ["None", "HSV", "LAB", "RGB", "Video Input", "Image"]
                },
                "color_coherence_image_path": {
                    "type": "lineedit",
                    "default": ""
                },
                "color_coherence_video_every_N_frames": {
                    "type": "spinbox",
                    "min": 1,
                    "max": 100,
                    "default": 1,
                    "step": 1
                },
                "color_force_grayscale": {
                    "type": "checkbox",
                    "default": False
                },
                "legacy_colormatch": {
                    "type": "checkbox",
                    "default": False
                },
                "diffusion_cadence": {
                    "type": "spinbox",
                    "min": 1,
                    "max": 100,
                    "default": 4,
                    "step": 1
                },
                "optical_flow_cadence": {
                    "type": "dropdown",
                    "choices": ["None", "RAFT", "DIS Medium", "DIS Fine", "Farneback"]
                },
                "cadence_flow_factor_schedule": {
                    "type": "lineedit",
                    "default": "0: (1)"
                },
                "optical_flow_redo_generation": {
                    "type": "dropdown",
                    "choices": ["None", "RAFT", "DIS Medium", "DIS Fine", "Farneback"]
                },
                "redo_flow_factor_schedule": {
                    "type": "lineedit",
                    "default": "0: (1)"
                },
                "diffusion_redo": {
                    "type": "lineedit",
                    "default": "0"
                },
                "noise_type": {
                    "type": "dropdown",
                    "choices": ["uniform", "perlin"]
                },
                "perlin_w": {
                    "type": "spinbox",
                    "min": 1,
                    "max": 100,
                    "default": 8,
                    "step": 1
                },
                "perlin_h": {
                    "type": "spinbox",
                    "min": 1,
                    "max": 100,
                    "default": 8,
                    "step": 1
                },
                "perlin_octaves": {
                    "type": "spinbox",
                    "min": 1,
                    "max": 10,
                    "default": 4,
                    "step": 1
                },
                "perlin_persistence": {
                    "type": "doublespinbox",
                    "min": 0.1,
                    "max": 1.0,
                    "default": 0.5,
                    "step": 0.1
                },
                "use_depth_warping": {
                    "type": "checkbox",
                    "default": True
                },
                "depth_algorithm": {
                    "type": "dropdown",
                    "choices": [
                        "Zoe",
                        "Midas-3-Hybrid",
                        "Midas+AdaBins (old)",
                        "Zoe+AdaBins (old)",
                        "Midas-3.1-BeitLarge",
                        "AdaBins",
                        "Leres"
                    ],
                    "default": "Zoe"
                },
                "midas_weight": {
                    "type": "doublespinbox",
                    "min": 0,
                    "max": 1,
                    "default": 0.2,
                    "step": 0.01
                },
                "padding_mode": {
                    "type": "dropdown",
                    "choices": ["border", "reflection", "zeros"],
                    "default": "border"
                },
                "sampling_mode": {
                    "type": "dropdown",
                    "choices": ["bicubic", "bilinear", "nearest"],
                    "default": "bicubic"
                },
                "save_depth_maps": {
                    "type": "checkbox",
                    "default": False
                },
                "video_init_path": {
                    "type": "lineedit",
                    "default": "https://deforum.github.io/a1/V1.mp4"
                },
                "extract_nth_frame": {
                    "type": "spinbox",
                    "min": 1,
                    "max": 100,
                    "default": 1,
                    "step": 1
                },
                "extract_from_frame": {
                    "type": "spinbox",
                    "min": 0,
                    "max": 1000,
                    "default": 0,
                    "step": 1
                },
                "extract_to_frame": {
                    "type": "spinbox",
                    "min": -1,
                    "max": 1000,
                    "default": -1,
                    "step": 1
                },
                "overwrite_extracted_frames": {
                    "type": "checkbox",
                    "default": True
                },
                "use_mask_video": {
                    "type": "checkbox",
                    "default": False
                },
                "video_mask_path": {
                    "type": "lineedit",
                    "default": "https://deforum.github.io/a1/VM1.mp4"
                },
                "hybrid_generate_inputframes": {
                    "type": "checkbox",
                    "default": False
                },
                "hybrid_generate_human_masks": {
                    "type": "dropdown",
                    "choices": ["None", "PNGs", "Video", "Both"],
                    "default": "None"
                },
                "hybrid_use_first_frame_as_init_image": {
                    "type": "checkbox",
                    "default": True
                },
                "hybrid_motion": {
                    "type": "dropdown",
                    "choices": ["None", "Optical Flow", "Perspective", "Affine"],
                    "default": "None"
                },
                "hybrid_motion_use_prev_img": {
                    "type": "checkbox",
                    "default": False
                },
                "hybrid_flow_consistency": {
                    "type": "checkbox",
                    "default": False
                },
                "hybrid_consistency_blur": {
                    "type": "spinbox",
                    "min": 0,
                    "max": 10,
                    "default": 2,
                    "step": 1
                },
                "hybrid_flow_method": {
                    "type": "dropdown",
                    "choices": ["RAFT", "DIS Medium", "DIS Fine", "Farneback"],
                    "default": "RAFT"
                },
                "hybrid_composite": {
                    "type": "dropdown",
                    "choices": ["None", "Normal", "Before Motion", "After Generation"],
                    "default": "None"
                },
                "hybrid_use_init_image": {
                    "type": "checkbox",
                    "default": False
                },
                "hybrid_comp_mask_type": {
                    "type": "dropdown",
                    "choices": ["None", "Depth", "Video Depth", "Blend", "Difference"],
                    "default": "None"
                },
                "hybrid_comp_mask_inverse": {
                    "type": "checkbox",
                    "default": False
                },
                "hybrid_comp_mask_equalize": {
                    "type": "dropdown",
                    "choices": ["None", "Before", "After", "Both"],
                    "default": "None"
                },
                "hybrid_comp_mask_auto_contrast": {
                    "type": "checkbox",
                    "default": False
                },
                "hybrid_comp_save_extra_frames": {
                    "type": "checkbox",
                    "default": False
                },
                "resume_from_timestring": {
                    "type": "checkbox",
                    "default": False
                },
                "resume_timestring": {
                    "type": "lineedit",
                    "default": "20230129210106"
                },
                "enable_ddim_eta_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "ddim_eta_schedule": {
                    "type": "lineedit",
                    "default": "0:(0)"
                },
                "enable_ancestral_eta_scheduling": {
                    "type": "checkbox",
                    "default": False
                },
                "ancestral_eta_schedule": {
                    "type": "lineedit",
                    "default": "0:(1)"
        }
    }


deforum_args_2_layout = {
    "W": {
      "type": "spinbox",
      "default": 512,
      "min": 64,
      "max": 4096,
      "step": 64
    },
    "H": {
      "type": "spinbox",
      "default": 512,
      "min": 64,
      "max": 4096,
      "step": 64
    },
    "seed": {
      "type": "lineedit",
      "default": "-1",
    },
    "sampler": {
      "type": "lineedit",
      "default": "euler_ancestral"
    },
    "steps": {
      "type": "spinbox",
      "default": 25,
      "min": 0,
      "max": 10000,
      "step": 1
    },
    "scale": {
      "type": "spinbox",
      "default": 7,
      "min": 0,
      "max": 10000,
      "step": 1
    },
    "save_settings": {
      "type": "checkbox",
      "default": True
    },
    "save_sample_per_step": {
      "type": "checkbox",
      "default": False
    },
    "prompt_weighting": {
      "type": "checkbox",
      "default": False
    },
    "normalize_prompt_weights": {
      "type": "checkbox",
      "default": True
    },
    "log_weighted_subprompts": {
      "type": "checkbox",
      "default": False
    },
    "n_batch": {
      "type": "spinbox",
      "default": 1,
      "min": 0,
      "max": 10000,
      "step": 1
    },
    "batch_name": {
      "type": "lineedit",
      "default": "Deforum_{timestring}"
    },
    "seed_behavior": {
      "type": "dropdown",
      "choices": ["iter","fixed","random","ladder","alternate","schedule"]
    },
    "seed_iter_N": {
      "type": "spinbox",
      "default": 1,
      "min": 0,
      "max": 10000,
      "step": 1
    },
    "outdir": {
      "type": "lineedit",
      "default": "output/deforum"
    },
    "use_init": {
      "type": "checkbox",
      "default": False
    },
    "strength": {
      "type": "doublespinbox",
      "default": 0.8,
      "min": 0,
      "max": 1,
      "step": 0.01
    },
    "strength_0_no_init": {
      "type": "checkbox",
      "default": True
    },
    "init_image": {
      "type": "lineedit",
      "default": "https://deforum.github.io/a1/I1.png"
    },
    "use_mask": {
      "type": "checkbox",
      "default": False
    },
    "use_alpha_as_mask": {
      "type": "checkbox",
      "default": False
    },
    "mask_file": {
      "type": "lineedit",
      "default": "https://deforum.github.io/a1/M1.jpg"
    },
    "invert_mask": {
      "type": "checkbox",
      "default": False
    },
    "mask_contrast_adjust": {
      "type": "doublespinbox",
      "default": 1.0,
      "min": 0,
      "max": 10,
      "step": 0.1
    },
    "mask_brightness_adjust": {
      "type": "doublespinbox",
      "default": 1.0,
      "min": 0,
      "max": 10,
      "step": 0.1
    },
    "overlay_mask": {
      "type": "checkbox",
      "default": True
    },
    "mask_overlay_blur": {
      "type": "spinbox",
      "default": 4,
      "min": 0,
      "max": 100,
      "step": 1
    },
    "fill": {
      "type": "spinbox",
      "default": 1,
      "min": 0,
      "max": 10000,
      "step": 1
    },
    "full_res_mask": {
      "type": "checkbox",
      "default": True
    },
    "full_res_mask_padding": {
      "type": "spinbox",
      "default": 4,
      "min": 0,
      "max": 10000,
      "step": 1
    },
    "prompt": {
      "type": "lineedit",
      "default": ""
    },
    "timestring": {
      "type": "lineedit",
      "default": ""
    },
    "init_sample": {
      "type": "lineedit",
      "default": None
    },
    "mask_image": {
      "type": "lineedit",
      "default": None
    },
    "noise_mask": {
      "type": "lineedit",
      "default": None
    },
    "seed_internal": {
      "type": "spinbox",
      "default": 0,
      "min": 0,
      "max": 10000,
      "step": 1
    }
  }
def make_valid_json_string(json_like_string):
    # Remove whitespace and line breaks
    json_string = re.sub(r"\s", "", json_like_string)

    # Check if the string starts with '{' and ends with '}'
    if not json_string.startswith("{") or not json_string.endswith("}"):
        raise ValueError("Invalid JSON-like structure")

    # Remove outer curly braces
    json_string = json_string[1:-1]

    # Add double quotes around keys and values
    json_string = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'"\1":', json_string)
    json_string = re.sub(r":\s*([a-zA-Z_][a-zA-Z0-9_]*)", r':"\1"', json_string)

    # Replace single quotes with double quotes
    json_string = json_string.replace("'", '"')

    return json_string

class DeforumPromptWidget(QDMNodeContentWidget):
    def initUI(self):
        self.createUI()

    def createUI(self):
        self.data = {}
        self.row_widgets = []

        self.layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(self.layout)

        self.addButton = QtWidgets.QPushButton("Add Row")
        self.addButton.clicked.connect(self.add_row)
        self.layout.addWidget(self.addButton)

        self.loadButton = QtWidgets.QPushButton("Load Data")
        self.loadButton.clicked.connect(self.load_data)
        self.layout.addWidget(self.loadButton)

    def add_row(self):
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)

        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(0, 90)
        row_layout.addWidget(spinbox)

        textedit = QtWidgets.QTextEdit()
        row_layout.addWidget(textedit)

        removeButton = QtWidgets.QPushButton("Remove")
        removeButton.clicked.connect(lambda: self.remove_row(row_widget))
        row_layout.addWidget(removeButton)

        self.row_widgets.append(row_widget)
        self.layout.insertWidget(self.layout.count() - 1, row_widget)

    def serialize(self):
        res = self.get_values()
        return res

    def deserialize(self, data, hashmap={}, restore_id:bool=True):
        #self.clear_rows()


        for value, text in data.items():
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            spinbox = QtWidgets.QSpinBox()
            spinbox.setRange(0, 10000)
            spinbox.setValue(int(value))
            row_layout.addWidget(spinbox)

            textedit = QtWidgets.QTextEdit()
            textedit.setPlainText(text.replace("\"", ""))
            row_layout.addWidget(textedit)

            removeButton = QtWidgets.QPushButton("Remove")
            removeButton.clicked.connect(partial(self.remove_row, row_widget))
            row_layout.addWidget(removeButton)

            self.row_widgets.append(row_widget)
            self.layout.insertWidget(self.layout.count() - 1, row_widget)
        super().deserialize(data, hashmap={}, restore_id=True)

    def remove_row(self, row_widget):

        self.row_widgets.remove(row_widget)
        self.layout.removeWidget(row_widget)
        row_widget.deleteLater()

    def get_values(self):
        self.data = {}
        for i, row_widget in enumerate(self.row_widgets):
            spinbox = row_widget.layout().itemAt(0).widget()
            textedit = row_widget.layout().itemAt(1).widget()

            value = spinbox.value()
            text = textedit.toPlainText()

            # Encode the text in a JSON-friendly format
            encoded_text = json.dumps(text)

            self.data[str(value)] = encoded_text

        return self.data

    def clear_rows(self):
        for row_widget in self.row_widgets:
            self.layout.removeWidget(row_widget)
            row_widget.deleteLater()
        self.row_widgets = []

        #self.update_data()

    def get_values(self):
        self.data = {}
        for i, row_widget in enumerate(self.row_widgets):
            spinbox = row_widget.layout().itemAt(0).widget()
            textedit = row_widget.layout().itemAt(1).widget()

            value = spinbox.value()
            text = textedit.toPlainText()

            # Encode the text in a JSON-friendly format
            encoded_text = json.dumps(text)

            self.data[str(value)] = encoded_text

        return self.data

    def load_data(self):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Load Data")

        layout = QtWidgets.QVBoxLayout(dialog)

        textedit = QtWidgets.QTextEdit()
        layout.addWidget(textedit)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            json_data = textedit.toPlainText()
            data = json.loads(json_data)
            self.deserialize(data)

class DeforumDataWidget(QDMNodeContentWidget):
    def initUI(self):
        self.createUI()
        self.create_main_layout(grid=3)
        print(self.get_values())

    def createUI(self):
        for key, value in deforum_args_layout.items():
            t = value["type"]
            if t == "dropdown":
                self.create_combo_box(value["choices"], f"{key}_value_combobox", accessible_name=key)
            elif t == "checkbox":
                self.create_check_box(key, accessible_name=f"{key}_value_checkbox", checked=value['default'])
            elif t == "lineedit":
                self.create_line_edit(key, accessible_name=f"{key}_value_lineedit", default=value['default'])
            elif t == "spinbox":
                self.create_spin_box(key, int(value["min"]), int(value["max"]), int(value["default"]), int(value["step"]), accessible_name=f"{key}_value_spinbox")
            elif t == "doublespinbox":
                self.create_double_spin_box(key, int(value["min"]), int(value["max"]), int(value["step"]), int(value["default"]), accessible_name=f"{key}_value_doublespinbox")


    def get_values(self):

        values = {}

        for widget in self.widget_list:
            if isinstance(widget, QtWidgets.QWidget):
                name = widget.objectName()
                acc_name = widget.accessibleName()
                real_name = None
                if "_value_" in name:

                    real_name = name
                    acc_name = acc_name

                elif "_value_" in acc_name:

                    real_name = acc_name
                    acc_name = name

                if real_name is not None:
                    if "_combobox" in real_name:
                        values[acc_name] = widget.currentText()
                    elif "_lineedit" in real_name:
                        values[acc_name] = widget.text()
                    elif "_checkbox" in real_name:
                        values[acc_name] = widget.isChecked()
                    elif "_spinbox" in real_name or "_doublespinbox" in real_name:
                        values[acc_name] = widget.value()

        return values

class DeforumArgsDataWidget(QDMNodeContentWidget):
    def initUI(self):
        self.createUI()
        self.create_main_layout(grid=3)
        print(self.get_values())


    def createUI(self):
        for key, value in deforum_args_2_layout.items():
            t = value["type"]
            if t == "dropdown":
                self.create_combo_box(value["choices"], f"{key}_value_combobox", accessible_name=key)
            elif t == "checkbox":
                self.create_check_box(key, accessible_name=f"{key}_value_checkbox")
            elif t == "lineedit":
                self.create_line_edit(key, accessible_name=f"{key}_value_lineedit", default=value['default'])
            elif t == "spinbox":
                self.create_spin_box(key, int(value["min"]), int(value["max"]), int(value["default"]), int(value["step"]), accessible_name=f"{key}_value_spinbox")
            elif t == "doublespinbox":
                self.create_double_spin_box(key, int(value["min"]), int(value["max"]), int(value["step"]), int(value["default"]), accessible_name=f"{key}_value_doublespinbox")



    def get_values(self):

        values = {}

        for widget in self.widget_list:
            if isinstance(widget, QtWidgets.QWidget):
                name = widget.objectName()
                acc_name = widget.accessibleName()
                real_name = None
                if "_value_" in name:

                    real_name = name
                    acc_name = acc_name

                elif "_value_" in acc_name:

                    real_name = acc_name
                    acc_name = name

                print(real_name, acc_name)
                if real_name is not None:
                    if "_combobox" in real_name:
                        values[acc_name] = widget.currentText()
                    elif "_lineedit" in real_name:
                        values[acc_name] = widget.text()
                    elif "_checkbox" in real_name:
                        values[acc_name] = widget.isChecked()
                    elif "_spinbox" in real_name or "_doublespinbox" in real_name:
                        values[acc_name] = widget.value()

        return values





@register_node(OP_NODE_DEFORUM_PROMPT)
class DeforumPromptNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_DEFORUM_PROMPT
    op_title = "Deforum Prompt Node"
    content_label_objname = "deforum_prompt_node"
    category = "Data"


    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 1], outputs=[6, 1])

    def initInnerClasses(self):
        self.content = DeforumPromptWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 400
        self.grNode.height = 800
        self.content.setMinimumWidth(400)
        self.content.setMinimumHeight(600)
        self.content.eval_signal.connect(self.evalImplementation)


    def evalImplementation_thread(self, index=0):
        self.busy = True

        input_data = self.getInputData(0)

        prompts = self.content.get_values()

        data = {"animation_prompts":prompts}

        if input_data is not None:
            data = merge_dicts(input_data, data)

        return data


    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.setOutput(0, result)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)



@register_node(OP_NODE_DEFORUM_ARGS_DATA)
class DeforumArgsDataNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_DEFORUM_ARGS_DATA
    op_title = "Deforum Args Node"
    content_label_objname = "deforum_args_node"
    category = "Data"

    custom_input_socket_name = ["DATA", "COND", "SAMPLER", "EXEC"]
    output_socket_name = ["IMAGE", "EXEC"]

    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 1], outputs=[6, 1])

    def initInnerClasses(self):
        self.content = DeforumArgsDataWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 1280
        self.grNode.height = 500
        self.content.setMinimumWidth(1280)
        self.content.setMinimumHeight(400)
        self.content.eval_signal.connect(self.evalImplementation)


    def evalImplementation_thread(self, index=0):
        self.busy = True
        input_data = self.getInputData(0)
        data = self.content.get_values()
        if input_data is not None:
            data = merge_dicts(input_data, data)
        return data


    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        self.setOutput(0, result)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)



@register_node(OP_NODE_DEFORUM_DATA)
class DeforumDataNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_DEFORUM_DATA
    op_title = "Deforum Node"
    content_label_objname = "deforum_node"
    category = "Data"

    custom_input_socket_name = ["DATA", "COND", "SAMPLER", "EXEC"]
    output_socket_name = ["IMAGE", "EXEC"]

    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 3, 5, 1], outputs=[5, 1])

    def initInnerClasses(self):
        self.content = DeforumDataWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 1280
        self.grNode.height = 1340
        self.content.setMinimumWidth(1280)
        self.content.setMinimumHeight(1200)
        self.content.eval_signal.connect(self.evalImplementation)
        deforum_folder_name = "custom_nodes/ainodes_backend_base_nodes/ainodes_backend/deforum"
        sys.path.extend([os.path.join(deforum_folder_name, 'scripts', 'deforum_helpers', 'src')])

    def evalImplementation_thread(self, index=0):
        self.busy = True

        data = self.getInputData(0)
        own_data = self.content.get_values()
        data = merge_dicts(own_data, data)

        #if gs.debug:
        #    print("DATA1", data, "DATA2", data)
        from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.deforum.deforum_helpers.render import \
            render_animation, \
            Root, DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, DeformAnimKeys, DeforumAnimPrompts, ParseqArgs, \
            LoopArgs
        args_dict = DeforumArgs()
        anim_args_dict = DeforumAnimArgs()
        video_args_dict = DeforumOutputArgs()
        parseq_args_dict = ParseqArgs()
        loop_args_dict = LoopArgs()
        controlnet_args = None
        root_dict = Root()

        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        video_args = SimpleNamespace(**video_args_dict)
        parseq_args = SimpleNamespace(**parseq_args_dict)
        loop_args = SimpleNamespace(**loop_args_dict)
        root = SimpleNamespace(**root_dict)




        #for key, value in data.items():
        #    print(key, value)

        def keyframeExamples():
            return '''{
            "0": "Red sphere",
            "max_f/4-5": "Cyberpunk city",
            "max_f/2-10": "Cyberpunk robot",
            "3*max_f/4-15": "Portrait of a cyberpunk soldier",
            "max_f-20": "Portrait of a cyberpunk robot soldier"
        }'''

        args_dict['animation_prompts'] = keyframeExamples()

        root.animation_prompts = json.loads(args_dict['animation_prompts'])

        for key, value in args.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]

                setattr(args, key, val)

        for key, value in anim_args.__dict__.items():
            if key in data:
                if data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = data[key]
                setattr(anim_args, key, val)

        for key, value in root.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]

                setattr(root, key, val)

        animation_prompts = root.animation_prompts

        args.timestring = time.strftime('%Y%m%d%H%M%S')
        current_arg_list = [args, anim_args, video_args, parseq_args]
        full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
        root.raw_batch_name = args.batch_name
        args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
        args.outdir = os.path.join(full_base_folder_path, str(args.batch_name))
        test = render_animation(self, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root, callback=self.handle_callback)

        return True
        #print(test)

        new_data = {}

        if data is not None:
            data_name = self.content.data_name.text()
            data_value = self.content.data_equal.text()

    def handle_callback(self, image):
        pixmap = pil_image_to_pixmap(image)
        for node in self.getOutputs(0):
            if isinstance(node, ImagePreviewNode):
                node.content.preview_signal.emit(pixmap)
            elif isinstance(node, VideoOutputNode):
                frame = np.array(image)
                node.content.video.add_frame(frame, dump=node.content.dump_at.value())





    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)
        return
        self.markDirty(False)
        self.setOutput(0, result)
        pass
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)

    def onInputChanged(self, socket=None):
        pass


def merge_dicts(dict1, dict2):
    result_dict = dict1.copy()
    for key, value in dict2.items():
        if key in result_dict:
            result_dict[key] = value
        else:
            result_dict[key] = value
    return result_dict


