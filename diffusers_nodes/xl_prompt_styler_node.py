import secrets
import subprocess

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor, torch_gc
from ai_nodes.ainodes_engine_base_nodes.diffusers_nodes.diffusers_helpers import scheduler_type_values, SchedulerType, \
    get_scheduler
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

import json
import os

def read_json_file(file_path):
    try:
        # Open file, load JSON content into python dictionary, and return it.
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            return json_data
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def read_sdxl_styles(json_data):
    # Check that data is a list
    if not isinstance(json_data, list):
        print("Error: input data must be a list")
        return None

    names = []

    # Iterate over each item in the data list
    for item in json_data:
        # Check that the item is a dictionary
        if isinstance(item, dict):
            # Check that 'name' is a key in the dictionary
            if 'name' in item:
                # Append the value of 'name' to the names list
                names.append(item['name'])

    return names


def read_sdxl_templates_replace_and_combine(json_data, template_name, positive_prompt, negative_prompt):
    try:
        # Check if json_data is a list
        if not isinstance(json_data, list):
            raise ValueError("Invalid JSON data. Expected a list of templates.")

        for template in json_data:
            # Check if template contains 'name' and 'prompt' fields
            if 'name' not in template or 'prompt' not in template:
                raise ValueError("Invalid template. Missing 'name' or 'prompt' field.")

            # Replace {prompt} in the matching template
            if template['name'] == template_name:
                positive_prompt = template['prompt'].replace('{prompt}', positive_prompt)

                json_negative_prompt = template.get('negative_prompt', "")
                if negative_prompt:
                    negative_prompt = f"{json_negative_prompt}, {negative_prompt}" if json_negative_prompt else negative_prompt
                else:
                    negative_prompt = json_negative_prompt

                return positive_prompt, negative_prompt

        # If function hasn't returned yet, no matching template was found
        raise ValueError(f"No template found with name '{template_name}'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


OP_NODE_XL_PROMPT_STYLER = get_next_opcode()



class XLPromptWidget(QDMNodeContentWidget):
    def initUI(self):
        self.prompt = self.create_text_edit("Prompt", placeholder="Prompt")
        self.n_prompt = self.create_text_edit("Negative Prompt", placeholder="Negative Prompt")
        file_path = os.path.join("config", 'xl_styles.json')
        # Read JSON from file
        self.json_data = read_json_file(file_path)
        # Retrieve styles from JSON data
        styles = read_sdxl_styles(self.json_data)


        self.styles = self.create_combo_box(styles, "Styles")
        self.create_main_layout(grid=1)


@register_node(OP_NODE_XL_PROMPT_STYLER)
class XLPromptNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "XL"
    op_code = OP_NODE_XL_PROMPT_STYLER
    op_title = "Diffusers XL - Style"
    content_label_objname = "sd_xlstyle_node"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = XLPromptWidget
    dim = (340, 800)
    output_data_ports = [0]
    exec_port = 1

    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 1], outputs=[6, 1])

        file_path = os.path.join("config", 'xl_styles.json')
        # Read JSON from file
        self.json_data = read_json_file(file_path)
        # Retrieve styles from JSON data
        self.styles = read_sdxl_styles(self.json_data)



    def evalImplementation_thread(self, index=0):
        prompt = self.content.prompt.toPlainText()
        negative_prompt = self.content.n_prompt.toPlainText()
        negative_prompt_2 = negative_prompt

        data = self.getInputData(1)
        if data is not None:
            if "prompt" in data:
                prompt = data["prompt"]
            if "prompt_2" in data:
                prompt_2 = data["prompt_2"]
            if "negative_prompt" in data:
                negative_prompt = data["negative_prompt"]
            if "negative_prompt_2" in data:
                negative_prompt_2 = data["negative_prompt_2"]
        positive_prompt, negative_prompt = read_sdxl_templates_replace_and_combine(self.json_data, self.content.styles.currentText(), prompt, negative_prompt)


        if data is not None:
            data["prompt"] = positive_prompt
            data["prompt_2"] = prompt
            data["negative_prompt"] = negative_prompt
            data["negative_prompt_2"] = negative_prompt_2
        else:
            data = {
                "prompt_2":prompt,
                "prompt":positive_prompt,
                "negative_prompt_2":negative_prompt,
                "negative_prompt":negative_prompt_2
            }

        return [data]

    def remove(self):
        super().remove()
