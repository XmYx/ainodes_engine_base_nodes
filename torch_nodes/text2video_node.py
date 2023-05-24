import copy
import os
import random
import secrets
import shlex
import subprocess
import threading
import time

import torch
from qtpy import QtCore
from qtpy import QtWidgets
from PIL import Image
import cv2


from ..ainodes_backend.model_loader import ModelLoader
from ..ainodes_backend import torch_gc, pil_image_to_pixmap

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs
from ..ainodes_backend.t2v_pipeline import TextToVideoSynthesis
OP_NODE_T2V = get_next_opcode()
class Text2VideoWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_widgets()
        self.create_main_layout()


    def create_widgets(self):
        self.prompt = self.create_text_edit("PROMPT")
        self.n_prompt = self.create_text_edit("N_PROMPT")
        self.seed = self.create_line_edit("SEED")
        self.steps = self.create_spin_box("STEPS", 1, 1000, 15, 1)
        self.frames = self.create_spin_box("FRAMES", 1, 1000, 15, 1)
        self.scale = self.create_double_spin_box("GUIDANCE", 0.01, 100.0, 0.01, 7.5)
        self.eta = self.create_double_spin_box("ETA", 0.00, 100.0, 0.01, 0.0)
        self.strength = self.create_double_spin_box("STRENGTH", 0.00, 100.0, 0.01, 0.0)
        self.width_value = self.create_spin_box("Width", 64, 4096, 320, 64)
        self.height_value = self.create_spin_box("Height", 64, 4096, 320, 64)
        self.cpu_vae = self.create_check_box("CPU VAE")
        self.random_prompt = self.create_check_box("RANDOM PROMPT")
        self.continue_sampling = self.create_check_box("CONTINUE")
        self.edit_prompt_button = QtWidgets.QPushButton("Edit Prompt")
        self.create_button_layout([self.edit_prompt_button])
        self.edit_prompt_button.clicked.connect(self.prompt_editor)
        self.prompts = []


    def prompt_editor(self):
        prompts = EditPromptsDialog(self.prompts)
        result = prompts.exec()
        if result == QtWidgets.QDialog.Accepted:
            self.prompts = prompts.prompts
            # Save the modified prompt list to a file or do something else with it
            print("Modified prompts:", self.prompts)
class CenterExpandingSizePolicy(QtWidgets.QSizePolicy):
    def __init__(self, parent=None):
        super().__init__(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.parent = parent
        self.setHorizontalStretch(0)
        self.setVerticalStretch(0)
        self.setRetainSizeWhenHidden(True)
        self.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)


@register_node(OP_NODE_T2V)
class Text2VideoNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_T2V
    op_title = "Text2Video Node"
    content_label_objname = "t2v_loader_node"
    category = "Sampling"
    input_socket_name = ["EXEC"]
    output_socket_name = ["EXEC"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[5,1])
        self.loader = ModelLoader()
        self.last_latent = None
        self.content.eval_signal.connect(self.evalImplementation)


    def initInnerClasses(self):
        self.content = Text2VideoWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 340
        self.grNode.height = 600
        self.content.setMinimumHeight(400)
        self.content.setMinimumWidth(340)
        #pass
        self.iterating = False
        self.index = 0
        self.prompts = [
            "A moonlit night with a dark purple sky.",
            "A forest filled with neon, glowing mushrooms.",
            "A mysterious portal leading to a hidden dimension.",
            "Walls of the floating castle pulsate with vibrant colors.",
            "A room inside the castle full of swirling, hypnotic patterns.",
            "Crystal gardens that refract light into mesmerizing prisms.",
            "A waterfall cascading down from the castle, morphing into different shapes.",
            "Shadowy figures dancing in a trance-like state.",
            "A river of liquid light flowing through the landscape.",
            "The castle walls adorned with intricate, moving murals.",
            "A hall of mirrors distorting reality and reflections.",
            "A glowing labyrinth leading to the heart of the castle.",
            "A room filled with floating orbs that emit otherworldly music.",
            "A spiral staircase that changes direction and shape as it is climbed.",
            "A bridge made of shifting, iridescent energy.",
            "A library filled with ancient, enchanted books.",
            "A garden of plants that change color and shape with every touch.",
            "A room with walls made of undulating waves of color.",
            "An observatory where celestial bodies form mesmerizing patterns.",
            "A banquet hall filled with surreal, transformative feasts.",
            "A surreal landscape of melting mountains and flowing skies.",
            "A throne room where shadows come to life and bow to their ruler.",
            "A dark, foggy forest with trees that whisper secrets.",
            "A hidden cavern filled with pulsating crystal formations.",
            "A room that transports visitors to alternate dimensions.",
            "A balcony with a view of a horizon that shifts and transforms.",
            "A courtyard where shadows become solid and can be sculpted.",
            "A room of portals that connect to different worlds.",
            "An underground chamber with walls of shifting, living darkness.",
            "A sky filled with swirling, hypnotic clouds.",
            "A marketplace where dreams and nightmares can be bought and sold.",
            "A room that reflects the deepest desires and fears of its occupants.",
            "A dark, enchanting forest that beckons travelers to wander.",
            "A room filled with holographic projections of past and future events.",
            "A gallery of paintings that come to life and tell stories.",
            "A realm where the natural laws of reality are constantly in flux.",
            "A room where echoes of lost souls can be heard.",
            "A garden where plants sing haunting melodies.",
            "A chamber that can manipulate the flow of time.",
            "A room where dreams and memories manifest as tangible objects.",
            "A chasm filled with dark, shimmering water.",
            "A room where the walls are made of ever-changing fractal patterns.",
            "A dance floor where participants merge and separate like liquid.",
            "A fountain that flows with glowing, transformative liquid.",
            "A room where shadows play out scenes from forgotten myths.",
            "A cosmic garden filled with plants from other planets.",
            "A hall filled with statues that come to life.",
            "A room that can control the elements of nature.",
            "A floating island with a dark, enchanted forest.",
            "A room where visitors can enter the minds of others.",
            "A cavern with walls that tell stories of ancient civilizations.",
            "A room where the floor moves and shifts like a liquid.",
            "A magical workshop filled with enchanted tools and materials.",
            "A room that contains the essence of every emotion.",
            "A celestial observatory with views into other galaxies.",
            "A chamber that can alter the appearance of its occupants.",
            "A room where the walls are made of living, breathing plants."]
    def evalImplementation_thread(self, index=0):
        if "t2v" not in gs.models:
            #gs.models["t2v"] = TextToVideoSynthesis(model_dir="models/t2v")
            from diffusers import DPMSolverMultistepScheduler
            import torch
            from diffusers import TextToVideoSDPipeline
            gs.models["t2v"] = TextToVideoSDPipeline.from_pretrained("strangeman3107/animov-0.1.1", torch_dtype=torch.float16,
                                                     variant="fp16", local_files_only=False)
            gs.models["t2v"].scheduler = DPMSolverMultistepScheduler.from_config(gs.models["t2v"].scheduler.config)
            gs.models["t2v"].enable_model_cpu_offload()
            gs.models["t2v"].enable_vae_slicing()
            gs.models["t2v"].enable_xformers_memory_efficient_attention()

        return_pixmaps = []
        try:
            prompt = self.content.prompt.toPlainText()
            prompt = self.get_next_prompt() if self.content.random_prompt.isChecked() else prompt
            n_prompt = self.content.n_prompt.toPlainText()
            seed = self.content.seed.text()
            frames = self.content.frames.value()
            scale = self.content.scale.value()
            width = self.content.width_value.value()
            height = self.content.height_value.value()
            eta = self.content.eta.value()
            strength = self.content.strength.value()
            cpu_vae = self.content.cpu_vae.isChecked()
            strength = strength if strength != 0 else None
            try:
                seed = int(seed)
            except:
                choice = random.choice([-1, 1])
                seed = secrets.randbelow(9999999999999999)
                seed = choice * seed
            steps = self.content.steps.value()
            import torch._dynamo
            #if "t2v_pipeline" not in gs.models:

            torch.manual_seed(seed)
            fancy_readout(prompt, steps, frames, scale, width, height, seed)

            style = "diffusers"
            if style == "classic":
                if self.last_latent is not None and self.content.continue_sampling.isChecked():
                    latents = self.last_latent
                else:
                    latents = None
                generator = torch.Generator(device="cuda").manual_seed(seed)
                return_samples, latent = gs.models["t2v"](prompt, n_prompt, steps, frames, scale, width=width, height=height, eta=eta, cpu_vae=cpu_vae, latents=latents, strength=strength, generator=generator)
                self.last_latent = latent
                return_pixmaps = []
            elif style == "diffusers":
                return_samples = gs.models["t2v"](
                    prompt = prompt,
                    height = height,
                    width = width,
                    num_frames = frames,
                    num_inference_steps = steps,
                    guidance_scale = scale,
                    negative_prompt = n_prompt,
                    eta = eta,
                    output_type = "np"

                ).frames

            for frame in return_samples:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = Image.fromarray(copy.deepcopy(frame))
                pixmap = pil_image_to_pixmap(image)
                return_pixmaps.append(pixmap)

            #if len(self.getOutputs(1)) > 0:
            #    self.iterate_frames(return_samples)
            #    while self.iterating == True:
            #        time.sleep(0.15)


        except Exception as e:
            print(e)
            try:
                gs.models["t2v"].cleanup()
                gs.models["t2v"] = None
                del gs.models["t2v"]
            except:
                pass
        finally:
            return return_pixmaps
    @QtCore.Slot(object)
    def onWorkerFinished(self, result):
        super().onWorkerFinished(None)

        self.setOutput(0, result)
        self.markDirty(True)
        self.markInvalid(False)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        #pass

    def eval(self, index=0):
        self.markDirty(True)
        self.content.eval_signal.emit()
    def onInputChanged(self, socket=None):
        pass

    def iterate_frames(self, frames):
        self.iterating = True
        for frame in frames:
            node = None
            if len(self.getOutputs(1)) > 0:
                node = self.getOutputs(1)[0]
            if node is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                image = Image.fromarray(copy.deepcopy(frame))
                pixmap = pil_image_to_pixmap(image)
                self.setOutput(0, pixmap)
                node.eval()
                time.sleep(0.1)
        self.iterating = False
    def get_next_prompt(self):
        self.prompts = self.content.prompts
        self.index = (self.index + 1) % len(self.prompts)
        prompt = self.prompts[self.index]
        return prompt
def fancy_readout(prompt, steps, frames, scale, width, height, seed):
    # Define the box-drawing characters
    top_left = "┏"
    top_right = "┓"
    bottom_left = "┗"
    bottom_right = "┛"
    horizontal = "━"
    vertical = "┃"
    space = " "

    # Define the readout message
    message = [
        "Running inference with the following parameters:",
        "",
        f"  Prompt: {prompt}",
        f"  Steps: {steps}",
        f"  Frames: {frames}",
        f"  Scale: {scale}",
        f"  Seed: {seed}",
        "",
        "Status:",
        f"  Width: {width}",
        f"  Height: {height}",
        "",
        "  CPU VAE: ",
        "",

    ]

    # Calculate the maximum line width
    max_width = max(len(line) for line in message)

    # Print the framed readout
    print(top_left + horizontal * (max_width + 2) + top_right)
    print(vertical + space * (max_width + 2) + vertical)
    print(vertical + f"{'TEXT2VIDEO:':^{max_width}}{vertical}")
    print(vertical + space * (max_width + 2) + vertical)
    for line in message:
        print(vertical + f"  {line:<{max_width}}  {vertical}")
    print(vertical + space * (max_width + 2) + vertical)
    print(bottom_left + horizontal * (max_width + 2) + bottom_right)

def generate_video_prompt_2():
    activities = [
        "running", "dancing", "swimming", "climbing", "painting", "cooking", "writing", "meditating", "gardening",
        "skating", "hiking", "fishing", "yoga", "cycling", "photography", "singing", "acting", "drawing",
    ]

    colors = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "violet",
    ]

    subjects = [
        "love", "friendship", "betrayal", "tragedy", "redemption", "survival", "discovery", "escape", "journey",
        "revenge", "magic", "heroes", "villains", "aliens", "monsters", "superpowers", "ghosts", "treasure", "time travel",
    ]

    movements = [
        "slow motion", "fast forward", "reverse", "jump cut", "crossfade", "fade in", "fade out", "zoom in", "zoom out",
    ]

    categories = [activities, colors, subjects, movements]
    prompt_elements = []

    for category in categories:
        element_count = random.randint(1, 3)
        prompt_elements.extend(random.sample(category, element_count))

    random.shuffle(prompt_elements)
    prompt = ' '.join(prompt_elements[:77])

    return prompt

def generate_video_prompt():

    human_subjects = [
        "chef", "athlete", "scientist", "teacher", "doctor", "engineer", "painter", "dancer", "musician", "astronaut",
        "detective", "actor", "writer", "farmer", "nurse", "pilot", "construction worker", "firefighter",
        "police officer",
        "magician", "salesperson", "librarian", "lifeguard", "student", "plumber", "electrician", "architect",
        "photographer",
        "bus driver", "bartender", "hairdresser", "gardener", "chef", "journalist", "coach", "mechanic", "programmer",
        "veterinarian", "banker", "designer", "comedian", "judge", "lawyer", "dentist", "waiter", "receptionist"
    ]

    inorganic_subjects = [
        "robot", "drone", "airplane", "spaceship", "car", "bicycle", "boat", "train", "bus", "truck",
        "smartphone", "computer", "camera", "television", "clock", "washing machine", "dishwasher", "refrigerator",
        "oven",
        "fan", "air conditioner", "calculator", "printer", "scanner", "vacuum cleaner", "blender", "microwave", "lamp",
        "sewing machine", "keyboard", "mouse", "laptop", "headphones", "speaker", "guitar", "piano", "violin", "drums"
    ]

    organic_subjects = [
        "giraffe", "goldendoodle", "panda bear", "teddy bear", "dog", "monkey", "litter of puppies",
        "elephant", "kangaroo", "tiger", "lion", "orca", "octopus", "dolphin", "eagle", "parrot", "flamingo", "peacock",
        "penguin", "koala", "sloth", "rhinoceros", "hippopotamus", "iguana", "chameleon", "crocodile", "platypus",
        "anteater",
        "otter", "seal", "walrus", "hamster", "rabbit", "squirrel", "fox", "owl", "gorilla", "gazelle", "zebra", "deer",
        "buffalo", "wildebeest", "butterfly", "dragonfly", "bee", "ant", "spider", "snake", "lizard", "frog", "toad",
        "tortoise", "turtle", "shark", "whale", "jellyfish", "starfish", "coral", "fish"
    ]

    subjects = random.choice([human_subjects, inorganic_subjects, organic_subjects])

    actions = [
        "underneath a microwave", "playing in a park by a lake", "driving a car", "running in New York City",
        "flythrough of a fast food restaurant",
        "wearing a Superhero outfit with red cape flying through the sky", "learning to play the piano",
        "running through the yard", "dancing in times square",
        "reading a book", "riding a bicycle", "eating ice cream", "swimming in a pool", "jumping on a trampoline",
        "baking a cake",
        "painting a picture", "performing a magic trick", "playing soccer", "building a sandcastle",
        "skiing down a mountain",
        "climbing a tree", "playing a video game", "flying a kite", "singing a song", "writing a letter",
        "watching a movie",
        "gardening", "doing yoga", "riding a roller coaster", "taking a selfie", "making a snowman",
        "playing hide and seek",
        "skateboarding", "surfing", "making a pizza", "solving a puzzle", "taking a nap", "fishing",
        "playing an instrument",
        "practicing karate", "playing basketball", "giving a speech", "attending a party", "going shopping",
        "riding a scooter",
        "juggling", "cooking a meal", "making pottery", "drawing a comic", "doing laundry", "exploring a cave",
        "going for a hike",
        "playing with a toy", "taking a bath", "riding a unicycle", "playing chess", "making a sculpture",
        "dressing up",
        "having a picnic", "playing tennis", "playing catch", "rock climbing", "gymnastics", "building a robot",
        "knitting",
        "doing a cartwheel", "listening to music", "meditating", "photographing nature", "playing ping pong",
        "dancing ballet",
        "making a collage", "performing in a play", "writing poetry", "playing cards", "traveling in a spaceship",
        "riding a skateboard",
        "creating a website", "making a video", "planting a tree", "taking care of a pet", "playing volleyball",
        "ice skating",
        "playing the stock market", "attending a concert", "riding a horse", "walking on a tightrope",
        "blowing bubbles", "making a campfire"
    ]

    environments = [
        "on a regular day",
        "on a rainy day",
        "on a snowy day",
        "on a dystopian alien planet",
        "in a futuristic city",
        "in a haunted house",
        "in a magical forest",
        "in an underwater world",
        "on a desert island"
    ]

    subject = random.choice(subjects)
    action = random.choice(actions)
    environment = random.choice(environments)

    prompt = f"{subject} {action} {environment}."
    return prompt
def get_video_prompt():
    prompt = "Generate a prompt for a short video that includes activities, colors, subjects, and movements."

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=77,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()


class EditPromptsDialog(QtWidgets.QDialog):
    def __init__(self, prompts):
        super().__init__()
        self.prompts = prompts.copy()
        self.edited_prompts = []
        self.list_widget = QtWidgets.QListWidget()
        for prompt in self.prompts:
            self.list_widget.addItem(prompt)
        add_button = QtWidgets.QPushButton("Add")
        add_button.clicked.connect(self.add_prompt)
        edit_button = QtWidgets.QPushButton("Edit")
        edit_button.clicked.connect(self.edit_prompt)
        delete_button = QtWidgets.QPushButton("Delete")
        delete_button.clicked.connect(self.delete_prompt)
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(add_button)
        button_layout.addWidget(edit_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(ok_button)
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.list_widget)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def add_prompt(self):
        dialog = LargeTextEditDialog("Add Prompt")
        result = dialog.exec()

        if result == QtWidgets.QDialog.Accepted:
            prompt = dialog.get_text()
            if prompt not in self.prompts and prompt != "":
                self.prompts.append(prompt)
                self.edited_prompts.append(prompt)
                self.list_widget.addItem(prompt)

    def edit_prompt(self):
        selected_items = self.list_widget.selectedItems()
        if len(selected_items) == 1:
            old_prompt = selected_items[0].text()
            dialog = LargeTextEditDialog("Edit Prompt", old_prompt)
            result = dialog.exec()

            if result == QtWidgets.QDialog.Accepted:
                new_prompt = dialog.get_text()
                if new_prompt != "" and new_prompt != old_prompt:
                    self.prompts[self.prompts.index(old_prompt)] = new_prompt
                    self.edited_prompts.append(new_prompt)
                    selected_items[0].setText(new_prompt)

    def delete_prompt(self):
        selected_items = self.list_widget.selectedItems()
        if len(selected_items) > 0:
            for item in selected_items:
                self.prompts.remove(item.text())
                self.list_widget.takeItem(self.list_widget.row(item))
                self.edited_prompts.append(item.text())


class LargeTextEditDialog(QtWidgets.QDialog):
    def __init__(self, title, text="", parent=None):
        super(LargeTextEditDialog, self).__init__(parent)

        self.setWindowTitle(title)

        self.text_edit = QtWidgets.QPlainTextEdit(self)
        self.text_edit.setPlainText(text)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.button_box)

    def get_text(self):
        return self.text_edit.toPlainText()