import os.path
from copy import deepcopy

import gdown
import numpy as np
from PIL import Image
from einops import rearrange
from huggingface_hub import hf_hub_url, cached_download
from omegaconf import DictConfig

from .ksampler_node import get_fixed_seed
from ..ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, pil2tensor, tensor2pil

import torch
from qtpy import QtWidgets, QtCore, QtGui

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget


from kandinsky2 import CONFIG_2_1, Kandinsky2_1

from ..image_nodes.image_preview_node import ImagePreviewNode
from ..video_nodes.video_save_node import VideoOutputNode

OP_NODE_KANDINSKY = get_next_opcode()

SCHEDULERS = ["karras", "normal", "simple", "ddim_uniform"]
SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
            "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"]

class KandinskyWidget(QDMNodeContentWidget):
    seed_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)
    text_signal = QtCore.Signal(str)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        try:
            import flash_attn
            self.flash_attn = self.create_check_box("Use Flash Attn")
            self.flash_attn_avail = True
        except:
            self.flash_attn_avail = False
        self.use_finetune = self.create_check_box("Use Finetune")
        self.task = self.create_combo_box(["TXT2IMG", "INPAINT"], "Task")
        self.tensor_preview = self.create_check_box("Tensor Preview")
        self.prompt = self.create_text_edit("Prompt:", placeholder="Prompt")
        self.negative_prompt = self.create_text_edit("Negative Prompt:", placeholder="Negative Prompt")
        self.negative_prior_prompt = self.create_text_edit("Negative Prior Prompt:", placeholder="Negative Prior Prompt")
        self.seed = self.create_line_edit("Seed:", placeholder="Leave empty for random seed")
        self.steps = self.create_spin_box("Steps:", 1, 10000, 25)
        self.cfg_scale = self.create_spin_box("Guidance Scale:", 0, 1000, 4)
        self.w_param = self.create_spin_box("Width:", 64, 2048, 512, 64)
        self.h_param = self.create_spin_box("Height:", 64, 2048, 512, 64)
        self.strength = self.create_double_spin_box("Strength:", 0.00, 1.00, 0.01, 0.84)
        self.sampler = self.create_combo_box(["p_sampler", "ddim_sampler", "plms_sampler"], "Sampler:")
        self.prior_cf_scale = self.create_spin_box("Prior Scale:", 0, 1000, 4)
        self.prior_steps = self.create_spin_box("Prior Steps:", 0, 1000, 5)
        self.force_values = self.create_check_box("Force Values:", False)
        self.button = QtWidgets.QPushButton("Run")


@register_node(OP_NODE_KANDINSKY)
class KandinskyNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/kandinsky.png"
    op_code = OP_NODE_KANDINSKY
    op_title = "Kandinsky"
    content_label_objname = "kandinsky_node"
    category = "aiNodes Base/Sampling"
    def __init__(self, scene, inputs=[], outputs=[]):
        super().__init__(scene, inputs=[5,5,6,1], outputs=[5,1])
        self.content.button.clicked.connect(self.evalImplementation)

        # Create a worker object
    def initInnerClasses(self):
        self.content = KandinskyWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.grNode.height = 750
        self.grNode.width = 256
        self.content.setMinimumWidth(256)
        self.content.setMinimumHeight(256)
        self.seed = ""
        self.content.seed_signal.connect(self.setSeed)
        self.content.progress_signal.connect(self.setProgress)
        self.progress_value = 0
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.text_signal.connect(self.set_prompt)
        self.busy = False
        self.task = None
        self.latent_rgb_factors = torch.tensor([
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=torch.float, device='cpu')

    ##@QtCore.Slot(str)
    def set_prompt(self, text):
        self.content.prompt.setText(text)


    def evalImplementation_thread(self, prompt_override=None, args=None, init_image=None):

        task = self.content.task.currentText()
        if task == "TXT2IMG":
            task_type = 'text2img'
        else:
            task_type = 'inpainting'

        if f"kandinsky" not in gs.models or gs.loaded_kandinsky != task_type:
            use_finetune = self.content.use_finetune.isChecked()
            flash = False if not self.content.flash_attn_avail else self.content.flash_attn.isChecked()
            gs.models["kandinsky"] = get_kandinsky2_1('cuda', task_type=task_type, use_flash_attention=flash, use_finetune=use_finetune)
            gs.loaded_kandinsky = task_type


        masks = self.getInputData(0)
        images = self.getInputData(1)
        data = self.getInputData(2)

        prompt = self.content.prompt.toPlainText()
        n_prompt = self.content.negative_prompt.toPlainText()
        n_p_prompt = self.content.negative_prior_prompt.toPlainText()
        if data:
            if "prompt" in data:
                prompt = data["prompt"]
                self.content.text_signal.emit(prompt)
            else:
                prompt = self.content.prompt.toPlainText()
        num_steps = self.content.steps.value()
        guidance_scale = self.content.cfg_scale.value()
        h = self.content.h_param.value()
        w = self.content.w_param.value()
        sampler = self.content.sampler.currentText()
        prior_cf_scale = self.content.prior_cf_scale.value()
        prior_steps = self.content.prior_steps.value()
        self.seed = self.content.seed.text()
        try:
            self.seed = int(self.seed)
        except:
            self.seed = get_fixed_seed('')
        return_images = []
        return_pil_images = []
        strength = self.content.strength.value()
        if prompt_override is not None:
            images = init_image
            h = args.H
            w = args.W
            if not self.content.force_values.isChecked():
                num_steps = args.steps
                prompt = prompt_override
                strength = args.strength
                guidance_scale = int(args.scale)
                print(prompt, strength, guidance_scale, num_steps)
                self.seed = args.seed
        torch.manual_seed(self.seed)
        print(f"KANDINSKY NODE: seed:{self.seed}")
        gs.models["kandinsky"].clip_model.to("cuda")
        gs.models["kandinsky"].image_encoder.to("cuda")

        if images is not None:
            for image in images:

                if task_type == "text2img":

                    pil_img = tensor2pil(image)
                    return_pil_images = gs.models["kandinsky"].generate_img2img(
                        prompt,
                        pil_img,
                        strength=strength,
                        num_steps=num_steps,
                        batch_size=1,
                        guidance_scale=guidance_scale,
                        h=pil_img.size[1],
                        w=pil_img.size[0],
                        sampler=sampler,
                        prior_cf_scale=prior_cf_scale,
                        prior_steps=str(prior_steps),
                        callback=self.callback
                    )
                else:
                    pil_img = tensor2pil(image)
                    img_mask = pixmap_to_tensor(masks[0]).convert("L")

                    # Get the original dimensions
                    #original_height, original_width = img_mask.size

                    # Calculate the new dimensions
                    #new_width = int(original_width // 8)
                    #new_height = int(original_height // 8)

                    # Resize the image
                    resized_img_mask = img_mask.resize(pil_img.size)

                    return_pil_images = gs.models["kandinsky"].generate_inpainting(prompt,
                                                                                    pil_img,
                                                                                    np.array(resized_img_mask),
                                                                                    num_steps=num_steps,
                                                                                    batch_size=1,
                                                                                    guidance_scale=guidance_scale,
                                                                                    h=pil_img.size[1],
                                                                                    w=pil_img.size[0],
                                                                                    sampler="ddim_sampler",
                                                                                    prior_cf_scale=prior_cf_scale,
                                                                                    prior_steps=str(prior_steps),
                                                                                    negative_prior_prompt="",
                                                                                    negative_decoder_prompt="",
                    )
        else:
            return_pil_images = gs.models["kandinsky"].generate_text2img(
                prompt,
                negative_prior_prompt=n_p_prompt,
                negative_decoder_prompt=n_prompt,
                num_steps=num_steps,
                batch_size=1,
                guidance_scale=guidance_scale,
                h=h, w=w,
                sampler=sampler,
                prior_cf_scale=prior_cf_scale,
                prior_steps=str(prior_steps),
                callback=self.callback

            )
        for image in return_pil_images:
            tensor = pil2tensor(image)
            return_images.append(tensor)
        return return_images
    def callback(self, tensors):
        i = tensors["i"]

        if self.content.tensor_preview.isChecked():
            if i < self.content.steps.value() - 2:
                latent = tensors["denoised"][0].detach().to("cpu")
                latent = torch.einsum('...lhw,lr -> ...rhw', latent, self.latent_rgb_factors)
                latent = (((latent + 1) / 2)
                          .clamp(0, 1)  # change scale from -1..1 to 0..1
                          .mul(0xFF)  # to 0..255
                          .byte())
                # Copying to cpu as numpy array
                latent = rearrange(latent, 'c h w -> h w c').detach().cpu().numpy()
                img = Image.fromarray(latent)
                img = img.resize((img.size[0] * 8, img.size[1] * 8), resample=Image.LANCZOS)
                latent_pixmap = tensor_image_to_pixmap(img)
                if len(self.getOutputs(0)) > 0:
                    nodes = self.getOutputs(0)
                    for node in nodes:
                        if isinstance(node, ImagePreviewNode):
                            node.content.preview_signal.emit(latent_pixmap)
                        if isinstance(node, VideoOutputNode):
                            frame = np.array(img)
                            node.content.video.add_frame(frame, dump=node.content.dump_at.value())


    ##@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        self.busy = False
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        self.progress_value = 0
        self.executeChild(output_index=1)

    ##@QtCore.Slot()
    def setSeed(self):
        self.content.seed.setText(str(self.seed))

    ##@QtCore.Slot()
    def setProgress(self, progress=None):
        if progress != 100 and progress != 0:
            self.progress_value = self.progress_value + self.single_step
        self.content.progress_bar.setValue(self.progress_value)

    def onInputChanged(self, socket=None):
        pass

def get_kandinsky2_1(
    device,
    task_type="text2img",
    cache_dir="models/kandinsky2",
    use_auth_token=None,
    use_flash_attention=False,
    use_finetune=False,
):
    cache_dir = os.path.join(cache_dir, "2_1") if not use_finetune else os.path.join(cache_dir, "2_1_finetune")

    os.makedirs(cache_dir, exist_ok=True)

    config = DictConfig(deepcopy(CONFIG_2_1))
    config["model_config"]["use_flash_attention"] = use_flash_attention
    if task_type == "text2img":
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name)
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name)

    if not use_finetune:
        cached_download(
            config_file_url,
            cache_dir=cache_dir,
            force_filename=model_name,
            use_auth_token=use_auth_token,
        )
    else:
        assert task_type == "text2img", "There is only normal finetune available yet, make sure to set Kandinsky to Text2Image mode."
        if not os.path.isfile(os.path.join(cache_dir, model_name)):
            done = download_pretrained_models({model_name:"1w3q5C0uzQSdGRwNSnokfIiLBlT8liTjy"}, cache_dir)


    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=prior_name)

    if not use_finetune:
        cached_download(
            config_file_url,
            cache_dir=cache_dir,
            force_filename=prior_name,
            use_auth_token=use_auth_token,
        )
    else:
        assert task_type == "text2img", "There is only normal finetune available yet, make sure to set Kandinsky to Text2Image mode."
        if not os.path.isfile(os.path.join(cache_dir, prior_name)):
            done = download_pretrained_models({prior_name:"1dp90YLwYXXQxZX2EWrujqSAwWk2xdL3V"}, cache_dir)

    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=f"text_encoder/{name}")
        cached_download(
            config_file_url,
            cache_dir=cache_dir_text_en,
            force_filename=name,
            use_auth_token=use_auth_token,
        )

    config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename="movq_final.ckpt")
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="movq_final.ckpt",
        use_auth_token=use_auth_token,
    )

    config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename="ViT-L-14_stats.th")
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="ViT-L-14_stats.th",
        use_auth_token=use_auth_token,
    )

    config["tokenizer_name"] = cache_dir_text_en
    config["text_enc_params"]["model_path"] = cache_dir_text_en
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")
    cache_model_name = os.path.join(cache_dir, model_name)
    cache_prior_name = os.path.join(cache_dir, prior_name)
    model = Kandinsky2_1(config, cache_model_name, cache_prior_name, device, task_type=task_type)
    return model


def download_pretrained_models(file_ids, save_path_root):
    import os.path as osp
    import gdown

    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        file_url = 'https://drive.google.com/uc?id=' + file_id
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            user_response = input(f'{file_name} already exist. Do you want to cover it? Y/N\n')
            if user_response.lower() == 'y':
                print(f'Covering {file_name} to {save_path}')
                gdown.download(file_url, save_path, quiet=False)
                # download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Wrong input. Only accepts Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            gdown.download(file_url, save_path, quiet=False)
            # download_file_from_google_drive(file_id, save_path)
    return True