import inspect
import os
from typing import Callable, List, Optional, Union

import numpy as np
import torch

import PIL
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor

# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker, StableDiffusionPipelineOutput
# from resize_right import resize_right
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

# from aitemplate.compiler import Model

from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    logging,
)
from diffusers import DiffusionPipeline
# from pipelines.prompt_crossattention.crossattention import LocalBlend, AttentionReplace
# from pipelines.prompt_crossattention.ptp_utils import register_attention_control
# from pipelines.tensor import interp_methods


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class StableDiffusionInstructPix2PixPipeline(DiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.prompt_embeds = None
        self.prompt = None
        self.hw = None
        self.dev = torch.device("cuda")
        # self.vae = torch.compile(self.vae, mode="reduce-overhead", fullgraph=True)
        # self.text_encoder = torch.compile(self.text_encoder, mode="reduce-overhead", fullgraph=True)
        # self.scheduler = torch.compile(self.scheduler, mode="reduce-overhead", fullgraph=True)

    @torch.no_grad()
    def default_infer(
        self,
        prompt: Union[str, List[str]] = None,
        image=None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        opt_pipeline: Optional[bool] = False,
    ):
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if self.prompt_embeds == None or self.prompt != prompt:
            print("new token")
            self.prompt = prompt
            self.prompt_embeds = self._encode_prompt(
                prompt,
                self.dev,
                num_images_per_prompt,
                True,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
        image = image.astype(np.float16) / 255.0
        image = (image.transpose(2, 0, 1)[None, ...] * 2.0) - 1.0
        image = torch.from_numpy(image).to(device=self.device).repeat(3, 1, 1, 1)
        height, width = image.shape[-2:]
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        if self.hw != (height, width):
            self.hw = height, width
        latents = self.prepare_latents(
            batch_size * 1,
            4,
            height,
            width,
            self.prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        extra_step_kwargs = {}
        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = latents.repeat(3, 1, 1, 1)
            scaled_latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            scaled_latent_model_input = torch.concatenate(
                [scaled_latent_model_input, image_latents], dim=1
            )
            noise_pred = self.unet(
                scaled_latent_model_input, t, encoder_hidden_states=self.prompt_embeds
            ).sample
            if self.scheduler_is_in_sigma_space:
                step_index = (self.scheduler.timesteps == t).nonzero().item()
                sigma = self.scheduler.sigmas[step_index]
                noise_pred = latent_model_input - sigma * noise_pred
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            if self.scheduler_is_in_sigma_space:
                noise_pred = (noise_pred - latents) / (-sigma)
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample
        return latents

    # @torch.no_grad()
    def __call__(
        self,
        prompt="",
        image=None,
        generator=None,
        negative_prompt="",
        num_inference_steps=25,
        guidance_scale=7.5,
        image_guidance_scale=1.0,
        *args,
        **kwargs
    ):
        bs = 1  # if isinstance(prompt, str) else len(prompt)
        prompts = [
            "A painting of a squirrel eating a burger",
            "A painting of a squirrel eating a lasagne",
            "A painting of a squirrel eating a pretzel",
            "A painting of a squirrel eating a sushi",
        ]

        """lb = LocalBlend(self.tokenizer, prompts, ("burger", "lasagne", "pretzel", "sushi"))
        controller = AttentionReplace(self.tokenizer, prompts, num_inference_steps,
                                      cross_replace_steps={"default_": 1., "lasagne": .2, "pretzel": .2, "sushi": .2},
                                      self_replace_steps=0.2, local_blend=lb)
        #register_attention_control(self.unet, controller)"""

        if self.prompt != prompt:
            self.prompt, self.prompt_embeds = prompt, self._encode_prompt(
                prompt, self.dev, 1, True, negative_prompt, None, None
            )
        img = (image.astype(np.float16) / 255).transpose(2, 0, 1)[None, ...] * 2 - 1
        img = torch.from_numpy(img).to(self.device).repeat(3, 1, 1, 1)
        h, w = img.shape[-2:]
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        with torch.no_grad():
            img_latents = self.vae.encode(img).latent_dist.mode()
        latents = self.prepare_latents(
            bs, 4, h, w, self.prompt_embeds.dtype, self.device, generator, None
        )
        for i, t in enumerate(self.scheduler.timesteps):
            lat_input = latents.repeat(3, 1, 1, 1)
            sc_lat_input = self.scheduler.scale_model_input(lat_input, t)
            sc_lat_input = torch.cat([sc_lat_input, img_latents], dim=1)
            with torch.no_grad():
                noise_pred = self.unet(
                    sc_lat_input, t, encoder_hidden_states=self.prompt_embeds
                ).sample
            noise_pred_t, noise_pred_i, noise_pred_u = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_u
                + guidance_scale * (noise_pred_t - noise_pred_i)
                + image_guidance_scale * (noise_pred_i - noise_pred_u)
            )
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            # latents = controller.step_callback(latents)
        """latents = resize_right.resize(latents, scale_factors=0.5, out_shape=None,
                            interp_method=interp_methods.cubic, support_sz=None,
                            antialiasing=True, by_convs=False, scale_tolerance=None,
                            max_numerator=10, pad_mode='constant')"""
        return latents
        """latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.permute(0, 2, 3, 1).float().cpu().numpy()
        image = (image * 255).astype("uint8")
        return image[0]"""

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(
                self.safety_checker, execution_device=device, offload_buffers=True
            )

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            prompt_embeds = torch.cat(
                [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
            )

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(self, prompt, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(
        self,
        image,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        do_classifier_free_guidance,
        generator=None,
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            image_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.mode()
                for i in range(batch_size)
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.mode()

        if (
            batch_size > image_latents.shape[0]
            and batch_size % image_latents.shape[0] == 0
        ):
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat(
                [image_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > image_latents.shape[0]
            and batch_size % image_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat(
                [image_latents, image_latents, uncond_image_latents], dim=0
            )

        return image_latents
