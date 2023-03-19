import os
from typing import Optional, Dict

import numpy as np
import torch
from PIL import Image
from diffusers.optimization import get_scheduler
from ainodes_frontend import singleton
gs = singleton.Singleton.instance()
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import custom_nodes.ainodes_engine_base_nodes.ainodes_backend.video_diffusion.pipelines.p2pDDIMSpatioTemporalPipeline
from einops import rearrange

import importlib

from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.video_diffusion.common.image_util import log_train_samples
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.video_diffusion.data.dataset import ImageSequenceDataset
from .common import logger
from .common.util import get_function_args, get_time_string
from .models.unet_3d_condition import UNetPseudo3DConditionModel
from .pipelines.p2pvalidation_loop import p2pSampleLogger


def collate_fn(examples):
    batch = {
        "prompt_ids": torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples]),
    }
    return batch
def instantiate_from_config(config:dict, **args_from_code):
    """Util funciton to decompose differenct modules using config
    Args:
        config (dict): with key of "target" and "params", better from yaml
        static
        args_from_code: additional con
    Returns:
        a validation/training pipeline, a module
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **args_from_code)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_fate_zero():

    Omegadict = OmegaConf.load("fate_zero.yaml")
    if 'unet' in os.listdir(Omegadict['pretrained_model_path']):

        test(config="fate_zero.yaml", **Omegadict)


    """if "fate_sd" not in gs.models:
        model_config = OmegaConf.load("fate_zero.yaml")
        pretrained_model_path = "ckpt/stable-diffusion-v1-5"
        print(model_config)
        kwargs = {'epsilon': 1e-05, 'train_steps': 10, 'learning_rate': 1e-05, 'train_temporal_conv': False, 'guidance_scale': 7.5}

        test_pipeline_config = {'target': 'custom_nodes.ainodes_engine_base_nodes.ainodes_backend.video_diffusion.pipelines.p2pDDIMSpatioTemporalPipeline.p2pDDIMSpatioTemporalPipeline', 'num_inference_steps': '${..validation_sample_logger.num_inference_steps}'}

        model_config = {'lora': 160, 'SparseCausalAttention_index': ['mid'], 'least_sc_channel': 1280}

        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            subfolder="tokenizer",
            use_fast=False,
        )

        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path,
            subfolder="text_encoder",
        )

        vae = AutoencoderKL.from_pretrained(
            pretrained_model_path,
            subfolder="vae",
        )

        from .models.unet_3d_condition import UNetPseudo3DConditionModel
        unet = UNetPseudo3DConditionModel.from_2d_model(
            os.path.join(pretrained_model_path, "unet"), model_config=model_config
        )

        test_pipeline_config = {
            "target": 'custom_nodes.ainodes_engine_base_nodes.ainodes_backend.video_diffusion.pipelines.stable_diffusion.SpatioTemporalStableDiffusionPipeline',
            "num_inference_steps": 50

        }

        pipeline = instantiate_from_config(
            test_pipeline_config,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(
                pretrained_model_path,
                subfolder="scheduler",
            ),
        )
        pipeline.scheduler.set_timesteps(50)
        pipeline.set_progress_bar_config(disable=False)
        pipeline.enable_xformers_memory_efficient_attention()

        prompt_ids = tokenizer(
            "Cityscape",
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        train_dataset = {
                        "path": "data/car-turn",
                        "prompt": "a silver jeep driving down a curvy road in the countryside",
                        "n_sample_frame": 8,
                        "sampling_rate": 1,
                        "stride": 80,
                        "offset":{
                                "left": 0,
                                "right": 0,
                                "top": 0,
                                "bottom": 0}
                      }

        train_dataset = ImageSequenceDataset(**train_dataset, prompt_ids=prompt_ids)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )
        train_sample_save_path = os.path.join("output", "train_samples.gif")
        log_train_samples(save_path=train_sample_save_path, train_dataloader=train_dataloader)

        # breakpoint()
        unet, train_dataloader = accelerator.prepare(
            unet, train_dataloader
        )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            print('use fp16')
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("video")  # , config=vars(args))
        #logger.info("***** wait to fix the logger path *****")

        if model_config["validation_sample_logger_config"] is not None and accelerator.is_main_process:
            validation_sample_logger = p2pSampleLogger(**model_config["validation_sample_logger_config"], logdir="output")
            # validation_sample_logger.log_sample_images(
            #     pipeline=pipeline,
            #     device=accelerator.device,
            #     step=0,
            # )

        def make_data_yielder(dataloader):
            while True:
                for batch in dataloader:
                    yield batch
                accelerator.wait_for_everyone()

        train_data_yielder = make_data_yielder(train_dataloader)
        # breakpoint()

        # while step < train_steps:
        batch = next(train_data_yielder)

        batch['ddim_init_latents'] = None

        vae.eval()
        text_encoder.eval()
        unet.train()

        # with accelerator.accumulate(unet):
        # Convert images to latent space
        images = batch["images"].to(dtype=weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")

        if accelerator.is_main_process:

            if validation_sample_logger is not None:
                unet.eval()
                # breakpoint()
                validation_sample_logger.log_sample_images(
                    image=images,  # torch.Size([8, 3, 512, 512])
                    pipeline=pipeline,
                    device=accelerator.device,
                    step=0,
                    latents=batch['ddim_init_latents'],
                    save_dir="output"
                )
            # accelerator.log(logs, step=step)

        accelerator.end_training()"""
def image_to_torch(image, device):
    source_w, source_h = image.size
    w, h = map(lambda x: x - x % 64, (source_w, source_h))  # resize to integer multiple of 32
    if source_w != w or source_h != h:
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image.half().to(device)


def infer_fate(image=None):
    return
    image = Image.open("test.png")

    image_torch = image_to_torch(image, "cuda")
    if "fate_sd" in gs.models:
        images = gs.models["fate_sd"](
                            edit_type = "swap",
                            prompt = "Test Prompt",
                            source_prompt = "Test Source prompt",
                            image= image_torch,
                            height= 512,
                            width= 512,
                            strength= 1.0,
                            num_inference_steps=  50,
                            cross_replace_steps=0.8,
                            self_replace_steps=0.9,
                            use_inversion_attention=False,
                            clip_length= 8,
                            guidance_scale= 7.5,
                            negative_prompt= None,
                            num_images_per_prompt=1,
                            eta=  0.0,
                            generator= None,
                            latents = None,
                            output_type = "pil",
                            return_dict= True,
                            callback = None,
                            callback_steps= 1)
        print(images)


def test(
        config: str,
        pretrained_model_path: str,
        train_dataset: Dict,
        logdir: str = None,
        validation_sample_logger_config: Optional[Dict] = None,
        test_pipeline_config: Optional[Dict] = None,
        gradient_accumulation_steps: int = 1,
        seed: Optional[int] = None,
        mixed_precision: Optional[str] = "fp16",
        train_batch_size: int = 1,
        model_config: dict = {},
        **kwargs

):
    print("kwargs", kwargs)
    print("test_pipeline_config", test_pipeline_config)
    print("model_config", model_config)


    args = get_function_args()

    print("args", args)

    time_string = get_time_string()
    logdir = "output"
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    )

    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet"), model_config=model_config
    )
    use_8bit_adam = True
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=3e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )


    if 'target' not in test_pipeline_config:
        test_pipeline_config[
            'target'] = 'video_diffusion.pipelines.stable_diffusion.SpatioTemporalStableDiffusionPipeline'
    pipeline = instantiate_from_config(
        test_pipeline_config,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        ),
    )
    pipeline.scheduler.set_timesteps(validation_sample_logger_config['num_inference_steps'])
    pipeline.set_progress_bar_config(disable=True)

    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception as e:
        logger.warning(
            "Could not enable memory efficient attention. Make sure xformers is installed"
            f" correctly and a GPU is available: {e}"
        )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    prompt_ids = tokenizer(
        train_dataset["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    train_dataset = ImageSequenceDataset(**train_dataset, prompt_ids=prompt_ids)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    train_sample_save_path = os.path.join(logdir, "train_samples.gif")
    log_train_samples(save_path=train_sample_save_path, train_dataloader=train_dataloader)
    gradient_accumulation_steps = 1
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0 * gradient_accumulation_steps,
        num_training_steps=300 * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    accelerator.register_for_checkpointing(lr_scheduler)
    # breakpoint()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        print('use fp16')
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  # , config=vars(args))
    #logger.info("***** wait to fix the logger path *****")

    if validation_sample_logger_config is not None and accelerator.is_main_process:
        validation_sample_logger = p2pSampleLogger(**validation_sample_logger_config, logdir=logdir)
        # validation_sample_logger.log_sample_images(
        #     pipeline=pipeline,
        #     device=accelerator.device,
        #     step=0,
        # )

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)
    # breakpoint()

    # while step < train_steps:
    batch = next(train_data_yielder)
    if validation_sample_logger_config.get('use_train_latents', False):
        # Precompute the latents for this video to align the initial latents in training and test
        assert batch["images"].shape[0] == 1, "Only support, overfiting on a single video"
        # we only inference for latents, no training
        vae.eval()
        text_encoder.eval()
        unet.eval()

        text_embeddings = pipeline._encode_prompt(
            train_dataset.prompt,
            device=accelerator.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None
        )

        use_inversion_attention = validation_sample_logger_config.get('use_inversion_attention', False)
        batch['latents_all_step'] = pipeline.prepare_latents_ddim_inverted(
            rearrange(batch["images"].to(dtype=weight_dtype), "b c f h w -> (b f) c h w"),
            batch_size=1,
            num_images_per_prompt=1,  # not sure how to use it
            text_embeddings=text_embeddings,
            prompt=train_dataset.prompt,
            store_attention=use_inversion_attention,
            LOW_RESOURCE=True,  # not classifier-free guidance
            save_path=logdir
        )

        batch['ddim_init_latents'] = batch['latents_all_step'][-1]

    else:
        batch['ddim_init_latents'] = None

    vae.eval()
    text_encoder.eval()
    unet.train()

    # with accelerator.accumulate(unet):
    # Convert images to latent space
    images = batch["images"].to(dtype=weight_dtype)
    b = images.shape[0]
    images = rearrange(images, "b c f h w -> (b f) c h w")

    if accelerator.is_main_process:

        if validation_sample_logger is not None:
            unet.eval()
            # breakpoint()
            validation_sample_logger.log_sample_images(
                image=images,  # torch.Size([8, 3, 512, 512])
                pipeline=pipeline,
                device=accelerator.device,
                step=0,
                latents=batch['ddim_init_latents'],
                save_dir=logdir
            )
        # accelerator.log(logs, step=step)

    accelerator.end_training()