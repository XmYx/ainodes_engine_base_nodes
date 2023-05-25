#!/usr/bin/env python


import os

from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
)

cache_dir = "models/stable-diffusion-v1-5-cache"
vae_cache_dir = "models/sd-vae-ft-mse-cache"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(vae_cache_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
)

pipe.save_pretrained(cache_dir)


pretrained_vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse", subfolder=None, revision=None
)
pretrained_vae.save_pretrained(vae_cache_dir)