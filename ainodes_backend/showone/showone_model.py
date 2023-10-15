import os

import torch
from PIL import Image
from diffusers import IFSuperResolutionPipeline, VideoToVideoSDPipeline
from diffusers.utils.torch_utils import randn_tensor

from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.pipelines import TextToVideoIFPipeline, \
    TextToVideoIFInterpPipeline
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.pipelines.pipeline_t2v_sr_pixel_cond import \
    TextToVideoIFSuperResolutionPipeline_Cond, tensor2vid


class VideoGenerator:
    def __init__(self, seed=345):
        # Define model paths
        self.base_model_path = "showlab/show-1-base"
        self.interpolation_model_path = "showlab/show-1-interpolation"
        self.super_resolution_model_path = "DeepFloyd/IF-II-L-v1.0"
        self.sr1_model_path = "showlab/show-1-sr1"
        self.sr2_model_path = "showlab/show-1-sr2"

        # Initialize base model
        self.pipe_base = TextToVideoIFPipeline.from_pretrained(self.base_model_path,
                                                               torch_dtype=torch.float16,
                                                               variant="fp16")
        self.pipe_base.enable_model_cpu_offload()

        # Initialize interpolation model
        self.pipe_interp_1 = TextToVideoIFInterpPipeline.from_pretrained(
            self.interpolation_model_path, torch_dtype=torch.float16, variant="fp16")
        self.pipe_interp_1.enable_model_cpu_offload()

        # Initialize image super-resolution model
        self.pipe_sr_1_image = IFSuperResolutionPipeline.from_pretrained(
            self.super_resolution_model_path,
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16)
        self.pipe_sr_1_image.enable_model_cpu_offload()

        # Initialize super-resolution model 1
        self.pipe_sr_1_cond = TextToVideoIFSuperResolutionPipeline_Cond.from_pretrained(
            self.sr1_model_path, torch_dtype=torch.float16)
        self.pipe_sr_1_cond.enable_model_cpu_offload()

        # Initialize super-resolution model 2
        self.pipe_sr_2 = VideoToVideoSDPipeline.from_pretrained(self.sr2_model_path,
                                                                torch_dtype=torch.float16)
        self.pipe_sr_2.enable_model_cpu_offload()
        self.pipe_sr_2.enable_vae_slicing()

        # Set seed
        self.seed = seed

    def __call__(self,
                 prompt,
                 output_dir="output/showone-tensors",
                 negative_prompt="low resolution, blur",
                 frames=8,
                 width=64,
                 height=40):
        # Ensure output directory exists
        #os.makedirs(output_dir, exist_ok=True)

        # Text embeddings
        prompt_embeds, negative_embeds = self.pipe_base.encode_prompt(prompt)

        # Keyframes generation
        video_frames = self.pipe_base(prompt_embeds=prompt_embeds,
                                      negative_prompt_embeds=negative_embeds,
                                      num_frames=frames,
                                      height=height,
                                      width=width,
                                      num_inference_steps=75,
                                      guidance_scale=9.0,
                                      generator=torch.manual_seed(self.seed),
                                      output_type="pt").frames

        # Interpolation
        bsz, channel, num_frames, height, width = video_frames.shape
        new_num_frames = 3 * (num_frames - 1) + num_frames
        new_video_frames = torch.zeros((bsz, channel, new_num_frames, height, width),
                                       dtype=video_frames.dtype,
                                       device=video_frames.device)
        new_video_frames[:, :, torch.arange(0, new_num_frames, 4), ...] = video_frames
        init_noise = randn_tensor((bsz, channel, 5, height, width),
                                  generator=torch.manual_seed(self.seed),
                                  device=video_frames.device,
                                  dtype=video_frames.dtype)
        for i in range(num_frames - 1):
            batch_i = torch.zeros((bsz, channel, 5, height, width),
                                  dtype=video_frames.dtype,
                                  device=video_frames.device)
            batch_i[:, :, 0, ...] = video_frames[:, :, i, ...]
            batch_i[:, :, -1, ...] = video_frames[:, :, i + 1, ...]
            batch_i = self.pipe_interp_1(
                pixel_values=batch_i,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                num_frames=batch_i.shape[2],
                height=40,
                width=64,
                num_inference_steps=75,
                guidance_scale=4.0,
                generator=torch.manual_seed(self.seed),
                output_type="pt",
                init_noise=init_noise,
                cond_interpolation=True,
            ).frames
            new_video_frames[:, :, i * 4:i * 4 + 5, ...] = batch_i
        video_frames = new_video_frames

        # SR1
        bsz, channel, num_frames, height, width = video_frames.shape
        window_size, stride = 8, 7
        new_video_frames = torch.zeros(
            (bsz, channel, num_frames, height * 4, width * 4),
            dtype=video_frames.dtype,
            device=video_frames.device)
        for i in range(0, num_frames - window_size + 1, stride):
            batch_i = video_frames[:, :, i:i + window_size, ...]
            all_frame_cond = None
            if i == 0:
                first_frame_cond = self.pipe_sr_1_image(
                    image=video_frames[:, :, 0, ...],
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    height=height * 4,
                    width=width * 4,
                    num_inference_steps=70,
                    guidance_scale=4.0,
                    noise_level=150,
                    generator=torch.manual_seed(self.seed),
                    output_type="pt").images
                first_frame_cond = first_frame_cond.unsqueeze(2)
            else:
                first_frame_cond = new_video_frames[:, :, i:i + 1, ...]
            batch_i = self.pipe_sr_1_cond(image=batch_i,
                                          prompt_embeds=prompt_embeds,
                                          negative_prompt_embeds=negative_embeds,
                                          first_frame_cond=first_frame_cond,
                                          height=height * 4,
                                          width=width * 4,
                                          num_inference_steps=125,
                                          guidance_scale=7.0,
                                          noise_level=250,
                                          generator=torch.manual_seed(self.seed),
                                          output_type="pt").frames
            new_video_frames[:, :, i:i + window_size, ...] = batch_i
        video_frames = new_video_frames

        # SR2
        video_frames = [
            Image.fromarray(frame).resize((576, 320))
            for frame in tensor2vid(video_frames.clone())
        ]
        video_frames = self.pipe_sr_2(prompt,
                                      negative_prompt=negative_prompt,
                                      video=video_frames,
                                      strength=0.8,
                                      num_inference_steps=50,
                                      generator=torch.manual_seed(self.seed),
                                      output_type="pt").frames

        return video_frames
