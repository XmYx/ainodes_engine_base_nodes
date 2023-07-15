import torch

#from comfy import model_management, samplers
from ainodes_frontend import singleton as gs
from .torch_gc import torch_gc





def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    #device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]
    from comfy.sample import prepare_noise
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    # previewer = latent_preview.get_previewer(device, model.model.latent_format)
    #
    # pbar = comfy.utils.ProgressBar(steps)
    # def callback(step, x0, x, total_steps):
    #     preview_bytes = None
    #     if previewer:
    #         preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
    #     pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=None, seed=seed)

    out = latent.copy()
    out["samples"] = samples


    del negative
    del positive
    if "controlnet" in gs.models:
        gs.models["controlnet"].control_model.cpu()
    torch_gc()


    return (out, )


import torch

import math
import numpy as np


def prepare_noise(latent_image, seed, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                           generator=generator, device="cpu")

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout,
                            generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def prepare_mask(noise_mask, shape, device):
    """ensures noise mask is of proper dimensions"""
    noise_mask = torch.nn.functional.interpolate(
        noise_mask.reshape((-1, 1, noise_mask.shape[-2], noise_mask.shape[-1])), size=(shape[2], shape[3]),
        mode="bilinear")
    noise_mask = noise_mask.round()
    noise_mask = torch.cat([noise_mask] * shape[1], dim=1)
    if noise_mask.shape[0] < shape[0]:
        noise_mask = noise_mask.repeat(math.ceil(shape[0] / noise_mask.shape[0]), 1, 1, 1)[:shape[0]]
    noise_mask = noise_mask.to(device)
    return noise_mask


def broadcast_cond(cond, batch, device):
    """broadcasts conditioning to the batch size"""
    copy = []
    for p in cond:
        t = p[0]
        if t.shape[0] < batch:
            t = torch.cat([t] * batch)
        t = t.to(device)
        copy += [[t] + p[1:]]
    return copy


def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c[1]:
            models += [c[1][model_type]]
    return models


def load_additional_models(positive, negative, dtype):
    from comfy import model_management
    """loads additional models in positive and negative conditioning"""
    control_nets = get_models_from_cond(positive, "control") + get_models_from_cond(negative, "control")
    gligen = get_models_from_cond(positive, "gligen") + get_models_from_cond(negative, "gligen")
    gligen = [x[1].to(dtype) for x in gligen]
    models = control_nets + gligen
    model_management.load_controlnet_gpu(models)
    return models


def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        m.cleanup()


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
           disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None,
           callback=None, disable_pbar=False, seed=None):
    from comfy import model_management
    from comfy import samplers

    device = model_management.get_torch_device()

    if noise_mask is not None:
        noise_mask = prepare_mask(noise_mask, noise.shape, device)

    real_model = None
    model_management.load_model_gpu(model)
    #real_model = model.model

    noise = noise.to(device)
    latent_image = latent_image.to(device)

    positive_copy = broadcast_cond(positive, noise.shape[0], device)
    negative_copy = broadcast_cond(negative, noise.shape[0], device)

    models = load_additional_models(positive, negative, model.model_dtype())

    sampler = KSampler(model.model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler,
                                      denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(model.model, noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image,
                             start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise,
                             denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar,
                             seed=seed)
    samples = samples.cpu()

    cleanup_additional_models(models)

    del negative_copy
    del positive_copy
    del sampler.model_k
    del sampler.model_wrap.inner_model
    del sampler.model_wrap
    del sampler.model_denoise
    del sampler


    return samples


def common_ksampler_(device, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, callback=None, model=None, control_model=None, noise_mask=None):
    latent_image = latent
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=torch.manual_seed(seed), device="cpu")


    if noise_mask is not None:
        noise_mask = torch.nn.functional.interpolate(noise_mask, size=(noise.shape[2], noise.shape[3]), mode="bilinear")
        noise_mask = noise_mask.round()
        noise_mask = torch.cat([noise_mask] * noise.shape[1], dim=1)
        noise_mask = torch.cat([noise_mask] * noise.shape[0])
        noise_mask = noise_mask.to(device)
    real_model = None
    noise = noise.to(device)
    latent_image = latent.to(device)
    positive_copy = []
    negative_copy = []
    control_nets = []
    for p in positive:
        t = p[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        t = t.to(device)
        if 'control' in p[1]:
            control_nets += [p[1]['control']]
        positive_copy += [[t] + p[1:]]
    for n in negative:
        t = n[0]
        if t.shape[0] < noise.shape[0]:
            t = torch.cat([t] * noise.shape[0])
        t = t.to(device)
        if 'control' in n[1]:
            control_nets += [n[1]['control']]
        negative_copy += [[t] + n[1:]]
    control_net_models = []
    for x in control_nets:
        control_net_models += x.get_control_models()

        print("CONTROLMODELS", len(control_net_models))

        for i in control_net_models:
            print(i.__class__)
            i.cuda()
    # if control_model is not None:
    #     control_model.cuda()
    # if "controlnet" in gs.models:
    #     gs.models["controlnet"].control_model.cuda()
    #gs.models["sd"].model.cuda()
    model.model.cuda()

    if sampler_name in KSampler.SAMPLERS:
        #model = gs.models["sd"].clone()
        if 'transformer_options' in model.model_options:
            for key, value in model.model_options.items():
                #print(key, value)
                for item_name, items in value.items():
                    for _, models in items.items():
                        for m in models:
                            m.to(device)

        sampler = KSampler(steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model=model, model_options=model.model_options)

    else:
        #other samplers
        pass

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, callback=callback)
    if sampler_name in KSampler.SAMPLERS:
        if 'transformer_options' in model.model_options:
            for key, value in model.model_options.items():
                # print(key, value)
                for item_name, items in value.items():
                    for _, models in items.items():
                        for m in models:
                            m.to("cpu")
                            del m

    #sampler.model.model.cpu()
    #del model
    #del sampler.model
    #gs.models["sd"].model.cpu()
    #samples = samples.cpu()
    model.model.cpu()
    for c in control_nets:
        c.cleanup()
        c = None
        del c
    noise = noise.to("cpu")
    latent_image = latent_image.to("cpu")
    del noise
    del latent_image


    del negative
    del positive
    del negative_copy
    del positive_copy
    del control_nets
    del sampler.model_k
    del sampler.model_wrap.inner_model
    del sampler.model_wrap
    del sampler.model_denoise
    del sampler
    if "controlnet" in gs.models:
        gs.models["controlnet"].control_model.cpu()
    torch_gc()

    return samples

class KSampler:
    SCHEDULERS = ["normal", "karras", "exponential", "simple", "ddim_uniform"]
    SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"]

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        from comfy.k_diffusion import external as k_diffusion_external
        #self.model = model
        from comfy.samplers import CFGNoisePredictor, CompVisVDenoiser, KSamplerX0Inpaint

        self.model_denoise = CFGNoisePredictor(model)
        if model.parameterization == "v":
            self.model_wrap = CompVisVDenoiser(self.model_denoise, quantize=True)
        else:
            self.model_wrap = k_diffusion_external.CompVisDenoiser(self.model_denoise, quantize=True)
        self.model_wrap.parameterization = model.parameterization
        self.model_k = KSamplerX0Inpaint(self.model_wrap)
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.sigma_min=float(self.model_wrap.sigma_min)
        self.sigma_max=float(self.model_wrap.sigma_max)
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from . import samplers, CFGNoisePredictor, CompVisVDenoiser, KSamplerX0Inpaint, simple_scheduler, \
            ddim_scheduler, \
            create_cond_with_same_area_if_none

        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in ['dpm_2', 'dpm_2_ancestral']:
            steps += 1
            discard_penultimate_sigma = True

        if self.scheduler == "karras":
            sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        elif self.scheduler == "exponential":
            sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        elif self.scheduler == "normal":
            sigmas = self.model_wrap.get_sigmas(steps)
        elif self.scheduler == "simple":
            sigmas = simple_scheduler(self.model_wrap, steps)
        elif self.scheduler == "ddim_uniform":
            sigmas = ddim_scheduler(self.model_wrap, steps)
        else:
            print("error invalid scheduler", self.scheduler)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, model, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        from comfy.k_diffusion import sampling as k_diffusion_sampling
        from comfy.extra_samplers import uni_pc
        from comfy.ldm.models.diffusion.ddim import DDIMSampler
        from comfy.samplers import resolve_cond_masks, apply_empty_x_to_equal_area, encode_adm, \
            blank_inpaint_image_like
        from comfy.samplers import sampling_function

        from . import samplers, CFGNoisePredictor, CompVisVDenoiser, KSamplerX0Inpaint, simple_scheduler, \
            ddim_scheduler, \
            create_cond_with_same_area_if_none
        print("using new sampler")
        if sigmas is None:
            sigmas = self.sigmas
        sigma_min = self.sigma_min

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigma_min = sigmas[last_step]
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        positive = positive[:]
        negative = negative[:]

        resolve_cond_masks(positive, noise.shape[2], noise.shape[3], self.device)
        resolve_cond_masks(negative, noise.shape[2], noise.shape[3], self.device)

        #make sure each cond area has an opposite one with the same area
        for c in positive:
            create_cond_with_same_area_if_none(negative, c)
        for c in negative:
            create_cond_with_same_area_if_none(positive, c)

        apply_empty_x_to_equal_area(positive, negative, 'control', lambda cond_cnets, x: cond_cnets[x])
        apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

        if model.is_adm():
            positive = encode_adm(model, positive, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "positive")
            negative = encode_adm(model, negative, noise.shape[0], noise.shape[3], noise.shape[2], self.device, "negative")

        if latent_image is not None:
            latent_image = model.process_latent_in(latent_image)

        extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": self.model_options, "seed":seed}

        cond_concat = None
        if hasattr(model, 'concat_keys'): #inpaint
            cond_concat = []
            for ck in model.concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask[:,:1])
                    elif ck == "masked_image":
                        cond_concat.append(latent_image) #NOTE: the latent_image should be masked by the mask in pixel space
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:,:1])
                    elif ck == "masked_image":
                        cond_concat.append(blank_inpaint_image_like(noise))
            extra_args["cond_concat"] = cond_concat

        if sigmas[0] != self.sigmas[0] or (self.denoise is not None and self.denoise < 1.0):
            max_denoise = False
        else:
            max_denoise = True


        if self.sampler == "uni_pc":
            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask, callback=callback, disable=disable_pbar)
        elif self.sampler == "uni_pc_bh2":
            samples = uni_pc.sample_unipc(self.model_wrap, noise, latent_image, sigmas, sampling_function=sampling_function, max_denoise=max_denoise, extra_args=extra_args, noise_mask=denoise_mask, callback=callback, variant='bh2', disable=disable_pbar)
        elif self.sampler == "ddim":
            timesteps = []
            for s in range(sigmas.shape[0]):
                timesteps.insert(0, self.model_wrap.sigma_to_t(sigmas[s]))
            noise_mask = None
            if denoise_mask is not None:
                noise_mask = 1.0 - denoise_mask

            ddim_callback = None
            if callback is not None:
                total_steps = len(timesteps) - 1
                ddim_callback = lambda pred_x0, i: callback(i, pred_x0, None, total_steps)

            sampler = DDIMSampler(model, device=self.device)
            sampler.make_schedule_timesteps(ddim_timesteps=timesteps, verbose=False)
            z_enc = sampler.stochastic_encode(latent_image, torch.tensor([len(timesteps) - 1] * noise.shape[0]).to(self.device), noise=noise, max_denoise=max_denoise)
            samples, _ = sampler.sample_custom(ddim_timesteps=timesteps,
                                                    conditioning=positive,
                                                    batch_size=noise.shape[0],
                                                    shape=noise.shape[1:],
                                                    verbose=False,
                                                    unconditional_guidance_scale=cfg,
                                                    unconditional_conditioning=negative,
                                                    eta=0.0,
                                                    x_T=z_enc,
                                                    x0=latent_image,
                                                    img_callback=ddim_callback,
                                                    denoise_function=sampling_function,
                                                    extra_args=extra_args,
                                                    mask=noise_mask,
                                                    to_zero=sigmas[-1]==0,
                                                    end_step=sigmas.shape[0] - 1,
                                                    disable_pbar=disable_pbar)

        else:
            extra_args["denoise_mask"] = denoise_mask
            self.model_k.latent_image = latent_image
            self.model_k.noise = noise

            if max_denoise:
                noise = noise * torch.sqrt(1.0 + sigmas[0] ** 2.0)
            else:
                noise = noise * sigmas[0]

            k_callback = None
            total_steps = len(sigmas) - 1
            if callback is not None:
                k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

            if latent_image is not None:
                noise += latent_image
            if self.sampler == "dpm_fast":
                samples = k_diffusion_sampling.sample_dpm_fast(self.model_k, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=k_callback, disable=disable_pbar)
            elif self.sampler == "dpm_adaptive":
                samples = k_diffusion_sampling.sample_dpm_adaptive(self.model_k, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=k_callback, disable=disable_pbar)
            else:
                samples = getattr(k_diffusion_sampling, "sample_{}".format(self.sampler))(self.model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar)

        return model.process_latent_out(samples.to(torch.float32))