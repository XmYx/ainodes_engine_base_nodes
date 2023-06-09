import torch

#from comfy import model_management, samplers
from ainodes_frontend import singleton as gs
from . import samplers
from .torch_gc import torch_gc


def common_ksampler(device, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, callback=None, model_key="sd", noise_mask=None):
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
        if 'control' in p[1]:
            control_nets += [p[1]['control']]
        negative_copy += [[t] + n[1:]]
    control_net_models = []
    for x in control_nets:
        control_net_models += x.get_control_models()
        for i in control_net_models:
            i.cuda()
    if "controlnet" in gs.models:
        gs.models["controlnet"].control_model.cuda()
    #gs.models["sd"].model.cuda()
    if sampler_name in samplers.KSampler.SAMPLERS:
        model = gs.models["sd"].clone()
        model.model.cuda()
        if 'transformer_options' in model.model_options:
            for key, value in model.model_options.items():
                #print(key, value)
                for item_name, items in value.items():
                    for _, models in items.items():
                        for m in models:
                            m.to("cuda")

        sampler = samplers.KSampler(steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model=model, model_options=model.model_options)

    else:
        #other samplers
        pass

    samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, callback=callback, model_key=model_key)
    if sampler_name in samplers.KSampler.SAMPLERS:
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