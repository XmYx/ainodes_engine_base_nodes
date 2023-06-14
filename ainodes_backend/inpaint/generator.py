import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from einops import repeat, rearrange
from ainodes_frontend import singleton as gs
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.torch_gc import torch_gc
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.inpaint.ddim_sampler import DDIMSampler


def run_inpaint(init_image, mask_img, prompt, seed, scale, steps, blend_mask, mask_blur, recons_blur):
    #print(gs.models)
    torch_gc()
    sampler = DDIMSampler(gs.models["sd"].model)
    image_guide = image_to_torch(init_image, gs.device)[0]
    mask = mask_img
    # Convert the image to grayscale
    grayscale_image = mask.convert("L")

    # Convert the grayscale image to RGB
    mask = grayscale_image.convert("RGB")
    mask = ImageOps.invert(mask)

    [mask_for_reconstruction, latent_mask_for_blend] = get_mask_for_latent_blending(device=gs.device, mask_image=mask,
                                                                                    blur=mask_blur,
                                                                                    recons_blur=recons_blur)
    masked_image_for_blend = (1 - mask_for_reconstruction) * image_guide[0]




    image = init_image
    h = image.size[1]
    w = image.size[0]

    #try:
    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=steps,
        num_samples=1,
        h=h, w=w,
        device=gs.device,
        mask_for_reconstruction=mask_for_reconstruction,
        masked_image_for_blend=masked_image_for_blend,
        callback=None)
    #except Exception as e:
    #    print('inpainting caused an error: ', e)
    #    result = -1
    #    return result
    if result != -1:
        return result[0]


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, device, mask_for_reconstruction,
            masked_image_for_blend, num_samples=1, w=512, h=512, callback=None):
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        with torch.autocast(gs.device):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = gs.models["clip"].encode(prompt)

            c = c.to(gs.device)

            c_cat = list()
            for ck in gs.models["sd"].model.concat_keys:
                cc = batch[ck].float()
                if ck != gs.models["sd"].model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    gs.models["vae"].first_stage_model.cuda()
                    cc = gs.models["sd"].model.get_first_stage_encoding(gs.models["vae"].encode(cc))
                    gs.models["vae"].first_stage_model.cpu()
                cc = cc.to(gs.device)

                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = gs.models["sd"].model.get_unconditional_conditioning(num_samples, "")

            uc_cross = uc_cross.to(gs.device)
            c_cat = c_cat.to(gs.device)

            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [gs.models["sd"].model.channels, h // 8, w // 8]

            if torch.cuda.is_available():
                gs.models["sd"].model.cuda()
            else:
                gs.models["sd"].model.to(gs.device)


            samples_cfg, intermediates = sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=0.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
                img_callback=callback,
            )

            #print(samples_cfg)
            gs.models["sd"].model.cpu()
            #x_samples = encoded_to_torch_image(
            #    gs.models["sd"].model, samples_cfg)  # [1, 3, 512, 512]

            x_samples = gs.models["vae"].decode(samples_cfg.half())
            x_samples = 255. * x_samples[0].detach().numpy()
            result = [Image.fromarray(x_samples.astype(np.uint8))]
            #print("END", x_samples[0].shape, mask_for_reconstruction.shape, masked_image_for_blend.shape)
    return result


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

def image_to_torch(image, device):
    source_w, source_h = image.size
    w, h = map(lambda x: x - x % 64, (source_w, source_h))  # resize to integer multiple of 32
    if source_w != w or source_h != h:
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image.half().to(device)
def encoded_to_torch_image(model, encoded_image):
    decoded = model.decode_first_stage(encoded_image)
    return torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
def get_mask_for_latent_blending(device, mask_image, blur = 0, recons_blur=0):
    #print(path)
    mask_image = mask_image.convert("L")

    if blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur))

    mask_for_reconstruction = mask_image.point(lambda x: 255 if x > 0 else 0)
    if recons_blur > 0:
        mask_for_reconstruction = mask_for_reconstruction.filter(
            ImageFilter.GaussianBlur(radius=recons_blur))
    mask_for_reconstruction = mask_for_reconstruction.point(
        lambda x: 255 if x > 127 else x * 2)

    mask_for_reconstruction = torch.from_numpy(
        (np.array(mask_for_reconstruction) / 255.0).astype(np.float16)).to(device)

    source_w, source_h = mask_image.size


    mask = np.array(
        mask_image.resize(
            (int(source_w / 8), int(source_h / 8)), resample=Image.Resampling.LANCZOS).convert("L"))
    mask = (mask / 255.0).astype(np.float16)

    mask = mask[None]
    mask = 1 - mask

    mask = torch.from_numpy(mask)
    #mask = torch.stack([mask, mask, mask, mask], 1).to(device)  # FIXME
    return [mask_for_reconstruction, mask]
def sampleToImage (sample):
    #sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
    return Image.fromarray(sample.astype(np.uint8))
