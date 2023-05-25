import torch
from packaging import version
import importlib
#from modules import devices
#from modules.sd_hijack_utils import CondFunc


class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


#th = TorchHijackForUnet()

dtype_unet = torch.float16
# Below are monkey patches to enable upcasting a float16 UNet for float32 sampling
def apply_model(orig_func, self, x_noisy, t, cond, **kwargs):

    if isinstance(cond, dict):
        for y in cond.keys():
            cond[y] = [x.to(dtype_unet) if isinstance(x, torch.Tensor) else x for x in cond[y]]

    with torch.autocast("cuda"):
        return orig_func(self, x_noisy.to(dtype_unet), t.to(dtype_unet), cond, **kwargs).float()


class GELUHijack(torch.nn.GELU, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.GELU.__init__(self, *args, **kwargs)
    def forward(self, x):
        #if devices.unet_needs_upcast:
        #    return torch.nn.GELU.forward(self.float(), x.float()).to(dtype_unet)
        #else:
        return torch.nn.GELU.forward(self, x)

class CondFunc:
    def __new__(cls, orig_func, sub_func, cond_func):
        self = super(CondFunc, cls).__new__(cls)
        if isinstance(orig_func, str):
            func_path = orig_func.split('.')
            for i in range(len(func_path)-1, -1, -1):
                try:
                    resolved_obj = importlib.import_module('.'.join(func_path[:i]))
                    break
                except ImportError:
                    pass
            for attr_name in func_path[i:-1]:
                resolved_obj = getattr(resolved_obj, attr_name)
            orig_func = getattr(resolved_obj, func_path[-1])
            setattr(resolved_obj, func_path[-1], lambda *args, **kwargs: self(*args, **kwargs))
        self.__init__(orig_func, sub_func, cond_func)
        return lambda *args, **kwargs: self(*args, **kwargs)
    def __init__(self, orig_func, sub_func, cond_func):
        self.__orig_func = orig_func
        self.__sub_func = sub_func
        self.__cond_func = cond_func
    def __call__(self, *args, **kwargs):
        if not self.__cond_func or self.__cond_func(self.__orig_func, *args, **kwargs):
            return self.__sub_func(self.__orig_func, *args, **kwargs)
        else:
            return self.__orig_func(*args, **kwargs)

ddpm_edit_hijack = None
def hijack_ddpm_edit():
    global ddpm_edit_hijack
    if not ddpm_edit_hijack:
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
        ddpm_edit_hijack = CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)


"""unet_needs_upcast = lambda *args, **kwargs: False
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)
CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', lambda orig_func, timesteps, *args, **kwargs: orig_func(timesteps, *args, **kwargs).to(torch.float32 if timesteps.dtype == torch.int64 else dtype_unet), unet_needs_upcast)
if version.parse(torch.__version__) <= version.parse("1.13.2") or torch.cuda.is_available():
    CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', lambda orig_func, self, *args, **kwargs: orig_func(self.float(), *args, **kwargs), unet_needs_upcast)
    CondFunc('ldm.modules.attention.GEGLU.forward', lambda orig_func, self, x: orig_func(self.float(), x.float()).to(dtype_unet), unet_needs_upcast)
    CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)

first_stage_cond = lambda _, self, *args, **kwargs: False and self.model.diffusion_model.dtype == torch.float16
first_stage_sub = lambda orig_func, self, x, **kwargs: orig_func(self, x.to(torch.float16), **kwargs)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).float(), first_stage_cond)"""