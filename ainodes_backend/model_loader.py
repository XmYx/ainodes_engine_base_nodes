"""
aiNodes node engine

stable diffusion pytorch model loader

www.github.com/XmYx/ainodes-engine
miklos.mnagy@gmail.com
"""
import hashlib
import os

import numpy as np
from omegaconf import OmegaConf
from torch.nn.functional import silu

from ldm.models.autoencoder import AutoencoderKL
from .chainner_models import model_loading
from .lora_loader import ModelPatcher
from .sd_optimizations.sd_hijack import apply_optimizations
from .torch_gc import torch_gc
from ldm.util import instantiate_from_config
from ainodes_frontend import singleton as gs

from .ESRGAN import model as upscaler

import torch
from torch import nn, autocast
import safetensors.torch
import ldm.modules.diffusionmodules.model

import torch.onnx

import pycuda.driver as cuda
#import pycuda.autoinit
import tensorrt as trt

cuda.init()
TRT_LOGGER = trt.Logger()

engine_file = "unet.trt"

engine = None
device = cuda.Device(0)  # enter your Gpu id here

ctx = device.make_context()
ctx = cuda.Context.attach()

trt.init_libnvinfer_plugins(None, "")

trtcontext = None

class UpscalerLoader(torch.nn.Module):

    """
    Torch Upscale model loader
    """

    def __init__(self, parent=None):
        super().__init__()
        self.device = "cuda"
        self.loaded_model = None

    def load_model(self, file="", name=""):
        load = None
        if self.loaded_model:
            if self.loaded_model != name:
                gs.models[self.loaded_model] = None
                del gs.models[self.loaded_model]
                torch_gc()
                load = True
            else:
                load = None
        else:
            load = True

        if load:
            state_dict = load_torch_file(file)
            gs.models[name] = model_loading.load_state_dict(state_dict).eval().to("cuda")
            self.loaded_model = name

        return self.loaded_model
np_to_torch = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,
    np.uint8: torch.uint8,
    np.int32: torch.int32,
}
class ModelLoader(torch.nn.Module):
    """
    Torch SD Model Loader class
    Storage is a Singleton object
    """
    def __init__(self, parent=None):
        super().__init__()
        self.device = "cuda"
        print("PyTorch model loader")
        #self.convert_model()
        self.load_trt()
        #ldm.modules.diffusionmodules.model.nonlinearity = silu

    def load_model_from_config(self, config, ckpt, device=torch.device("cuda"), verbose=False):
        print(f"Loading model from {ckpt}")
        _, extension = os.path.splitext(ckpt)
        map_location = "cpu"
        if extension.lower() == ".safetensors":
            pl_sd = safetensors.torch.load_file(ckpt, device=map_location)
        else:
            pl_sd = torch.load(ckpt, map_location=map_location)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = self.get_state_dict_from_checkpoint(pl_sd)
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        if device == torch.device("cuda"):
            model.cuda()
        elif device == torch.device("cpu"):
            model.cpu()
            model.cond_stage_model.device = "cpu"
        else:
            raise ValueError(f"Incorrect device name. Received: {device}")
        model.eval()
        return model

    def load_model(self, file=None, config_name=None, inpaint=False, verbose=False):
        ckpt_path = f"models/checkpoints/{file}"
        config_path = os.path.join('models/configs', config_name)
        config = OmegaConf.load(config_path)
        model_config_params = config['model']['params']
        clip_config = model_config_params['cond_stage_config']
        scale_factor = model_config_params['scale_factor']
        vae_config = model_config_params['first_stage_config']

        clip = None
        vae = None

        class WeightsLoader(torch.nn.Module):
            pass

        w = WeightsLoader()
        load_state_dict_to = []
        vae = VAE(scale_factor=scale_factor, config=vae_config)
        w.first_stage_model = vae.first_stage_model
        load_state_dict_to = [w]
        vae.first_stage_model = w.first_stage_model.half()

        clip = CLIP(config=clip_config, embedding_directory="models/embeddings")
        w.cond_stage_model = clip.cond_stage_model
        load_state_dict_to = [w]
        clip.cond_stage_model = w.cond_stage_model

        model = instantiate_from_config(config.model)
        sd = load_torch_file(ckpt_path)
        model = load_model_weights(model, sd, verbose=False, load_state_dict_to=load_state_dict_to)
        model = model.half()

        gs.models["sd"] = ModelPatcher(model)
        gs.models["sd"].model.to("cuda")
        gs.models["clip"] = clip
        gs.models["vae"] = vae
        gs.models["sd"].model.model.diffusion_model.forward = self.UNetModel_forward
        print("LOADED")
        if gs.debug:
            print(gs.models["sd"],gs.models["clip"],gs.models["vae"])
    def convert_model(self):
        #model = self.load_model_from_config(config, ckpt_path)
        file = "dreamlike-diffusion-1.0.safetensors"
        config_name = "v1-inference.yaml"
        ckpt_path = f"models/checkpoints/{file}"
        config_path = os.path.join('models/configs', config_name)
        config = OmegaConf.load(config_path)

        model = self.load_model_from_config(config=config, ckpt=ckpt_path)
        model = model.eval().to("cuda")
        model = model.eval()
        device = "cuda"
        dtype = torch.float16
        x = torch.randn(1, 4, 16, 16).to(device, dtype)
        timesteps = torch.zeros((1,)).to(device, dtype)
        cond = torch.randn(1, 77, 768).to(device, dtype)
        #model.model.diffusion_model.forward = UNetModel_forward

        with autocast(device_type='cuda', dtype=torch.float16):
            # y = shared.sd_model.model.diffusion_model(x, timesteps, cond)

            # print(y)

            # Export the model
            torch.onnx.export(model.model.diffusion_model,  # model being run
                              (x, timesteps, cond),  # model input (or a tuple for multiple inputs)
                              "unet.onnx",  # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=16,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['x', 'timesteps', 'cond'],  # the model's input names
                              output_names=['output'],  # the model's output names
                              dynamic_axes={'x': [0, 2, 3],  # variable length axes
                                            'timesteps': [0],
                                            'cond': [0, 1],
                                            'output': {0: 'batch_size'}})


    def load_trt(self):
        #if self.engine is None:
        trt.init_libnvinfer_plugins(None, "")
        self.engine = load_engine(engine_file)
        self.trtcontext = self.engine.create_execution_context()

    def UNetModel_forward(self, x, timesteps=None, context=None, *args, **kwargs):
        # return ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui(self, x, timesteps, context, *args, **kwargs)

        #global engine
        # global ctx
        #global trtcontext

        # if ctx is None:
        #    device = cuda.Device(0)  # enter your Gpu id here
        #    ctx = cuda.Context.attach()
        # ctx.pop()
        ctx.push()

        # engine = build()
        binding_mapping = {"x": x, "timesteps": timesteps, "cond": context}
        # Allocate host and device buffers
        bindings = []
        input_bindings = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.trtcontext.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            if self.engine.binding_is_input(binding):
                tensor = binding_mapping.get(binding)
                self.trtcontext.set_binding_shape(binding_idx, tensor.shape)

                tensor = tensor.to(np_to_torch.get(dtype))

                bindings.append(int(tensor.data_ptr()))
                #tensor = binding_mapping.get(binding)

                #self.trtcontext.set_binding_shape(binding_idx, tensor.shape)
                #input_image = tensor.detach().cpu().numpy().astype(dtype)
                #input_buffer = np.ascontiguousarray(input_image)
                #input_memory = cuda.mem_alloc(input_image.nbytes)
                #bindings.append(int(input_memory))
                #input_bindings.append((input_memory, input_buffer))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        for input_memory, input_buffer in input_bindings:
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        self.trtcontext.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()

        ctx.pop()

        # ctx.pop()  # very important
        # del ctx

        output_buffer = torch.asarray(output_buffer, dtype=torch.float16, device="cuda").reshape(x.shape)

        return output_buffer

        #apply_optimizations()
    def load_model_old(self, file=None, config=None, inpaint=False, verbose=False):

        if file not in gs.loaded_models["loaded"]:
            gs.loaded_models["loaded"].append(file)
            ckpt = f"models/checkpoints/{file}"
            gs.force_inpaint = False
            ckpt_print = ckpt.replace('\\', '/')
            #config, version = self.return_model_version(ckpt)
            #if 'Inpaint' in version:
            #    gs.force_inpaint = True
            #    print("Forcing Inpaint")

            config = os.path.join('models/configs', config)
            self.prev_seamless = False
            if verbose:
                print(f"Loading model from {ckpt} with config {config}")
            config = OmegaConf.load(config)

            # print(config.model['params'])

            if 'num_heads' in config.model['params']['unet_config']['params']:
                gs.model_version = '1.5'
            elif 'num_head_channels' in config.model['params']['unet_config']['params']:
                gs.model_version = '2.0'
            if config.model['params']['conditioning_key'] == 'hybrid-adm':
                gs.model_version = '2.0'
            if 'parameterization' in config.model['params']:
                gs.model_resolution = 768
            else:
                gs.model_resolution = 512
            print(f'v {gs.model_version} found with resolution {gs.model_resolution}')
            if verbose:
                print('gs.model_version', gs.model_version)
            checkpoint_file = ckpt
            _, extension = os.path.splitext(checkpoint_file)
            map_location = "cpu"
            if extension.lower() == ".safetensors":
                pl_sd = safetensors.torch.load_file(checkpoint_file, device=map_location)
            else:
                pl_sd = torch.load(checkpoint_file, map_location=map_location)
            if "global_step" in pl_sd:
                print(f"Global Step: {pl_sd['global_step']}")
            sd = self.get_state_dict_from_checkpoint(pl_sd)
            model = instantiate_from_config(config.model)
            m, u = model.load_state_dict(sd, strict=False)

            k = list(sd.keys())
            for x in k:
                # print(x)
                if x.startswith("cond_stage_model.transformer.") and not x.startswith(
                        "cond_stage_model.transformer.text_model."):
                    y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
                    sd[y] = sd.pop(x)

            if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
                ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
                if ids.dtype == torch.float32:
                    sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

            keys_to_replace = {
                "cond_stage_model.model.positional_embedding": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
                "cond_stage_model.model.token_embedding.weight": "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
                "cond_stage_model.model.ln_final.weight": "cond_stage_model.transformer.text_model.final_layer_norm.weight",
                "cond_stage_model.model.ln_final.bias": "cond_stage_model.transformer.text_model.final_layer_norm.bias",
            }

            for x in keys_to_replace:
                if x in sd:
                    sd[keys_to_replace[x]] = sd.pop(x)

            resblock_to_replace = {
                "ln_1": "layer_norm1",
                "ln_2": "layer_norm2",
                "mlp.c_fc": "mlp.fc1",
                "mlp.c_proj": "mlp.fc2",
                "attn.out_proj": "self_attn.out_proj",
            }

            for resblock in range(24):
                for x in resblock_to_replace:
                    for y in ["weight", "bias"]:
                        k = "cond_stage_model.model.transformer.resblocks.{}.{}.{}".format(resblock, x, y)
                        k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock,
                                                                                                        resblock_to_replace[
                                                                                                            x], y)
                        if k in sd:
                            sd[k_to] = sd.pop(k)

                for y in ["weight", "bias"]:
                    k_from = "cond_stage_model.model.transformer.resblocks.{}.attn.in_proj_{}".format(resblock, y)
                    if k_from in sd:
                        weights = sd.pop(k_from)
                        for x in range(3):
                            p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                            k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock,
                                                                                                            p[x], y)
                            sd[k_to] = weights[1024 * x:1024 * (x + 1)]

            for x in []:
                x.load_state_dict(sd, strict=False)

            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
            model.half()

            model = ModelPatcher(model)

            value = "sd" if inpaint == False else "inpaint"

            gs.models[value] = model
            #gs.models["sd"].cond_stage_model.device = self.device
            for m in gs.models[value].model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m._orig_padding_mode = m.padding_mode

            autoencoder_version = self.get_autoencoder_version()

            gs.models[value].linear_decode = make_linear_decode(autoencoder_version, self.device)
            del pl_sd
            del sd
            del m, u
            del model
            torch_gc()

            #if gs.model_version == '1.5' and not 'Inpaint' in version:
            #    self.run_post_load_model_generation_specifics()

            gs.models[value].model.eval()

            # todo make this 'cuda' a parameter
            gs.models[value].model.to(self.device)

        return ckpt
    def return_model_version(self, model):
        print('calculating sha to estimate the model version')
        with open(model, 'rb') as file:
            # Read the contents of the file
            file_contents = file.read()
            # Calculate the SHA-256 hash
            sha256_hash = hashlib.sha256(file_contents).hexdigest()
            if sha256_hash == 'd635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824':
                version = '2.0 512'
                config = 'v2-inference.yaml'
            elif sha256_hash == '2a208a7ded5d42dcb0c0ec908b23c631002091e06afe7e76d16cd11079f8d4e3':
                version = '2.0 Inpaint'
                config = 'v2-inpainting-inference.yaml'
            elif sha256_hash == 'bfcaf0755797b0c30eb00a3787e8b423eb1f5decd8de76c4d824ac2dd27e139f':
                version = '2.0 768'
                config = 'v2-inference.yaml'
            elif sha256_hash == 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556':
                version = '1.4'
                config = 'v1-inference_fp16.yaml'
            elif sha256_hash == 'c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19':
                version = '1.5 Inpaint'
                config = 'v1-inpainting-inference.yaml'
            elif sha256_hash == 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516':
                version = '1.5 EMA Only'
                config = 'v1-inference_fp16.yaml'
            elif sha256_hash == '88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36':
                version = '2.1 512'
                config = 'v2-inference.yaml'
            elif sha256_hash == 'ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0':
                version = '2.1 768'
                config = 'v2-inference-v.yaml'
            else:
                version = 'unknown'
                config = 'v1-inference_fp16.yaml'
        del file
        del file_contents
        return config, version
    def get_state_dict_from_checkpoint(self, pl_sd):
        pl_sd = pl_sd.pop("state_dict", pl_sd)
        pl_sd.pop("state_dict", None)

        sd = {}
        for k, v in pl_sd.items():
            new_key = self.transform_checkpoint_dict_key(k)

            if new_key is not None:
                sd[new_key] = v

        pl_sd.clear()
        pl_sd.update(sd)
        sd = None
        return pl_sd
    def get_autoencoder_version(self):
        return "sd-v1"  # TODO this will be different for different models

    def transform_checkpoint_dict_key(self, k):
        chckpoint_dict_replacements = {
            'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
            'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
            'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
        }
        for text, replacement in chckpoint_dict_replacements.items():
            if k.startswith(text):
                k = replacement + k[len(text):]

        return k
    def load_inpaint_model(self, modelname):

        """if inpaint in model name: force inpaint
        else
        try load normal
        except error
            load inpaint"""



        """if "sd" in gs.models:
            gs.models["sd"].to('cpu')
            del gs.models["sd"]
            torch_gc()
        if "custom_model_name" in gs.models:
            del gs.models["custom_model_name"]
            torch_gc()"""
        """Load and initialize the model from configuration variables passed at object creation time"""
        if "inpaint" not in gs.models:
            weights = modelname
            config = 'models/configs/v1-inpainting-inference.yaml'
            embedding_path = None

            config = OmegaConf.load(config)

            model = instantiate_from_config(config.model)

            model.load_state_dict(torch.load(weights)["state_dict"], strict=False)

            device = self.device
            gs.models["inpaint"] = model.half().to(device)
            del model
            return

    def load_vae(self, file):
        path = os.path.join('models/vae', file)
        print("Loading", path)
        #gs.models["sd"].first_stage_model.cpu()
        gs.models["sd"].first_stage_model = None
        gs.models["sd"].first_stage_model = VAE(ckpt_path=path)
        print("VAE Loaded", file)
    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=False, enable_all_tactics=False, timing_cache=None, workspace_size=0):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        config_kwargs['preview_features'] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs['preview_features'].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        if workspace_size > 0:
            config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []

        engine = engine_from_network(
            network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]),
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )
        save_engine(engine, path=self.engine_path)

# Decodes the image without passing through the upscaler. The resulting image will be the same size as the latent
# Thanks to Kevin Turner (https://github.com/keturn) we have a shortcut to look at the decoded image!
def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode


class VAE:
    def __init__(self, ckpt_path=None, scale_factor=0.18215, device="cuda", config=None):
        if config is None:
            #default SD1.x/SD2.x VAE parameters
            ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
            self.first_stage_model = AutoencoderKL(ddconfig, {'target': 'torch.nn.Identity'}, 4, monitor="val/rec_loss", ckpt_path=ckpt_path)
        else:
            self.first_stage_model = AutoencoderKL(**(config['params']), ckpt_path=ckpt_path)
        self.first_stage_model = self.first_stage_model.eval()
        self.scale_factor = scale_factor
        self.device = device

    def decode(self, samples):
        #model_management.unload_model()
        self.first_stage_model = self.first_stage_model.to(self.device)
        samples = samples.to(self.device)
        pixel_samples = self.first_stage_model.decode(1. / self.scale_factor * samples)
        pixel_samples = torch.clamp((pixel_samples + 1.0) / 2.0, min=0.0, max=1.0)
        self.first_stage_model = self.first_stage_model.cpu()
        pixel_samples = pixel_samples.cpu().movedim(1,-1)
        return pixel_samples

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap = 8):
        #model_management.unload_model()
        output = torch.empty((samples.shape[0], 3, samples.shape[2] * 8, samples.shape[3] * 8), device="cpu")
        #self.first_stage_model = self.first_stage_model.to(self.device)
        for b in range(samples.shape[0]):
            s = samples[b:b+1]
            out = torch.zeros((s.shape[0], 3, s.shape[2] * 8, s.shape[3] * 8), device="cpu")
            out_div = torch.zeros((s.shape[0], 3, s.shape[2] * 8, s.shape[3] * 8), device="cpu")
            for y in range(0, s.shape[2], tile_y - overlap):
                for x in range(0, s.shape[3], tile_x - overlap):
                    s_in = s[:,:,y:y+tile_y,x:x+tile_x]

                    pixel_samples = self.first_stage_model.decode(1. / self.scale_factor * s_in.to(self.device))
                    pixel_samples = torch.clamp((pixel_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    ps = pixel_samples.cpu()
                    mask = torch.ones_like(ps)
                    feather = overlap * 8
                    for t in range(feather):
                            mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))
                            mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                            mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                            mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                    out[:,:,y*8:(y+tile_y)*8,x*8:(x+tile_x)*8] += ps * mask
                    out_div[:,:,y*8:(y+tile_y)*8,x*8:(x+tile_x)*8] += mask

            output[b:b+1] = out/out_div
        #self.first_stage_model = self.first_stage_model.cpu()
        return output.movedim(1,-1)

    def encode(self, pixel_samples):
        pixel_samples = pixel_samples.cuda()
        #self.first_stage_model = self.first_stage_model.to(self.device)
        #pixel_samples = pixel_samples.movedim(-1,1).to(self.device)
        samples = self.first_stage_model.encode(2. * pixel_samples - 1.).sample() * self.scale_factor
        pixel_samples = pixel_samples.detach().cpu()
        #self.first_stage_model = self.first_stage_model.cpu()
        samples = samples.detach().cpu()

        del pixel_samples

        torch_gc()

        return samples


class CLIP:
    def __init__(self, config={}, embedding_directory=None, no_init=False):
        if no_init:
            return
        self.target_clip = config["target"]
        if "params" in config:
            params = config["params"]
        else:
            params = {}

        if self.target_clip == "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder":
            clip = SD2ClipModel
            tokenizer = SD2Tokenizer
        elif self.target_clip == "ldm.modules.encoders.modules.FrozenCLIPEmbedder":
            clip = SD1ClipModel
            tokenizer = SD1Tokenizer

        self.cond_stage_model = clip(**(params))
        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ModelPatcher(self.cond_stage_model)
        self.layer_idx = -1

    def clone(self):
        n = CLIP(no_init=True)
        n.target_clip = self.target_clip
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def load_from_state_dict(self, sd):
        self.cond_stage_model.transformer.load_state_dict(sd, strict=False)

    def add_patches(self, patches, strength=1.0):
        return self.patcher.add_patches(patches, strength)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def encode(self, text):
        self.cond_stage_model.clip_layer(self.layer_idx)
        tokens = self.tokenizer.tokenize_with_weights(text)
        try:
            self.patcher.patch_model()
            cond = self.cond_stage_model.encode_token_weights(tokens)
            self.patcher.unpatch_model()
        except Exception as e:
            self.patcher.unpatch_model()
            raise e
        return cond


import os

from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig
import torch

class ClipTokenWeightEncoderSD1:
    def encode_token_weights(self, token_weight_pairs):
        z_empty = self.encode(self.empty_tokens)
        output = []
        for x in token_weight_pairs:
            tokens = [list(map(lambda a: a[0], x))]
            z = self.encode(tokens)
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = x[j][1]
                    z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
            output += [z]
        if (len(output) == 0):
            return self.encode(self.empty_tokens)
        return torch.cat(output, dim=-2)

class SD1ClipModel(torch.nn.Module, ClipTokenWeightEncoderSD1):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None, textmodel_json_config=None, textmodel_path=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        if textmodel_path is not None:
            self.transformer = CLIPTextModel.from_pretrained(textmodel_path)
        else:
            if textmodel_json_config is None:
                textmodel_json_config = os.path.join("models/configs/sd1_clip_config.json")
            config = CLIPTextConfig.from_json_file(textmodel_json_config)
            self.transformer = CLIPTextModel(config)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.empty_tokens = [[49406] + [49407] * 76]
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) <= 12
            self.clip_layer(layer_idx)

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        if abs(layer_idx) >= 12:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0]
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    tokens_temp += [y]
                else:
                    embedding_weights += [y]
                    tokens_temp += [next_new_token]
                    next_new_token += 1
            out_tokens += [tokens_temp]

        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token, current_embeds.weight.shape[1])
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:]
            n = token_dict_size
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            self.transformer.set_input_embeddings(new_embedding)
        return out_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
            z = self.transformer.text_model.final_layer_norm(z)

        return z

    def encode(self, tokens):
        return self(tokens)

def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result

def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out

def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text

def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text

def load_embed(embedding_name, embedding_directory):
    embed_path = os.path.join(embedding_directory, embedding_name)
    if not os.path.isfile(embed_path):
        extensions = ['.safetensors', '.pt', '.bin']
        valid_file = None
        for x in extensions:
            t = embed_path + x
            if os.path.isfile(t):
                valid_file = t
                break
        if valid_file is None:
            return None
        else:
            embed_path = valid_file

    if embed_path.lower().endswith(".safetensors"):
        import safetensors.torch
        embed = safetensors.torch.load_file(embed_path, device="cpu")
    else:
        if 'weights_only' in torch.load.__code__.co_varnames:
            embed = torch.load(embed_path, weights_only=True, map_location="cpu")
        else:
            embed = torch.load(embed_path, map_location="cpu")
    if 'string_to_param' in embed:
        values = embed['string_to_param'].values()
    else:
        values = embed.values()
    return next(iter(values))

class SD1Tokenizer:
    def __init__(self, tokenizer_path=None, max_length=77, pad_with_end=True, embedding_directory=None):
        if tokenizer_path is None:
            tokenizer_path = os.path.join("models/configs/sd1_tokenizer")
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.max_tokens_per_section = self.max_length - 2

        empty = self.tokenizer('')["input_ids"]
        self.start_token = empty[0]
        self.end_token = empty[1]
        self.pad_with_end = pad_with_end
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8

    def tokenize_with_weights(self, text):
        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        tokens = []
        for t in parsed_weights:
            to_tokenize = unescape_important(t[0]).replace("\n", " ").split(' ')
            while len(to_tokenize) > 0:
                word = to_tokenize.pop(0)
                temp_tokens = []
                embedding_identifier = "embedding:"
                if word.startswith(embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(embedding_identifier):].strip('\n')
                    embed = load_embed(embedding_name, self.embedding_directory)
                    if embed is None:
                        stripped = embedding_name.strip(',')
                        if len(stripped) < len(embedding_name):
                            embed = load_embed(stripped, self.embedding_directory)
                            if embed is not None:
                                to_tokenize.insert(0, embedding_name[len(stripped):])

                    if embed is not None:
                        if len(embed.shape) == 1:
                            temp_tokens += [(embed, t[1])]
                        else:
                            for x in range(embed.shape[0]):
                                temp_tokens += [(embed[x], t[1])]
                    else:
                        print("warning, embedding:{} does not exist, ignoring".format(embedding_name))
                elif len(word) > 0:
                    tt = self.tokenizer(word)["input_ids"][1:-1]
                    for x in tt:
                        temp_tokens += [(x, t[1])]
                tokens_left = self.max_tokens_per_section - (len(tokens) % self.max_tokens_per_section)

                #try not to split words in different sections
                if tokens_left < len(temp_tokens) and len(temp_tokens) < (self.max_word_length):
                    for x in range(tokens_left):
                        tokens += [(self.end_token, 1.0)]
                tokens += temp_tokens

        out_tokens = []
        for x in range(0, len(tokens), self.max_tokens_per_section):
            o_token = [(self.start_token, 1.0)] + tokens[x:min(self.max_tokens_per_section + x, len(tokens))]
            o_token += [(self.end_token, 1.0)]
            if self.pad_with_end:
                o_token +=[(self.end_token, 1.0)] * (self.max_length - len(o_token))
            else:
                o_token +=[(0, 1.0)] * (self.max_length - len(o_token))

            out_tokens += [o_token]

        return out_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))


class SD2ClipModel(SD1ClipModel):
    def __init__(self, arch="ViT-H-14", device="cpu", max_length=77, freeze=True, layer="penultimate", layer_idx=None):
        textmodel_json_config = os.path.join("models/configs/sd2_clip_config.json")
        super().__init__(device=device, freeze=freeze, textmodel_json_config=textmodel_json_config)
        self.empty_tokens = [[49406] + [49407] + [0] * 75]
        if layer == "last":
            pass
        elif layer == "penultimate":
            layer_idx = -1
            self.clip_layer(layer_idx)
        elif self.layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < 24
            self.clip_layer(layer_idx)
        else:
            raise NotImplementedError()

    def clip_layer(self, layer_idx):
        if layer_idx < 0:
            layer_idx -= 1 #The real last layer of SD2.x clip is the penultimate one. The last one might contain garbage.
        if abs(layer_idx) >= 24:
            self.layer = "hidden"
            self.layer_idx = -2
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

class SD2Tokenizer(SD1Tokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None):
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory)


def load_torch_file(ckpt):
    if ckpt.lower().endswith(".safetensors"):
        import safetensors.torch
        sd = safetensors.torch.load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd

def load_model_weights(model, sd, verbose=False, load_state_dict_to=[]):
    m, u = model.load_state_dict(sd, strict=False)

    k = list(sd.keys())
    for x in k:
        # print(x)
        if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
            y = x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")
            sd[y] = sd.pop(x)

    if 'cond_stage_model.transformer.text_model.embeddings.position_ids' in sd:
        ids = sd['cond_stage_model.transformer.text_model.embeddings.position_ids']
        if ids.dtype == torch.float32:
            sd['cond_stage_model.transformer.text_model.embeddings.position_ids'] = ids.round()

    keys_to_replace = {
        "cond_stage_model.model.positional_embedding": "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
        "cond_stage_model.model.token_embedding.weight": "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
        "cond_stage_model.model.ln_final.weight": "cond_stage_model.transformer.text_model.final_layer_norm.weight",
        "cond_stage_model.model.ln_final.bias": "cond_stage_model.transformer.text_model.final_layer_norm.bias",
    }

    for x in keys_to_replace:
        if x in sd:
            sd[keys_to_replace[x]] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(24):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "cond_stage_model.model.transformer.resblocks.{}.{}.{}".format(resblock, x, y)
                k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "cond_stage_model.model.transformer.resblocks.{}.attn.in_proj_{}".format(resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "cond_stage_model.transformer.text_model.encoder.layers.{}.{}.{}".format(resblock, p[x], y)
                    sd[k_to] = weights[1024*x:1024*(x + 1)]

    for x in load_state_dict_to:
        x.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

import sys
sys.path.extend("C:/trt86")
sys.path.extend("C:/trt86/lib")
sys.path.extend("C:/trt86/bin")



def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



#device = cuda.Device(0)  # enter your Gpu id here
#ctx = device.make_context()
# if ctx is None:
#    device = cuda.Device(0)  # enter your Gpu id here

def build():
    #ctx.pop()
    #
    engine = load_engine(engine_file)
    return engine



