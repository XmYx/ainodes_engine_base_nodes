import copy

import torch

from .controlnet_loader import load_torch_file
from ainodes_frontend import singleton as gs

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}

LORA_UNET_MAP = {
    "proj_in": "proj_in",
    "proj_out": "proj_out",
    "transformer_blocks.0.attn1.to_q": "transformer_blocks_0_attn1_to_q",
    "transformer_blocks.0.attn1.to_k": "transformer_blocks_0_attn1_to_k",
    "transformer_blocks.0.attn1.to_v": "transformer_blocks_0_attn1_to_v",
    "transformer_blocks.0.attn1.to_out.0": "transformer_blocks_0_attn1_to_out_0",
    "transformer_blocks.0.attn2.to_q": "transformer_blocks_0_attn2_to_q",
    "transformer_blocks.0.attn2.to_k": "transformer_blocks_0_attn2_to_k",
    "transformer_blocks.0.attn2.to_v": "transformer_blocks_0_attn2_to_v",
    "transformer_blocks.0.attn2.to_out.0": "transformer_blocks_0_attn2_to_out_0",
    "transformer_blocks.0.ff.net.0.proj": "transformer_blocks_0_ff_net_0_proj",
    "transformer_blocks.0.ff.net.2": "transformer_blocks_0_ff_net_2",
}

class ModelPatcher:
    def __init__(self, model):
        self.model = model
        self.patches = []
        self.backup = {}
        self.model_options = {"transformer_options": {}}

    def clone(self):
        n = ModelPatcher(self.model)
        n.patches = self.patches[:]
        n.model_options = copy.deepcopy(self.model_options)
        return n

    def set_model_tomesd(self, ratio):
        self.model_options["transformer_options"]["tomesd"] = {"ratio": ratio}

    def set_model_sampler_cfg_function(self, sampler_cfg_function):
        self.model_options["sampler_cfg_function"] = sampler_cfg_function

    def set_model_patch(self, patch, name):
        if "patches" not in self.model_options["transformer_options"]:
            self.model_options["transformer_options"]["patches"] = {}
        self.model_options["transformer_options"]["patches"][name] = self.model_options["transformer_options"]["patches"].get(name, []) + [patch]

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def model_patches_to(self, device):
        #self.model_options["transformer_options"]


        if "patches" in self.model_options["transformer_options"]:
            patches = self.model_options["transformer_options"]["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)

    def model_dtype(self):
        return self.model.diffusion_model.dtype

    def add_patches(self, patches, strength=1.0):
        p = {}
        model_sd = self.model.state_dict()
        for k in patches:
            if k in model_sd:
                p[k] = patches[k]
        self.patches += [(strength, p)]
        return p.keys()

    def patch_model(self):
        model_sd = self.model.state_dict()
        for p in self.patches:
            for k in p[1]:
                v = p[1][k]
                key = k
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", k)
                    continue

                weight = model_sd[key]
                if key not in self.backup:
                    self.backup[key] = weight.clone()

                alpha = p[0]

                if len(v) == 4:  # lora/locon
                    mat1 = v[0]
                    mat2 = v[1]
                    if v[2] is not None:
                        alpha *= v[2] / mat2.shape[0]
                    if v[3] is not None:
                        # locon mid weights, hopefully the math is fine because I didn't properly test it
                        final_shape = [mat2.shape[1], mat2.shape[0], v[3].shape[2], v[3].shape[3]]
                        mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1).float(),
                                        v[3].transpose(0, 1).flatten(start_dim=1).float()).reshape(
                            final_shape).transpose(0, 1)
                    weight += (alpha * torch.mm(mat1.flatten(start_dim=1).float(),
                                                mat2.flatten(start_dim=1).float())).reshape(weight.shape).type(
                        weight.dtype).to(weight.device)
                elif len(v) == 8:  # lokr
                    w1 = v[0]
                    w2 = v[1]
                    w1_a = v[3]
                    w1_b = v[4]
                    w2_a = v[5]
                    w2_b = v[6]
                    t2 = v[7]
                    dim = None

                    if w1 is None:
                        dim = w1_b.shape[0]
                        w1 = torch.mm(w1_a.float(), w1_b.float())

                    if w2 is None:
                        dim = w2_b.shape[0]
                        if t2 is None:
                            w2 = torch.mm(w2_a.float(), w2_b.float())
                        else:
                            w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2.float(), w2_b.float(),
                                              w2_a.float())

                    if len(w2.shape) == 4:
                        w1 = w1.unsqueeze(2).unsqueeze(2)
                    if v[2] is not None and dim is not None:
                        alpha *= v[2] / dim

                    weight += alpha * torch.kron(w1.float(), w2.float()).reshape(weight.shape).type(
                        weight.dtype).to(weight.device)
                else:  # loha
                    w1a = v[0]
                    w1b = v[1]
                    if v[2] is not None:
                        alpha *= v[2] / w1b.shape[0]
                    w2a = v[3]
                    w2b = v[4]
                    if v[5] is not None:  # cp decomposition
                        t1 = v[5]
                        t2 = v[6]
                        m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1.float(), w1b.float(), w1a.float())
                        m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2.float(), w2b.float(), w2a.float())
                    else:
                        m1 = torch.mm(w1a.float(), w1b.float())
                        m2 = torch.mm(w2a.float(), w2b.float())

                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype).to(weight.device)
        return self.model

    def unpatch_model(self):

        for key, value in self.model_options.items():
            self.model_options[key] = None

        model_sd = self.model.state_dict()
        keys = list(self.backup.keys())
        for k in keys:
            model_sd[k][:] = self.backup[k]
            del self.backup[k]

        self.backup = {}
        self.model_options = {"transformer_options": {}}

"""def load_lora_for_models(lora_path, strength_model, strength_clip):
    key_map = model_lora_keys(gs.models["sd"].model)
    #key_map = model_lora_keys(gs.models["sd"].cond_stage_model, key_map)
    loaded = load_lora(lora_path, key_map)

    model = gs.models["sd"].clone()
    k = model.add_patches(loaded, strength_model)

    k1 = model.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED", x)
        else:
            print("LOADED")
    gs.models["sd"].model = model.model"""

def load_lora_for_models(lora_path, strength_model, strength_clip):
    key_map = model_lora_keys(gs.models["sd"].model)

    key_map = model_lora_keys(gs.models["clip"].cond_stage_model, key_map)

    loaded = load_lora(lora_path, key_map)
    new_modelpatcher = gs.models["sd"].clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)
    new_clip = gs.models["clip"].clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED", x)


    gs.models["sd"] = new_modelpatcher
    gs.models["clip"] = new_clip
    print("done")



def model_lora_keys(model, key_map={}):
    sdk = model.state_dict().keys()

    counter = 0
    for b in range(12):
        tk = "model.diffusion_model.input_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_attentions_{}_{}".format(counter // 2, counter % 2, LORA_UNET_MAP[c])
                key_map[lora_key] = k
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    for c in LORA_UNET_MAP:
        k = "model.diffusion_model.middle_block.1.{}.weight".format(c)
        if k in sdk:
            lora_key = "lora_unet_mid_block_attentions_0_{}".format(LORA_UNET_MAP[c])
            key_map[lora_key] = k
    counter = 3
    for b in range(12):
        tk = "model.diffusion_model.output_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_attentions_{}_{}".format(counter // 3, counter % 3, LORA_UNET_MAP[c])
                key_map[lora_key] = k
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    counter = 0
    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    for b in range(24):
        for c in LORA_CLIP_MAP:
            k = "transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
    return key_map

def load_lora(path, to_load):
    lora = load_torch_file(path)
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        A_name = "{}.lora_up.weight".format(x)
        B_name = "{}.lora_down.weight".format(x)
        alpha_name = "{}.alpha".format(x)
        if A_name in lora.keys():
            alpha = None
            if alpha_name in lora.keys():
                alpha = lora[alpha_name].item()
                loaded_keys.add(alpha_name)
            patch_dict[to_load[x]] = (lora[A_name], lora[B_name], alpha)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
    for x in lora.keys():
        if x not in loaded_keys:
            print("lora key not loaded", x)
        else:
            pass
            #print("Lora loaded")
    return patch_dict