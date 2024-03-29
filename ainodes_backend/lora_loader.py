import copy

import torch

from . import torch_gc
from .controlnet_loader import load_torch_file
from ainodes_frontend import singleton as gs

def transformers_convert(sd, prefix_from, prefix_to, number):
    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}.transformer.resblocks.{}.{}.{}".format(prefix_from, resblock, x, y)
                k_to = "{}.encoder.layers.{}.{}.{}".format(prefix_to, resblock, resblock_to_replace[x], y)
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}.transformer.resblocks.{}.attn.in_proj_{}".format(prefix_from, resblock, y)
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}.encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from*x:shape_from*(x + 1)]
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

    sd = transformers_convert(sd, "cond_stage_model.model", "cond_stage_model.transformer.text_model", 24)

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

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}

LORA_UNET_MAP_ATTENTIONS = {
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

LORA_UNET_MAP_RESNET = {
    "in_layers.2": "resnets_{}_conv1",
    "emb_layers.1": "resnets_{}_time_emb_proj",
    "out_layers.3": "resnets_{}_conv2",
    "skip_connection": "resnets_{}_conv_shortcut"
}

def load_lora(path, to_load):
    lora = load_torch_file(path)
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        A_name = "{}.lora_up.weight".format(x)
        B_name = "{}.lora_down.weight".format(x)
        mid_name = "{}.lora_mid.weight".format(x)

        if A_name in lora.keys():
            mid = None
            if mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = (lora[A_name], lora[B_name], alpha, mid)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)


        ######## loha
        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            patch_dict[to_load[x]] = (lora[hada_w1_a_name], lora[hada_w1_b_name], alpha, lora[hada_w2_a_name], lora[hada_w2_b_name], hada_t1, hada_t2)
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)


        ######## lokr
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (lokr_w1 is not None) or (lokr_w2 is not None) or (lokr_w1_a is not None) or (lokr_w2_a is not None):
            patch_dict[to_load[x]] = (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2)

    for x in lora.keys():
        if x not in loaded_keys:
            print("lora key not loaded", x)
    return patch_dict

def model_lora_keys(model, key_map={}):
    sdk = model.state_dict().keys()

    counter = 0
    for b in range(12):
        tk = "model.diffusion_model.input_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP_ATTENTIONS:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_attentions_{}_{}".format(counter // 2, counter % 2, LORA_UNET_MAP_ATTENTIONS[c])
                key_map[lora_key] = k
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    for c in LORA_UNET_MAP_ATTENTIONS:
        k = "model.diffusion_model.middle_block.1.{}.weight".format(c)
        if k in sdk:
            lora_key = "lora_unet_mid_block_attentions_0_{}".format(LORA_UNET_MAP_ATTENTIONS[c])
            key_map[lora_key] = k
    counter = 3
    for b in range(12):
        tk = "model.diffusion_model.output_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP_ATTENTIONS:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_attentions_{}_{}".format(counter // 3, counter % 3, LORA_UNET_MAP_ATTENTIONS[c])
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


    #Locon stuff
    ds_counter = 0
    counter = 0
    for b in range(12):
        tk = "model.diffusion_model.input_blocks.{}.0".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_{}".format(counter // 2, LORA_UNET_MAP_RESNET[c].format(counter % 2))
                key_map[lora_key] = k
                key_in = True
        for bb in range(3):
            k = "{}.{}.op.weight".format(tk[:-2], bb)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_downsamplers_0_conv".format(ds_counter)
                key_map[lora_key] = k
                ds_counter += 1
        if key_in:
            counter += 1

    counter = 0
    for b in range(3):
        tk = "model.diffusion_model.middle_block.{}".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_mid_block_{}".format(LORA_UNET_MAP_RESNET[c].format(counter))
                key_map[lora_key] = k
                key_in = True
        if key_in:
            counter += 1

    counter = 0
    us_counter = 0
    for b in range(12):
        tk = "model.diffusion_model.output_blocks.{}.0".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_{}".format(counter // 3, LORA_UNET_MAP_RESNET[c].format(counter % 3))
                key_map[lora_key] = k
                key_in = True
        for bb in range(3):
            k = "{}.{}.conv.weight".format(tk[:-2], bb)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_upsamplers_0_conv".format(us_counter)
                key_map[lora_key] = k
                us_counter += 1
        if key_in:
            counter += 1

    return key_map
class ModelPatcher:
    def __init__(self, model, size=0):
        self.size = size
        self.model = model
        self.patches = []
        self.backup = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        size = 0
        for k in model_sd:
            t = model_sd[k]
            size += t.nelement() * t.element_size()
        self.size = size
        return size

    def clone(self):
        n = ModelPatcher(self.model, self.size)
        n.patches = self.patches[:]
        n.model_options = copy.deepcopy(self.model_options)
        return n

    def set_model_tomesd(self, ratio):
        self.model_options["transformer_options"]["tomesd"] = {"ratio": ratio}

    def set_model_sampler_cfg_function(self, sampler_cfg_function):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "patches" in to:
            patches = to["patches"]
            for name in patches:
                patch_list = patches[name]
                for i in range(len(patch_list)):
                    if hasattr(patch_list[i], "to"):
                        patch_list[i] = patch_list[i].to(device)

    def model_dtype(self):
        return self.model.get_dtype()

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

                if len(v) == 4: #lora/locon
                    mat1 = v[0]
                    mat2 = v[1]
                    if v[2] is not None:
                        alpha *= v[2] / mat2.shape[0]
                    if v[3] is not None:
                        #locon mid weights, hopefully the math is fine because I didn't properly test it
                        final_shape = [mat2.shape[1], mat2.shape[0], v[3].shape[2], v[3].shape[3]]
                        mat2 = torch.mm(mat2.transpose(0, 1).flatten(start_dim=1).float(), v[3].transpose(0, 1).flatten(start_dim=1).float()).reshape(final_shape).transpose(0, 1)
                    weight += (alpha * torch.mm(mat1.flatten(start_dim=1).float(), mat2.flatten(start_dim=1).float())).reshape(weight.shape).type(weight.dtype).to(weight.device)
                elif len(v) == 8: #lokr
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
                            w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2.float(), w2_b.float(), w2_a.float())

                    if len(w2.shape) == 4:
                        w1 = w1.unsqueeze(2).unsqueeze(2)
                    if v[2] is not None and dim is not None:
                        alpha *= v[2] / dim

                    weight += alpha * torch.kron(w1.float(), w2.float()).reshape(weight.shape).type(weight.dtype).to(weight.device)
                else: #loha
                    w1a = v[0]
                    w1b = v[1]
                    if v[2] is not None:
                        alpha *= v[2] / w1b.shape[0]
                    w2a = v[3]
                    w2b = v[4]
                    if v[5] is not None: #cp decomposition
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
        model_sd = self.model.state_dict()
        keys = list(self.backup.keys())
        for k in keys:
            model_sd[k][:] = self.backup[k]
            del self.backup[k]

        self.backup = {}

class ModelPatcher_:
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



def load_lora_for_models(lora_path, strength_model, strength_clip, unet, clip):
    key_map = model_lora_keys(unet.model)
    key_map = model_lora_keys(clip.cond_stage_model, key_map)
    loaded = load_lora(lora_path, key_map)
    new_modelpatcher = unet.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)
    new_clip = clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED", x)
    del unet
    del clip
    return new_modelpatcher, new_clip
    # gs.models["sd"].model.to("cpu")
    # gs.models["clip"].cond_stage_model.to("cpu")
    # del gs.models["sd"].model
    # del gs.models["clip"].cond_stage_model
    # del gs.models["sd"]
    # del gs.models["clip"]
    torch_gc()

    # gs.models["sd"] = new_modelpatcher
    # gs.models["clip"] = new_clip
    # print("done")
    return

