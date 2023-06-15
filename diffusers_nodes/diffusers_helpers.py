from typing import Union, Optional, Dict, Any, Tuple, List

import torch

from diffusers.models.controlnet import ControlNetOutput


def multiForward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    controlnet_cond: List[torch.tensor],
    conditioning_scale: List[float],
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guess_mode: bool = False,
    return_dict: bool = True,
) -> Union[ControlNetOutput, Tuple]:

    mid_block_res_sample = None
    down_block_res_samples = None

    for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):

        percentage = 100 - (int(timestep) / 10)
        if hasattr(controlnet, "start_control"):
            start = controlnet.start_control
        else:
            start = 0
        if hasattr(controlnet, "stop_control"):
            stop = controlnet.stop_control
        else:
            stop = 100

        if start < percentage < stop:
            print("DOING CNET", percentage)
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                guess_mode,
                return_dict,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                if down_block_res_samples is not None:
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                    ]
                else:
                    down_block_res_samples = down_samples
                if mid_block_res_sample is not None:
                    mid_block_res_sample += mid_sample
                else:
                    mid_block_res_sample = mid_sample

    return down_block_res_samples, mid_block_res_sample
