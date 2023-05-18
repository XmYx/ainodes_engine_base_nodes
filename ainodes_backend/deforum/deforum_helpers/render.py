import itertools
import json
import math
import os
import tempfile

import pandas as pd
import cv2
import numpy as np
import numexpr
import gc
import random
import PIL
import time

import torch
from PIL import Image, ImageOps

from custom_nodes.ainodes_engine_base_nodes.torch_nodes.kandinsky_node import KandinskyNode
from custom_nodes.ainodes_engine_base_nodes.torch_nodes.ksampler_node import KSamplerNode
from .noise import add_noise
from .animation import anim_frame_warp
from .animation_key_frames import DeformAnimKeys, LooperAnimKeys
from .video_audio_utilities import get_frame_name, get_next_frame
from .depth import DepthModel
from .colors import maintain_colors
from .parseq_adapter import ParseqAnimKeys
from .seed import next_seed
from .image_sharpening import unsharp_mask
from .load_images import get_mask, load_img, load_image, get_mask_from_file, prepare_mask, check_mask_for_errors
from .hybrid_video import (
    hybrid_generation, hybrid_composite, get_matrix_for_hybrid_motion, get_matrix_for_hybrid_motion_prev, get_flow_for_hybrid_motion,get_flow_for_hybrid_motion_prev,
    image_transform_ransac, image_transform_optical_flow, get_flow_from_images, abs_flow_to_rel_flow, rel_flow_to_abs_flow)
from .save_images import save_image
from .composable_masks import compose_mask_with_check
from .settings import save_settings_from_animation_run
#from .deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from .subtitle_handler import init_srt_file, write_frame_subtitle, format_animation_params
from .resume import get_resume_vars
from .masks import do_overlay_mask
from .prompt import prepare_prompt, split_weighted_subprompts, check_is_number

from .RAFT import RAFT
from ... import torch_gc, pixmap_to_pil_image, pil_image_to_pixmap
from ainodes_frontend import singleton as gs

def isJson(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

# Add pairwise implementation here not to upgrade
# the whole python to 3.10 just for one function
def pairwise_repl(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def Root():
    device = "cuda"
    models_path = "models/other"
    half_precision = True
    mask_preset_names = ['everywhere', 'video_mask']
    p = None
    frames_cache = []
    raw_batch_name = None
    raw_seed = None
    initial_seed = None
    initial_info = None
    first_frame = None
    outpath_samples = ""
    animation_prompts = None
    color_corrections = None
    initial_clipskip = None
    #current_user_os = get_os()
    tmp_deforum_run_duplicated_folder = os.path.join(tempfile.gettempdir(), 'tmp_run_deforum')
    return locals()


def DeforumAnimArgs():
    animation_mode = '3D'  # ['None', '2D', '3D', 'Video Input', 'Interpolation']
    max_frames = 120
    border = 'replicate'  # ['wrap', 'replicate']
    angle = "0:(0)"
    zoom = "0:(1.0025+0.002*sin(1.25*3.14*t/30))"
    translation_x = "0:(0)"
    translation_y = "0:(0)"
    translation_z = "0:(1.75)"
    transform_center_x = "0:(0.5)"
    transform_center_y = "0:(0.5)"
    rotation_3d_x = "0:(0)"
    rotation_3d_y = "0:(0)"
    rotation_3d_z = "0:(0)"
    enable_perspective_flip = False
    perspective_flip_theta = "0:(0)"
    perspective_flip_phi = "0:(0)"
    perspective_flip_gamma = "0:(0)"
    perspective_flip_fv = "0:(53)"
    noise_schedule = "0: (0.065)"
    strength_schedule = "0: (0.65)"
    contrast_schedule = "0: (1.0)"
    cfg_scale_schedule = "0: (7)"
    enable_steps_scheduling = False
    steps_schedule = "0: (25)"
    fov_schedule = "0: (70)"
    aspect_ratio_schedule = "0: (1)"
    aspect_ratio_use_old_formula = False
    near_schedule = "0: (200)"
    far_schedule = "0: (10000)"
    seed_schedule = '0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)'
    pix2pix_img_cfg_scale = "1.5"
    pix2pix_img_cfg_scale_schedule = "0:(1.5)"
    enable_subseed_scheduling = False
    subseed_schedule = "0:(1)"
    subseed_strength_schedule = "0:(0)"
    # Sampler Scheduling
    enable_sampler_scheduling = False
    sampler_schedule = '0: ("Euler a")'
    # Composable mask scheduling
    use_noise_mask = False
    mask_schedule = '0: ("{video_mask}")'
    noise_mask_schedule = '0: ("{video_mask}")'
    # Checkpoint Scheduling
    enable_checkpoint_scheduling = False
    checkpoint_schedule = '0: ("model1.ckpt"), 100: ("model2.safetensors")'
    # CLIP skip Scheduling
    enable_clipskip_scheduling = False
    clipskip_schedule = '0: (2)'
    # Noise Multiplier Scheduling
    enable_noise_multiplier_scheduling = True
    noise_multiplier_schedule = '0: (1.05)'
    # Anti-blur
    amount_schedule = "0: (0.1)"
    kernel_schedule = "0: (5)"
    sigma_schedule = "0: (1.0)"
    threshold_schedule = "0: (0.0)"
    # Hybrid video
    hybrid_comp_alpha_schedule = "0:(0.5)"
    hybrid_comp_mask_blend_alpha_schedule = "0:(0.5)"
    hybrid_comp_mask_contrast_schedule = "0:(1)"
    hybrid_comp_mask_auto_contrast_cutoff_high_schedule = "0:(100)"
    hybrid_comp_mask_auto_contrast_cutoff_low_schedule = "0:(0)"
    hybrid_flow_factor_schedule = "0:(1)"
    # Coherence
    color_coherence = 'LAB'  # ['None', 'HSV', 'LAB', 'RGB', 'Video Input', 'Image']
    color_coherence_image_path = ""
    color_coherence_video_every_N_frames = 1
    color_force_grayscale = False
    legacy_colormatch = False
    diffusion_cadence = '2'  # ['1','2','3','4','5','6','7','8']
    optical_flow_cadence = 'None'  # ['None', 'RAFT','DIS Medium', 'DIS Fine', 'Farneback']
    cadence_flow_factor_schedule = "0: (1)"
    optical_flow_redo_generation = 'None'  # ['None', 'RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
    redo_flow_factor_schedule = "0: (1)"
    diffusion_redo = '0'
    # **Noise settings:**
    noise_type = 'perlin'  # ['uniform', 'perlin']
    # Perlin params
    perlin_w = 8
    perlin_h = 8
    perlin_octaves = 4
    perlin_persistence = 0.5
    # **3D Depth Warping:**
    use_depth_warping = True
    depth_algorithm = 'Zoe' #  'Midas-3-Hybrid'  # ['Midas+AdaBins (old)','Zoe+AdaBins (old)', 'Midas-3-Hybrid','Midas-3.1-BeitLarge', 'AdaBins', 'Zoe', 'Leres'] Midas-3.1-BeitLarge is temporarily removed 04-05-23 until fixed
    midas_weight = 0.2  # midas/ zoe weight - only relevant in old/ legacy depth_algorithm modes. see above ^
    padding_mode = 'border'  # ['border', 'reflection', 'zeros']
    sampling_mode = 'bicubic'  # ['bicubic', 'bilinear', 'nearest']
    save_depth_maps = False
    # **Video Input:**
    video_init_path = 'https://deforum.github.io/a1/V1.mp4'
    extract_nth_frame = 1
    extract_from_frame = 0
    extract_to_frame = -1  # minus 1 for unlimited frames
    overwrite_extracted_frames = True
    use_mask_video = False
    video_mask_path = 'https://deforum.github.io/a1/VM1.mp4'
    # **Hybrid Video for 2D/3D Animation Mode:**
    hybrid_generate_inputframes = False
    hybrid_generate_human_masks = "None"  # ['None','PNGs','Video', 'Both']
    hybrid_use_first_frame_as_init_image = True
    hybrid_motion = "None"  # ['None','Optical Flow','Perspective','Affine']
    hybrid_motion_use_prev_img = False
    hybrid_flow_consistency = False
    hybrid_consistency_blur = 2
    hybrid_flow_method = "RAFT"  # ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
    hybrid_composite = 'None'  # ['None', 'Normal', 'Before Motion', 'After Generation']
    hybrid_use_init_image = False
    hybrid_comp_mask_type = "None"  # ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
    hybrid_comp_mask_inverse = False
    hybrid_comp_mask_equalize = "None"  # ['None','Before','After','Both']
    hybrid_comp_mask_auto_contrast = False
    hybrid_comp_save_extra_frames = False
    # **Resume Animation:**
    resume_from_timestring = False
    resume_timestring = "20230129210106"
    enable_ddim_eta_scheduling = False
    ddim_eta_schedule = "0:(0)"
    enable_ancestral_eta_scheduling = False
    ancestral_eta_schedule = "0:(1)"

    return locals()


def DeforumAnimPrompts():
    return r"""{
    "0": "tiny cute swamp bunny, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, in focus, by tomasz alen kopera",
    "30": "anthropomorphic clean cat, surrounded by fractals, epic angle and pose, symmetrical, 3d, depth of field, ruan jia and fenghua zhong",
    "60": "a beautiful coconut --neg photo, realistic",
    "90": "a beautiful durian, trending on Artstation"
}
    """


def DeforumArgs():
    # **Image Settings**
    W = 512  #
    H = 512  #
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    show_info_on_ui = True

    # **Webui stuff**
    tiling = False
    restore_faces = False
    seed_enable_extras = False
    subseed = -1
    subseed_strength = 0
    seed_resize_from_w = 0
    seed_resize_from_h = 0

    # **Sampling Settings**
    seed = -1  #
    sampler = 'euler_ancestral'  # ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 25  #
    scale = 7  #

    dynamic_threshold = None
    static_threshold = None

    # **Save & Display Settings**
    save_settings = True
    save_sample_per_step = False

    # **Prompt Settings**
    prompt_weighting = False
    normalize_prompt_weights = True
    log_weighted_subprompts = False

    # **Batch Settings**
    n_batch = 1  #
    batch_name = "Deforum_{timestring}"
    seed_behavior = "iter"  # ["iter","fixed","random","ladder","alternate","schedule"]
    seed_iter_N = 1
    outdir = "output/deforum"

    # **Init Settings**
    use_init = False
    strength = 0.8
    strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
    init_image = "https://deforum.github.io/a1/I1.png"
    # Whiter areas of the mask are areas that change more
    use_mask = False
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    mask_file = "https://deforum.github.io/a1/M1.jpg"
    invert_mask = False
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_contrast_adjust = 1.0
    mask_brightness_adjust = 1.0
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 4

    fill = 1  # MASKARGSEXPANSION Todo : Rename and convert to same formatting as used in img2img masked content
    full_res_mask = True
    full_res_mask_padding = 4
    reroll_blank_frames = 'reroll'  # reroll, interrupt, or ignore
    reroll_patience = 10

    n_samples = 1  # doesnt do anything
    precision = 'autocast'

    prompt = ""
    timestring = ""
    init_sample = None
    mask_image = None
    noise_mask = None
    seed_internal = 0

    return locals()


def keyframeExamples():
    return '''{
    "0": "https://deforum.github.io/a1/Gi1.png",
    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",
    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",
    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",
    "max_f-20": "https://deforum.github.io/a1/Gi1.png"
}'''


def LoopArgs():
    use_looper = False
    init_images = keyframeExamples()
    image_strength_schedule = "0:(0.75)"
    blendFactorMax = "0:(0.35)"
    blendFactorSlope = "0:(0.25)"
    tweening_frames_schedule = "0:(20)"
    color_correction_factor = "0:(0.075)"
    return locals()


def ParseqArgs():
    parseq_manifest = None
    parseq_use_deltas = True
    return locals()


def DeforumOutputArgs():
    skip_video_creation = False
    fps = 15
    make_gif = False
    delete_imgs = False  # True will delete all imgs after a successful mp4 creation
    image_path = "C:/SD/20230124234916_%09d.png"
    mp4_path = "testvidmanualsettings.mp4"
    add_soundtrack = 'None'  # ["File","Init Video"]
    soundtrack_path = "https://deforum.github.io/a1/A1.mp3"
    # End-Run upscaling
    r_upscale_video = False
    r_upscale_factor = 'x2'  # ['2x', 'x3', 'x4']
    r_upscale_model = 'realesr-animevideov3'  # 'realesr-animevideov3' (default of realesrgan engine, does 2-4x), the rest do only 4x: 'realesrgan-x4plus', 'realesrgan-x4plus-anime'
    r_upscale_keep_imgs = True

    store_frames_in_ram = False
    # **Interpolate Video Settings**
    frame_interpolation_engine = "None"  # ["None", "RIFE v4.6", "FILM"]
    frame_interpolation_x_amount = 2  # [2 to 1000 depends on the engine]
    frame_interpolation_slow_mo_enabled = False
    frame_interpolation_slow_mo_amount = 2  # [2 to 10]
    frame_interpolation_keep_imgs = False
    return locals()



def render_animation(node, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root, callback=None):

    DEBUG_MODE = gs.debug
    save_as_srt = None
    lowvram = None
    medvram = None

    max_f = anim_args.max_frames

    if save_as_srt: # create .srt file and set timeframe mechanism using FPS
        srt_filename = os.path.join(args.outdir, f"{args.timestring}.srt")
        srt_frame_duration = init_srt_file(srt_filename, video_args.fps)

    if anim_args.animation_mode in ['2D','3D']:
        # handle hybrid video generation
        if anim_args.hybrid_composite != 'None' or anim_args.hybrid_motion in ['Affine', 'Perspective', 'Optical Flow']:
            args, anim_args, inputfiles = hybrid_generation(args, anim_args, root)
            # path required by hybrid functions, even if hybrid_comp_save_extra_frames is False
            hybrid_frame_path = os.path.join(args.outdir, 'hybridframes')
        # initialize prev_flow
        if anim_args.hybrid_motion == 'Optical Flow':
            prev_flow = None

        if loop_args.use_looper:
            print("Using Guided Images mode: seed_behavior will be set to 'schedule' and 'strength_0_no_init' to False")
            if args.strength == 0:
                raise RuntimeError("Strength needs to be greater than 0 in Init tab")
            args.strength_0_no_init = False
            args.seed_behavior = "schedule"
            if not isJson(loop_args.init_images):
                raise RuntimeError("The images set for use with keyframe-guidance are not in a proper JSON format")

    # handle controlnet video input frames generation
    #if is_controlnet_enabled(controlnet_args):
    #    unpack_controlnet_vids(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, animation_prompts, root)

    # use parseq if manifest is provided
    use_parseq = parseq_args.parseq_manifest != None and parseq_args.parseq_manifest.strip()
    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args, args.seed) if not use_parseq else ParseqAnimKeys(parseq_args, anim_args, video_args)
    loopSchedulesAndData = LooperAnimKeys(loop_args, anim_args, args.seed)

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to:\n{args.outdir}")
    
    # save settings.txt file for the current run
    # save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)

    # resume from timestring
    if anim_args.resume_from_timestring:
        args.timestring = anim_args.resume_timestring

    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True        

    # expand prompts out to per-frame
    if use_parseq and keys.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
        for i, prompt in animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()

    # check for video inits
    using_vid_init = anim_args.animation_mode == 'Video Input'

    # load depth model for 3D
    predict_depths = (anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
    predict_depths = predict_depths or (anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth','Video Depth'])
    if predict_depths:
        keep_in_vram = True
        
        device = ('cpu' if lowvram or medvram else root.device)
        depth_model = DepthModel(root.models_path, device, root.half_precision, keep_in_vram=keep_in_vram, depth_algorithm=anim_args.depth_algorithm, Width=args.W, Height=args.H, midas_weight=anim_args.midas_weight)
            
        # depth-based hybrid composite mask requires saved depth maps
        if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type =='Depth':
            anim_args.save_depth_maps = True
    else:
        depth_model = None
        anim_args.save_depth_maps = False
        
    raft_model = None
    load_raft = (anim_args.optical_flow_cadence == "RAFT" and int(anim_args.diffusion_cadence) > 1) or \
            (anim_args.hybrid_motion == "Optical Flow" and anim_args.hybrid_flow_method == "RAFT") or \
            (anim_args.optical_flow_redo_generation == "RAFT")
    if load_raft:
        print("Loading RAFT model...")
        raft_model = RAFT()

    # state for interpolating between diffusion steps
    turbo_steps = 1 if using_vid_init else int(anim_args.diffusion_cadence)
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    # initialize vars
    prev_img = None
    color_match_sample = None
    start_frame = 0

    # resume animation (requires at least two frames - see function)
    if anim_args.resume_from_timestring:
        # determine last frame and frame to start on
        prev_frame, next_frame, prev_img, next_img = get_resume_vars(
            folder=args.outdir,
            timestring=anim_args.resume_timestring,
            cadence=turbo_steps
        )

        # set up turbo step vars
        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = prev_img, prev_frame
            turbo_next_image, turbo_next_frame_idx = next_img, next_frame
        
        # advance start_frame to next frame
        start_frame = next_frame + 1

    args.n_samples = 1
    frame_idx = start_frame

    # reset the mask vals as they are overwritten in the compose_mask algorithm
    mask_vals = {}
    noise_mask_vals = {}

    mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)
    noise_mask_vals['everywhere'] = Image.new('1', (args.W, args.H), 1)

    mask_image = None
    
    if args.use_init and args.init_image != None and args.init_image != '':
        _, mask_image = load_img(args.init_image, 
                                        shape=(args.W, args.H),  
                                        use_alpha_as_mask=args.use_alpha_as_mask)
        mask_vals['video_mask'] = mask_image
        noise_mask_vals['video_mask'] = mask_image
    
    # Grab the first frame masks since they wont be provided until next frame    
    # Video mask overrides the init image mask, also, won't be searching for init_mask if use_mask_video is set
    # Made to solve https://github.com/deforum-art/deforum-for-automatic1111-webui/issues/386
    if anim_args.use_mask_video:

        args.mask_file = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
        args.noise_mask = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)

        mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
        noise_mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
    elif mask_image is None and args.use_mask:
        mask_vals['video_mask'] = get_mask(args)
        noise_mask_vals['video_mask'] = get_mask(args) # TODO?: add a different default noisc mask

    # get color match for 'Image' color coherence only once, before loop
    if anim_args.color_coherence == 'Image':
        color_match_sample = load_image(anim_args.color_coherence_image_path)
        color_match_sample = color_match_sample.resize((args.W, args.H), PIL.Image.LANCZOS)
        color_match_sample = cv2.cvtColor(np.array(color_match_sample), cv2.COLOR_RGB2BGR)

    #Webui
    #state.job_count = anim_args.max_frames

    frame = 0

    while frame_idx < anim_args.max_frames:
        #Webui
        
        #state.job = f"frame {frame_idx + 1}/{anim_args.max_frames}"
        frame = frame_idx + 1
        
        """if state.skipped:
            print("\n** PAUSED **")
            state.skipped = False
            while not state.skipped:
                time.sleep(0.1)
            print("** RESUMING **")"""

        print(f"\033[36mAnimation frame: \033[0m{frame_idx}/{anim_args.max_frames}  ")

        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        scale = keys.cfg_scale_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        kernel = int(keys.kernel_schedule_series[frame_idx])
        sigma = keys.sigma_schedule_series[frame_idx]
        amount = keys.amount_schedule_series[frame_idx]
        threshold = keys.threshold_schedule_series[frame_idx]
        cadence_flow_factor = keys.cadence_flow_factor_schedule_series[frame_idx]
        redo_flow_factor = keys.redo_flow_factor_schedule_series[frame_idx]
        hybrid_comp_schedules = {
            "alpha": keys.hybrid_comp_alpha_schedule_series[frame_idx],
            "mask_blend_alpha": keys.hybrid_comp_mask_blend_alpha_schedule_series[frame_idx],
            "mask_contrast": keys.hybrid_comp_mask_contrast_schedule_series[frame_idx],
            "mask_auto_contrast_cutoff_low": int(keys.hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series[frame_idx]),
            "mask_auto_contrast_cutoff_high": int(keys.hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series[frame_idx]),
            "flow_factor": keys.hybrid_flow_factor_schedule_series[frame_idx]
        }
        scheduled_sampler_name = None
        scheduled_clipskip = None
        scheduled_noise_multiplier = None
        scheduled_ddim_eta = None
        scheduled_ancestral_eta = None
        
        mask_seq = None
        noise_mask_seq = None
        if anim_args.enable_steps_scheduling and keys.steps_schedule_series[frame_idx] is not None:
            args.steps = int(keys.steps_schedule_series[frame_idx])
        if anim_args.enable_sampler_scheduling and keys.sampler_schedule_series[frame_idx] is not None:
            scheduled_sampler_name = keys.sampler_schedule_series[frame_idx].casefold()
        if anim_args.enable_clipskip_scheduling and keys.clipskip_schedule_series[frame_idx] is not None:
            scheduled_clipskip = int(keys.clipskip_schedule_series[frame_idx])
        if anim_args.enable_noise_multiplier_scheduling and keys.noise_multiplier_schedule_series[frame_idx] is not None:
            scheduled_noise_multiplier = float(keys.noise_multiplier_schedule_series[frame_idx])
        if anim_args.enable_ddim_eta_scheduling and keys.ddim_eta_schedule_series[frame_idx] is not None:
            scheduled_ddim_eta = float(keys.ddim_eta_schedule_series[frame_idx])
        if anim_args.enable_ancestral_eta_scheduling and keys.ancestral_eta_schedule_series[frame_idx] is not None:
            scheduled_ancestral_eta = float(keys.ancestral_eta_schedule_series[frame_idx])
        if args.use_mask and keys.mask_schedule_series[frame_idx] is not None:
            mask_seq = keys.mask_schedule_series[frame_idx]
        if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
            noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]
        
        if args.use_mask and not anim_args.use_noise_mask:
            noise_mask_seq = mask_seq

        depth = None

        if anim_args.animation_mode == '3D' and (lowvram or medvram):
            # Unload the main checkpoint and load the depth model
            #lowvram.send_everything_to_cpu()
            #sd_hijack.model_hijack.undo_hijack(sd_model)
            torch_gc()
            if predict_depths: depth_model.to(root.device)
        
        if turbo_steps == 1 and save_as_srt:
            params_string = format_animation_params(keys, prompt_series, frame_idx)
            write_frame_subtitle(srt_filename, frame_idx, srt_frame_duration, f"F#: {frame_idx}; Cadence: false; Seed: {args.seed}; {params_string}")
            params_string = None
            
        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(start_frame, frame_idx-turbo_steps)
            cadence_flow = None
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                # update progress during cadence
                #state.job = f"frame {tween_frame_idx + 1}/{anim_args.max_frames}"
                #state.job_no = tween_frame_idx + 1
                # cadence vars
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx

                # optical flow cadence setup before animation warping
                if anim_args.animation_mode in ['2D', '3D'] and anim_args.optical_flow_cadence != 'None':
                    if keys.strength_schedule_series[tween_frame_start_idx] > 0:
                        if cadence_flow is None and turbo_prev_image is not None and turbo_next_image is not None:
                            cadence_flow = get_flow_from_images(turbo_prev_image, turbo_next_image, anim_args.optical_flow_cadence, raft_model) / 2
                            turbo_next_image = image_transform_optical_flow(turbo_next_image, -cadence_flow, 1)

                if save_as_srt:
                    params_string = format_animation_params(keys, prompt_series, tween_frame_idx)
                    write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration, f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {args.seed}; {params_string}")
                    params_string = None

                print(f"Creating in-between {'' if cadence_flow is None else anim_args.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")

                if depth_model is not None:
                    assert(turbo_next_image is not None)
                    depth = depth_model.predict(turbo_next_image, anim_args.midas_weight, root.half_precision)
                    
                if advance_prev:
                    turbo_prev_image, _ = anim_frame_warp(turbo_prev_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)
                if advance_next:
                    turbo_next_image, _ = anim_frame_warp(turbo_next_image, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)

                # hybrid video motion - warps turbo_prev_image or turbo_next_image to match motion
                if tween_frame_idx > 0:
                    if anim_args.hybrid_motion in ['Affine', 'Perspective']:
                        if anim_args.hybrid_motion_use_prev_img:
                            matrix = get_matrix_for_hybrid_motion_prev(tween_frame_idx-1, (args.W, args.H), inputfiles, prev_img, anim_args.hybrid_motion)
                            if advance_prev:
                                turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix, anim_args.hybrid_motion)
                            if advance_next:
                                turbo_next_image = image_transform_ransac(turbo_next_image, matrix, anim_args.hybrid_motion)
                        else:
                            matrix = get_matrix_for_hybrid_motion(tween_frame_idx-1, (args.W, args.H), inputfiles, anim_args.hybrid_motion)
                            if advance_prev:
                                turbo_prev_image = image_transform_ransac(turbo_prev_image, matrix, anim_args.hybrid_motion)
                            if advance_next:
                                turbo_next_image = image_transform_ransac(turbo_next_image, matrix, anim_args.hybrid_motion)
                    if anim_args.hybrid_motion in ['Optical Flow']:
                        if anim_args.hybrid_motion_use_prev_img:
                            flow = get_flow_for_hybrid_motion_prev(tween_frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, prev_img, anim_args.hybrid_flow_method, raft_model, anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)                            
                            if advance_prev:
                                turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow, hybrid_comp_schedules['flow_factor'])
                            if advance_next:
                                turbo_next_image = image_transform_optical_flow(turbo_next_image, flow, hybrid_comp_schedules['flow_factor'])
                            prev_flow = flow
                        else:
                            flow = get_flow_for_hybrid_motion(tween_frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, anim_args.hybrid_flow_method, raft_model, anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
                            if advance_prev:
                                turbo_prev_image = image_transform_optical_flow(turbo_prev_image, flow, hybrid_comp_schedules['flow_factor'])
                            if advance_next:
                                turbo_next_image = image_transform_optical_flow(turbo_next_image, flow, hybrid_comp_schedules['flow_factor'])
                            prev_flow = flow

                # do optical flow cadence after animation warping
                if cadence_flow is not None:
                    cadence_flow = abs_flow_to_rel_flow(cadence_flow, args.W, args.H)
                    cadence_flow, _ = anim_frame_warp(cadence_flow, args, anim_args, keys, tween_frame_idx, depth_model, depth=depth, device=root.device, half_precision=root.half_precision)
                    cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, args.W, args.H) * tween
                    if advance_prev:
                        turbo_prev_image = image_transform_optical_flow(turbo_prev_image, cadence_flow_inc, cadence_flow_factor)
                    if advance_next:
                        turbo_next_image = image_transform_optical_flow(turbo_next_image, cadence_flow_inc, cadence_flow_factor)

                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image*(1.0-tween) + turbo_next_image*tween
                else:
                    img = turbo_next_image
                
                # intercept and override to grayscale
                if anim_args.color_force_grayscale:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                

                # overlay mask
                if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
                    img = do_overlay_mask(args, anim_args, img, tween_frame_idx, True)

                # get prev_img during cadence
                prev_img = img

                # current image update for cadence frames (left commented because it doesn't currently update the preview)
                # state.current_image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))

                # saving cadence frames
                filename = f"{args.timestring}_{tween_frame_idx:09}.png"

                cv2.imwrite(os.path.join(args.outdir, filename), img)

                if callback is not None:
                    callback(Image.open(os.path.join(args.outdir, filename)))

                if anim_args.save_depth_maps:
                    depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{tween_frame_idx:09}.png"), depth)

        # get color match for video outside of prev_img conditional
        hybrid_available = anim_args.hybrid_composite != 'None' or anim_args.hybrid_motion in ['Optical Flow', 'Affine', 'Perspective']
        if anim_args.color_coherence == 'Video Input' and hybrid_available:
            if int(frame_idx) % int(anim_args.color_coherence_video_every_N_frames) == 0:
                prev_vid_img = Image.open(os.path.join(args.outdir, 'inputframes', get_frame_name(anim_args.video_init_path) + f"{frame_idx:09}.jpg"))
                prev_vid_img = prev_vid_img.resize((args.W, args.H), PIL.Image.LANCZOS)
                color_match_sample = np.asarray(prev_vid_img)
                color_match_sample = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2BGR)

        # after 1st frame, prev_img exists
        if prev_img is not None:
            # apply transforms to previous frame
            prev_img, depth = anim_frame_warp(prev_img, args, anim_args, keys, frame_idx, depth_model, depth=None, device=root.device, half_precision=root.half_precision)

            # do hybrid compositing before motion
            if anim_args.hybrid_composite == 'Before Motion':
                args, prev_img = hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root)

            # hybrid video motion - warps prev_img to match motion, usually to prepare for compositing
            if anim_args.hybrid_motion in ['Affine', 'Perspective']:
                if anim_args.hybrid_motion_use_prev_img:
                    matrix = get_matrix_for_hybrid_motion_prev(frame_idx-1, (args.W, args.H), inputfiles, prev_img, anim_args.hybrid_motion)
                else:
                    matrix = get_matrix_for_hybrid_motion(frame_idx-1, (args.W, args.H), inputfiles, anim_args.hybrid_motion)
                prev_img = image_transform_ransac(prev_img, matrix, anim_args.hybrid_motion)    
            if anim_args.hybrid_motion in ['Optical Flow']:
                if anim_args.hybrid_motion_use_prev_img:
                    flow = get_flow_for_hybrid_motion_prev(frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, prev_img, anim_args.hybrid_flow_method, raft_model, anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
                else:
                    flow = get_flow_for_hybrid_motion(frame_idx-1, (args.W, args.H), inputfiles, hybrid_frame_path, prev_flow, anim_args.hybrid_flow_method, raft_model, anim_args.hybrid_flow_consistency, anim_args.hybrid_consistency_blur, anim_args.hybrid_comp_save_extra_frames)
                prev_img = image_transform_optical_flow(prev_img, flow, hybrid_comp_schedules['flow_factor'])
                prev_flow = flow

            # do hybrid compositing after motion (normal)
            if anim_args.hybrid_composite == 'Normal':
                args, prev_img = hybrid_composite(args, anim_args, frame_idx, prev_img, depth_model, hybrid_comp_schedules, root)

            # apply color matching
            if anim_args.color_coherence != 'None':
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)

            # intercept and override to grayscale
            if anim_args.color_force_grayscale:
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)

            # apply scaling
            contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # anti-blur
            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold, mask_image if args.use_mask else None)
            # apply frame noising
            if args.use_mask or anim_args.use_noise_mask:
                args.noise_mask = compose_mask_with_check(root, args, noise_mask_seq, noise_mask_vals, Image.fromarray(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
            noised_image = add_noise(contrast_image, noise, args.seed, anim_args.noise_type,
                            (anim_args.perlin_w, anim_args.perlin_h, anim_args.perlin_octaves, anim_args.perlin_persistence),
                             args.noise_mask, args.invert_mask)

            # use transformed previous frame as init for current
            args.use_init = True
            args.init_sample = Image.fromarray(cv2.cvtColor(noised_image, cv2.COLOR_BGR2RGB))
            args.strength = max(0.0, min(1.0, strength))
        
        args.scale = scale
       
        # Pix2Pix Image CFG Scale - does *nothing* with non pix2pix checkpoints
        args.pix2pix_img_cfg_scale = float(keys.pix2pix_img_cfg_scale_series[frame_idx])

        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]

        print("PROMPT SERIES", prompt_series)
      
        if args.seed_behavior == 'schedule' or use_parseq:
            args.seed = int(keys.seed_schedule_series[frame_idx])

        if anim_args.enable_checkpoint_scheduling:
            args.checkpoint = keys.checkpoint_schedule_series[frame_idx]
        else:
            args.checkpoint = None
            
        #SubSeed scheduling
        if anim_args.enable_subseed_scheduling:
            args.subseed = int(keys.subseed_schedule_series[frame_idx])
            args.subseed_strength = float(keys.subseed_strength_schedule_series[frame_idx])

        if use_parseq:
            args.seed_enable_extras = True
            anim_args.enable_subseed_scheduling = True
            args.subseed = int(keys.subseed_schedule_series[frame_idx])
            args.subseed_strength = keys.subseed_strength_schedule_series[frame_idx]

        # set value back into the prompt - prepare and report prompt and seed
        args.prompt = prepare_prompt(args.prompt, anim_args.max_frames, args.seed, frame_idx)

        # grab init image for current frame
        if using_vid_init:
            init_frame = get_next_frame(args.outdir, anim_args.video_init_path, frame_idx, False)
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame
            args.strength = max(0.0, min(1.0, strength))
        if anim_args.use_mask_video:
            args.mask_file = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)
            args.noise_mask = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)

            mask_vals['video_mask'] = get_mask_from_file(get_next_frame(args.outdir, anim_args.video_mask_path, frame_idx, True), args)

        if args.use_mask:
            args.mask_image = compose_mask_with_check(root, args, mask_seq, mask_vals, args.init_sample) if args.init_sample is not None else None # we need it only after the first frame anyway

        # setting up some arguments for the looper
        loop_args.imageStrength = loopSchedulesAndData.image_strength_schedule_series[frame_idx]
        loop_args.blendFactorMax = loopSchedulesAndData.blendFactorMax_series[frame_idx]
        loop_args.blendFactorSlope = loopSchedulesAndData.blendFactorSlope_series[frame_idx]
        loop_args.tweeningFrameSchedule = loopSchedulesAndData.tweening_frames_schedule_series[frame_idx]
        loop_args.colorCorrectionFactor = loopSchedulesAndData.color_correction_factor_series[frame_idx]
        loop_args.use_looper = loopSchedulesAndData.use_looper
        loop_args.imagesToKeyframe = loopSchedulesAndData.imagesToKeyframe

        """if 'img2img_fix_steps' in opts.data and opts.data["img2img_fix_steps"]: # disable "with img2img do exactly x steps" from general setting, as it *ruins* deforum animations
            opts.data["img2img_fix_steps"] = False
        if scheduled_clipskip is not None:
            opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip
        if scheduled_noise_multiplier is not None:
            opts.data["initial_noise_multiplier"] = scheduled_noise_multiplier
        if scheduled_ddim_eta is not None:
            opts.data["eta_ddim"] = scheduled_ddim_eta
        if scheduled_ancestral_eta is not None:
            opts.data["eta_ancestral"] = scheduled_ancestral_eta"""
        
        if anim_args.animation_mode == '3D' and (lowvram or medvram):
            if predict_depths: depth_model.to('cpu')
            torch_gc()
            #lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
            #sd_hijack.model_hijack.hijack(sd_model)

        # optical flow redo before generation
        if anim_args.optical_flow_redo_generation != 'None' and prev_img is not None and strength > 0:
            print(f"Optical flow redo is diffusing and warping using {anim_args.optical_flow_redo_generation} optical flow before generation.")
            stored_seed = args.seed
            args.seed = random.randint(0, 2**32 - 1)

            disposable_image = generate(node, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=scheduled_sampler_name)

            disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
            disposable_flow = get_flow_from_images(prev_img, disposable_image, anim_args.optical_flow_redo_generation, raft_model)
            disposable_image = cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB)
            disposable_image = image_transform_optical_flow(disposable_image, disposable_flow, redo_flow_factor)
            args.seed = stored_seed
            args.init_sample = Image.fromarray(disposable_image)
            del(disposable_image,disposable_flow,stored_seed)
            gc.collect()

        # diffusion redo
        if int(anim_args.diffusion_redo) > 0 and prev_img is not None and strength > 0:
            stored_seed = args.seed
            for n in range(0,int(anim_args.diffusion_redo)):
                print(f"Redo generation {n+1} of {int(anim_args.diffusion_redo)} before final generation")
                args.seed = random.randint(0, 2**32 - 1)
                disposable_image = generate(args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=scheduled_sampler_name)
                disposable_image = cv2.cvtColor(np.array(disposable_image), cv2.COLOR_RGB2BGR)
                # color match on last one only
                if (n == int(anim_args.diffusion_redo)):
                    disposable_image = maintain_colors(prev_img, color_match_sample, anim_args.color_coherence)                
                args.seed = stored_seed
                args.init_sample = Image.fromarray(cv2.cvtColor(disposable_image, cv2.COLOR_BGR2RGB))
            del(disposable_image, stored_seed)
            gc.collect()

        # generation
        image = generate(node, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name=scheduled_sampler_name)
        
        if image is None:
            break

        # do hybrid video after generation
        if frame_idx > 0 and anim_args.hybrid_composite == 'After Generation':
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            args, image = hybrid_composite(args, anim_args, frame_idx, image, depth_model, hybrid_comp_schedules, root)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # color matching on first frame is after generation, color match was collected earlier, so we do an extra generation to avoid the corruption introduced by the color match of first output
        if frame_idx == 0 and (anim_args.color_coherence == 'Image' or (anim_args.color_coherence == 'Video Input' and hybrid_available)):
            image = maintain_colors(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), color_match_sample, anim_args.color_coherence)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif color_match_sample is not None and anim_args.color_coherence != 'None' and not anim_args.legacy_colormatch:
            image = maintain_colors(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), color_match_sample, anim_args.color_coherence)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # intercept and override to grayscale
        if anim_args.color_force_grayscale:
            image = ImageOps.grayscale(image)
            image = ImageOps.colorize(image, black ="black", white ="white")

        # overlay mask
        if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
            image = do_overlay_mask(args, anim_args, image, frame_idx)

        # on strength 0, set color match to generation
        if ((not anim_args.legacy_colormatch and not args.use_init) or (anim_args.legacy_colormatch and strength == 0)) and not anim_args.color_coherence in ['Image', 'Video Input']:
            color_match_sample = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if not using_vid_init:
            prev_img = opencv_image

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = opencv_image, frame_idx
            frame_idx += turbo_steps
        else:    
            filename = f"{args.timestring}_{frame_idx:09}.png"
            save_image(image, 'PIL', filename, args, video_args, root)
            if callback is not None:
                callback(image)

            """if anim_args.save_depth_maps:
                if cmd_opts.lowvram or cmd_opts.medvram:
                    lowvram.send_everything_to_cpu()
                    sd_hijack.model_hijack.undo_hijack(sd_model)
                    devices.torch_gc()
                    depth_model.to(root.device)
                depth = depth_model.predict(opencv_image, anim_args.midas_weight, root.half_precision)
                depth_model.save(os.path.join(args.outdir, f"{args.timestring}_depth_{frame_idx:09}.png"), depth)
                if cmd_opts.lowvram or cmd_opts.medvram:
                    depth_model.to('cpu')
                    devices.torch_gc()
                    lowvram.setup_for_low_vram(sd_model, cmd_opts.medvram)
                    sd_hijack.model_hijack.hijack(sd_model)"""
            frame_idx += 1

        #state.current_image = image

        args.seed = next_seed(args)
        
    if predict_depths and not keep_in_vram:
        depth_model.delete_model() # handles adabins too
            
    if load_raft:
        raft_model.delete_model()


def generate(node, args, keys, anim_args, loop_args, controlnet_args, root, frame=0, return_sample=False, sampler_name=None):
    #if state.interrupted:
    #    return None

    #if args.reroll_blank_frames == 'ignore':
    #    return generate_inner(node, args, keys, anim_args, loop_args, controlnet_args, root, frame, return_sample,
    #                          sampler_name)
    image, caught_vae_exception = generate_with_nans_check(node, args, keys, anim_args, loop_args, controlnet_args, root,
                                                           frame, return_sample, sampler_name)

    """if caught_vae_exception or not image.getbbox():
        patience = args.reroll_patience
        print("Blank frame detected! If you don't have the NSFW filter enabled, this may be due to a glitch!")
        if args.reroll_blank_frames == 'reroll':
            while caught_vae_exception or not image.getbbox():
                print("Rerolling with +1 seed...")
                args.seed += 1
                image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args,
                                                                       controlnet_args, root, frame, return_sample,
                                                                       sampler_name)
                patience -= 1
                if patience == 0:
                    print(
                        "Rerolling with +1 seed failed for 10 iterations! Try setting webui's precision to 'full' and if it fails, please report this to the devs! Interrupting...")
                    state.interrupted = True
                    state.current_image = image
                    return None
        elif args.reroll_blank_frames == 'interrupt':
            print("Interrupting to save your eyes...")
            state.interrupted = True
            state.current_image = image
            return None"""

    return image


def generate_with_nans_check(node, args, keys, anim_args, loop_args, controlnet_args, root, frame=0, return_sample=False,
                             sampler_name=None):
    disable_nan_check = True
    if disable_nan_check:
        image = generate_inner(node, args, keys, anim_args, loop_args, controlnet_args, root, frame, return_sample,
                               sampler_name)
    else:
        try:
            image = generate_inner(node, args, keys, anim_args, loop_args, controlnet_args, root, frame, return_sample,
                                   sampler_name)
        except Exception as e:
            if "A tensor with all NaNs was produced in VAE." in repr(e):
                print(e)
                return None, True
            else:
                raise e
    return image, False


def generate_inner(node, args, keys, anim_args, loop_args, controlnet_args, root, frame=0, return_sample=False,
                   sampler_name=None):
    assert args.prompt is not None

    # Setup the pipeline
    #p = get_webui_sd_pipeline(args, root, frame)
    prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame, anim_args.max_frames)

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0
    processed = None
    mask_image = None
    init_image = None
    image_init0 = None

    if loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']:
        args.strength = loop_args.imageStrength
        tweeningFrames = loop_args.tweeningFrameSchedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.colorCorrectionFactor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        # find which image to show
        parsedImages = {}
        frameToChoose = 0
        max_f = anim_args.max_frames - 1

        for key, value in jsonImages.items():
            if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                parsedImages[key] = value
            else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsedImages[int(numexpr.evaluate(key))] = value

        framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))

        # find which frame to do our swapping on for tweening
        skipFrame = 25
        for fs, fe in pairwise_repl(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs

        if frame % skipFrame <= tweeningFrames:  # number of tweening frames
            blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope * math.cos(
                (frame % tweeningFrames) / (tweeningFrames / 2))
        init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                  shape=(args.W, args.H),
                                  use_alpha_as_mask=args.use_alpha_as_mask)
        image_init0 = list(jsonImages.values())[0]

    else:  # they passed in a single init image
        image_init0 = args.init_image

    available_samplers = {
        'euler a': 'Euler a',
        'euler': 'Euler',
        'lms': 'LMS',
        'heun': 'Heun',
        'dpm2': 'DPM2',
        'dpm2 a': 'DPM2 a',
        'dpm++ 2s a': 'DPM++ 2S a',
        'dpm++ 2m': 'DPM++ 2M',
        'dpm++ sde': 'DPM++ SDE',
        'dpm fast': 'DPM fast',
        'dpm adaptive': 'DPM adaptive',
        'lms karras': 'LMS Karras',
        'dpm2 karras': 'DPM2 Karras',
        'dpm2 a karras': 'DPM2 a Karras',
        'dpm++ 2s a karras': 'DPM++ 2S a Karras',
        'dpm++ 2m karras': 'DPM++ 2M Karras',
        'dpm++ sde karras': 'DPM++ SDE Karras'
    }
    """if sampler_name is not None:
        if sampler_name in available_samplers.keys():
            p.sampler_name = available_samplers[sampler_name]
        else:
            raise RuntimeError(
                f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")"""

    #if args.checkpoint is not None:
    #    info = sd_models.get_closet_checkpoint_match(args.checkpoint)
    #    if info is None:
    #        raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
    #    sd_models.reload_model_weights(info=info)

    if args.init_sample is not None:
        # TODO: cleanup init_sample remains later
        img = args.init_sample
        init_image = img
        image_init0 = img
        if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D', '3D']:
            init_image = Image.blend(init_image, init_image2, blendFactor)
            correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
            color_corrections = [correction_colors]

    # this is the first pass
    elif (loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']) or (
            args.use_init and ((args.init_image != None and args.init_image != ''))):
        init_image, mask_image = load_img(image_init0,  # initial init image
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)

    else:

        if anim_args.animation_mode != 'Interpolation':
            print(f"Not using an init image (doing pure txt2img)")
        """p_txt = StableDiffusionProcessingTxt2Img(
            sd_model=sd_model,
            outpath_samples=root.tmp_deforum_run_duplicated_folder,
            outpath_grids=root.tmp_deforum_run_duplicated_folder,
            prompt=p.prompt,
            styles=p.styles,
            negative_prompt=p.negative_prompt,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=p.sampler_name,
            batch_size=p.batch_size,
            n_iter=p.n_iter,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            restore_faces=p.restore_faces,
            tiling=p.tiling,
            enable_hr=None,
            denoising_strength=None,
        )"""

        #print_combined_table(args, anim_args, p_txt, keys, frame)  # print dynamic table to cli

        #if is_controlnet_enabled(controlnet_args):
        #    process_with_controlnet(p_txt, args, anim_args, loop_args, controlnet_args, root, is_img2img=False,
        #                            frame_idx=frame)

        processed = generate_with_node(node, prompt, negative_prompt, args, root, frame, init_images=None)

    if processed is None:
        # Mask functions
        if args.use_mask:
            mask_image = args.mask_image
            mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                (args.W, args.H),
                                args.mask_contrast_adjust,
                                args.mask_brightness_adjust)
            inpainting_mask_invert = args.invert_mask
            inpainting_fill = args.fill
            inpaint_full_res = args.full_res_mask
            inpaint_full_res_padding = args.full_res_mask_padding
            # prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
            # doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
            mask = check_mask_for_errors(mask, args.invert_mask)
            args.noise_mask = mask

        else:
            mask = None

        assert not ((mask is not None and args.use_mask and args.overlay_mask) and (
                    args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

        init_images = [init_image]
        image_mask = mask
        image_cfg_scale = args.pix2pix_img_cfg_scale

        #print_combined_table(args, anim_args, p, keys, frame)  # print dynamic table to cli

        #if is_controlnet_enabled(controlnet_args):
        #    process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True,
        #                            frame_idx=frame)


        processed = generate_with_node(node, prompt, negative_prompt, args, root, frame, init_images)
        #processed = processing.process_images(p)

    #if root.initial_info == None:
    #    root.initial_seed = processed.seed
    #    root.initial_info = processed.info

    if root.first_frame == None:
        root.first_frame = processed

    return processed
def encode_latent_ainodes(init_image):
    gs.models["vae"].first_stage_model.cuda()
    image = init_image
    image = image.convert("RGB")



    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.detach().half().cpu()
    torch_gc()

    latent = gs.models["vae"].encode(image)
    latent = latent.to("cpu")
    image = image.detach().to("cpu")
    del image
    return latent

def generate_with_node(node, prompt, negative_prompt, args, root, frame, init_images=None):
    sampler_node, _ = node.getInput(2)
    if isinstance(sampler_node, KSamplerNode):
        latent = None
        if init_images is not None:
            latent = encode_latent_ainodes(init_images[0])
        else:
            latent = torch.zeros([1, 4, args.H // 8, args.W // 8])
        cond_node, index = node.getInput(1)
        conds, _ = cond_node.evalImplementation_thread(prompt_override=prompt)
        n_conds, _ = cond_node.evalImplementation_thread(prompt_override=negative_prompt)
        pixmaps, _ = sampler_node.evalImplementation_thread(cond_override=[conds, n_conds], args=args, latent_override=latent)
    elif isinstance(sampler_node, KandinskyNode):
        if init_images is not None:
            init_images[0] = pil_image_to_pixmap(init_images[0])
        pixmaps = sampler_node.evalImplementation_thread(prompt_override=prompt, args=args, init_image=init_images)
    image = pixmap_to_pil_image(pixmaps[0])
    return image