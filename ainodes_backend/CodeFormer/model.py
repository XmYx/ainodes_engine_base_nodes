import os
from types import SimpleNamespace

import cv2
import argparse
import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url

from custom_nodes.ainodes_engine_base_nodes.ainodes_backend import torch_gc
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.CodeFormer.facelib.utils.face_restoration_helper import \
    FaceRestoreHelper
from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.CodeFormer.facelib.utils.misc import is_gray

#from basicsr.utils.misc import gpu_is_available, get_device


#print("CODEFORMER", os.getcwd())
import sys

sys.path.append(os.path.join(os.getcwd(), "src", "CodeFormerBasicSR"))
from c_basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def run_codeformer(args, input_img_list):
    from ainodes_frontend import singleton as gs

    device = gs.device
    bg_upsampler = set_realesrgan(args)
    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan(args)
    else:
        face_upsampler = None
    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                          connect_list=['32', '64', '128', '256']).to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                   model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned:
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler is not None:
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device)
    result_images = []
    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()

        #print(f'[{i + 1}/{test_img_num}] Processing: {img_name}')
        img_path = img_path.convert("RGB")
        img = np.array(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if args.has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)

            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=False, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
            w = args.fidelity_weight
            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not args.has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if args.face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box,
                                                                      face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

        # # save faces
        # for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
        #     # save cropped face
        #     if not args.has_aligned:
        #         save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
        #         imwrite(cropped_face, save_crop_path)
        #     # save restored face
        #     if args.has_aligned:
        #         save_face_name = f'{basename}.png'
        #     else:
        #         save_face_name = f'{basename}_{idx:02d}.png'
        #     if args.suffix is not None:
        #         save_face_name = f'{save_face_name[:-4]}_{args.suffix}.png'
        #     save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
        #     imwrite(restored_face, save_restore_path)

        # # save restored img
        # if not args.has_aligned and restored_img is not None:
        #     if args.suffix is not None:
        #         basename = f'{basename}_{args.suffix}'
        #     os.makedirs("final_results")
        #     save_restore_path = os.path.join('final_results', f'{basename}.png')
        #     imwrite(restored_img, save_restore_path)
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(restored_img)
        result_images.append(image)

    net.cpu()
    del checkpoint
    del net
    torch_gc()
    return result_images

def set_realesrgan(args):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    use_half = True
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    return upsampler
