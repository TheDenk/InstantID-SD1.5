import torch
import numpy as np
from PIL import Image
from diffusers import (
    AutoencoderKL, 
    ControlNetModel, 
    UNet2DConditionModel,
)
from safetensors import safe_open
from insightface.app import FaceAnalysis
from transformers import CLIPTextModel, CLIPTokenizer

from instantid.utils.convert_from_ckpt import convert_ldm_unet_checkpoint
from instantid.utils.util import draw_2d_kps, draw_five_kps


def load_face_app(name='antelopev2', root='./'):
    face_app = FaceAnalysis(name=name, root=root, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640));
    return face_app
    

def load_pipeline_components(pretrained_model_path, torch_dtype=torch.float16):
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, 
        subfolder='tokenizer',
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, 
        subfolder='text_encoder',
        torch_dtype=torch_dtype,
    )
    
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path, 
        subfolder='vae',
        torch_dtype=torch_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path, 
        subfolder='unet', 
        torch_dtype=torch_dtype,
    )
    controlnet = ControlNetModel.from_unet(
        unet, 
    ).to(dtype=torch_dtype)
    return tokenizer, text_encoder, vae, unet, controlnet
    
    
def prepare_face_embeddings(source_image, face_app):
    target_image = np.array(source_image)
    face_info = face_app.get(target_image)
    
    face_info = sorted(face_info, key = lambda x:(x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[-1]  # only use the maximum face
    controlnet_image = draw_five_kps(target_image, face_info['kps'])
    controlnet_image = draw_2d_kps(controlnet_image, face_info['landmark_2d_106'], overlap_image=True)
    controlnet_image = Image.fromarray(controlnet_image)
    
    face_embedding = face_info['embedding'].astype(np.float16)
    return face_embedding, controlnet_image


def load_additional_unet(unet, additional_unet_path):
    unet_safe_tensors = {}
    
    with safe_open(additional_unet_path, framework='pt', device='cpu') as safe_sd:
        for key in safe_sd.keys():
            unet_safe_tensors[key] = safe_sd.get_tensor(key)
    
    unet_state_dict = convert_ldm_unet_checkpoint(unet_safe_tensors, unet.config)
    unet.load_state_dict(unet_state_dict)