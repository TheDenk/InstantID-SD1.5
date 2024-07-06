import math

import cv2
import PIL
import torch
import numpy as np
import torch.nn.functional as F

from instantid.ip_adapter.resampler import Resampler

if hasattr(F, "scaled_dot_product_attention"):  ## Torch 2. is available
    from instantid.ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from instantid.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


def set_image_proj_model(pipe, model_ckpt, image_emb_dim=512, num_tokens=16):
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=image_emb_dim,
        output_dim=pipe.unet.config.cross_attention_dim,
        ff_mult=4,
    )

    image_proj_model.eval()
    
    pipe.image_proj_model = image_proj_model.to(pipe.device, dtype=pipe.dtype)
    state_dict = torch.load(model_ckpt, map_location="cpu")
    if 'image_proj' in state_dict:
        state_dict = state_dict["image_proj"]
    pipe.image_proj_model.load_state_dict(state_dict)
    pipe.image_proj_model_in_features = image_emb_dim
    

def set_ip_adapter_processors(unet, num_tokens, scale, ignore_motion=False, state_dict=None):
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if (cross_attention_dim is None) or (ignore_motion and 'motion_modules.' in name):
            attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
        else:
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                cross_attention_dim=cross_attention_dim, 
                                                scale=scale,
                                                num_tokens=num_tokens).to(unet.device, dtype=unet.dtype)
            
            if state_dict is not None:
                attn_procs[name].load_state_dict(state_dict)
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules

def load_ip_adapter_weights(unet, model_ckpt):
    state_dict = torch.load(model_ckpt, map_location="cpu")
    ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
    if 'ip_adapter' in state_dict:
        state_dict = state_dict['ip_adapter']
    ip_layers.load_state_dict(state_dict)


def set_ip_adapter_scale(unet, scale):
    for attn_processor in unet.attn_processors.values():
        if isinstance(attn_processor, IPAttnProcessor):
            attn_processor.scale = scale


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

