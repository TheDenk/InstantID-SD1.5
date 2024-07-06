import argparse
import os

import torch
from PIL import Image
from diffusers import (
    DDIMScheduler, 
)

from instantid.pipelines.pipeline_instantid import StableDiffusionInstantPipeline
from instantid.utils.load_components import load_face_app, load_pipeline_components, prepare_face_embeddings, load_additional_unet
from instantid.utils.util import resize_max_side_to_target, check_if_file_set_but_not_exists
from instantid.utils.style_template import update_prompt_for_style

'''
## RUN EXAMPLE
#### SIMPLE RUN
CUDA_VISIBLE_DEVICES="4" python3 inference.py \
    --image_path=examples/faces/rock.jpg \
    --prompt="the professional high quality photo of the man, high quality, best quality, masterpeace" \
    --style="Film Noir" \
    --height=640 \
    --width=768 \
    --num_inference_steps=25 \
    --guidance_scale=8.0 \
    --num_images_per_prompt=4

    
#### SELECT MODELS 
CUDA_VISIBLE_DEVICES="4" python3 inference.py --pretrained_model_path=models/stable-diffusion-v1-5 \
    --adapter_ckpt_path=models/instantid-components/ip-state.ckpt \
    --image_proj_ckpt_path=models/instantid-components/image_proj.ckpt \
    --controlnet_ckpt_path=models/instantid-components/controlnet.ckpt \
    --additional_unet_path=models/additional-unets/epicphotogasm_lastUnicorn.safetensors \
    --image_path=examples/faces/rock.jpg \
    --prompt="the professional high quality photo of the man, high quality, best quality, masterpeace" \
    --style="Film Noir" \
    --height=640 \
    --width=768 \
    --num_inference_steps=25 \
    --guidance_scale=8.0 \
    --num_images_per_prompt=4
'''


def parse_args():
    parser = argparse.ArgumentParser(description="StableDiffusion 1.5 InstantID inference.")
    parser.add_argument(
        "--pretrained_model_path", type=str, default="models/stable-diffusion-v1-5", required=False, help="A folder containing sd15 models."
    )
    parser.add_argument(
        "--adapter_ckpt_path", type=str, default="models/instantid-components/ip-state.ckpt", required=False, help="Path to ip-apapter model checkpoint."
    )
    parser.add_argument(
        "--image_proj_ckpt_path", type=str, default="models/instantid-components/image_proj.ckpt", required=False, help="Path to image projection model checkpoint."
    )
    parser.add_argument(
        "--controlnet_ckpt_path", type=str, default="models/instantid-components/controlnet.ckpt", required=False, help="Path to controlnet model checkpoint."
    )
    parser.add_argument(
        "--additional_unet_path", type=str, default=None, required=False, help="Path to other finetuned unet."
    )
    parser.add_argument(
        "--image_path", type=str, default=None, required=True, help="Path to image for processing."
    )
    parser.add_argument(
        "--pose_image_path", type=str, default=None, required=False, help="Path target face position."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", required=False, help="A folder for generated images."
    )
    parser.add_argument(
        "--style", type=str, default=None, required=False, help="One style from classic InstanceID. [No style, Watercolor, Film Noir, Neon, Jungle, Mars, Vibrant Color, Snow, Line art]"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, required=True, help="Prompt for image generation."
    )
    parser.add_argument(
        "--n_prompt", type=str, default="hands, text, title, headline, worst quality, low quality", help="Negative prompt for image generation."
    )
    parser.add_argument(
        "--height", type=int, default=640, help="Output image height."
    )
    parser.add_argument(
        "--width", type=int, default=768, help="Output image width."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=25, help="Denoising steps."
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="Output images count."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="For random control."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=8.0, help="Prompt strength."
    )
    parser.add_argument(
        "--controlnet_conditioning_scale", type=float, default=0.8, help="Ensures that face keypoints will have same positions."
    )
    parser.add_argument(
        "--ip_adapter_scale", type=float, default=0.8, help="Ensures that face will be similar to original."
    )
    parser.add_argument(
        "--clip_skip", type=int, default=0, help="Clip last layer."
    )
    parser.add_argument(
        "--antelopev2_name", type=str, default="antelopev2", required=False, help="Name of dir for antilopev2 onnx models."
    )
    parser.add_argument(
        "--antelopev2_root", type=str, default="./", required=False, help="Path of root dir for antilopev2 onnx models. *root_dir*/models/*name_dir*"
    )
    args = parser.parse_args()
    return args


def main(args):
    for file_path in [args.adapter_ckpt_path, args.image_proj_ckpt_path, args.controlnet_ckpt_path, args.additional_unet_path]:
        check_if_file_set_but_not_exists(file_path)

    face_app = load_face_app(name=args.antelopev2_name, root=args.antelopev2_root)

    tokenizer, text_encoder, vae, unet, controlnet = load_pipeline_components(args.pretrained_model_path)

    if args.additional_unet_path is not None and os.path.isfile(args.additional_unet_path):
        load_additional_unet(unet, args.additional_unet_path)

    scheduler = DDIMScheduler(**{
        'beta_start': 0.00085,
        'beta_end': 0.012,
        'beta_schedule': 'linear',
        'steps_offset': 1,
        'clip_sample': False,
    })

    pipeline = StableDiffusionInstantPipeline(
        vae=vae, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        image_encoder=None,
        safety_checker=None,
        feature_extractor=None,
    ).to('cuda')

    pipeline.load_instantid_components(
        args.adapter_ckpt_path, 
        args.image_proj_ckpt_path, 
        args.controlnet_ckpt_path
    )
    
    source_image = Image.open(args.image_path).convert("RGB")
    source_image = resize_max_side_to_target(source_image, target_side=max(args.width, args.height))
    face_embedding, controlnet_image = prepare_face_embeddings(source_image, face_app)

    if args.pose_image_path is not None and os.path.isfile(args.pose_image_path):
        pose_image = Image.open(args.pose_image_path).convert("RGB").resize((args.width, args.height))
        pose_image = resize_max_side_to_target(pose_image, target_side=max(args.width, args.height))
        _, controlnet_image = prepare_face_embeddings(pose_image, face_app)

    prompt, n_prompt = update_prompt_for_style(args.prompt, args.n_prompt, args.style)

    gen_images = pipeline(
        prompt,
        image = controlnet_image,
        ip_adapter_image_embeds = face_embedding, 
        negative_prompt = n_prompt,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = args.guidance_scale,
        width = args.width,
        height = args.height,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        generator=torch.Generator(device='cuda').manual_seed(args.seed),
        clip_skip=args.clip_skip,
        num_images_per_prompt=args.num_images_per_prompt,
    )[0]

    os.makedirs(args.output_dir, exist_ok=True)
    for img_index, gen_image in enumerate(gen_images):
        out_image_path = os.path.join(args.output_dir, f'{img_index} {args.prompt[:50]}.jpg')
        gen_image.save(out_image_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)