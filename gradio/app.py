import sys
sys.path.append('./')

import os
import torch
import random
import numpy as np
import argparse

import PIL
import gradio as gr
from PIL import Image

from diffusers import (
    DDIMScheduler, 
)

from instantid.pipelines.pipeline_instantid import StableDiffusionInstantPipeline
from instantid.utils.load_components import load_face_app, load_pipeline_components, prepare_face_embeddings, load_additional_unet
from instantid.utils.style_template import update_prompt_for_style, instantid_styles as styles
from instantid.utils.util import check_if_file_set_but_not_exists


MAX_SEED = np.iinfo(np.int32).max
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Film Noir"


def main(args):
    for file_path in [args.adapter_ckpt_path, args.image_proj_ckpt_path, args.controlnet_ckpt_path, args.additional_unet_path]:
        check_if_file_set_but_not_exists(file_path)
    
    face_app = load_face_app()

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

    pipe = StableDiffusionInstantPipeline(
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

    pipe.load_instantid_components(
        args.adapter_ckpt_path, 
        args.image_proj_ckpt_path, 
        args.controlnet_ckpt_path
    )
    
    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed

    def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
                pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):

            w, h = input_image.size
            if size is not None:
                w_resize_new, h_resize_new = size
            else:
                ratio = min_side / min(h, w)
                w, h = round(ratio*w), round(ratio*h)
                ratio = max_side / max(h, w)
                input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
                w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
                h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
            input_image = input_image.resize([w_resize_new, h_resize_new], mode)

            if pad_to_max_side:
                res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
                offset_x = (max_side - w_resize_new) // 2
                offset_y = (max_side - h_resize_new) // 2
                res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
                input_image = Image.fromarray(res)
            return input_image

    def generate_image(face_image_path, pose_image_path, prompt, negative_prompt, style_name, num_steps, 
                       identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed):
        
        source_image = Image.open(face_image_path).convert("RGB")
        source_image = resize_img(source_image, max_side=1024, min_side=512)
        face_embedding, controlnet_image = prepare_face_embeddings(source_image, face_app)
        width, height = controlnet_image.size

        if pose_image_path is not None and os.path.isfile(pose_image_path):
            pose_image = Image.open(pose_image_path).convert("RGB")
            source_image = resize_img(pose_image, max_side=1024, min_side=512)
            _, controlnet_image = prepare_face_embeddings(pose_image, face_app)
            width, height = controlnet_image.size

        prompt, n_prompt = update_prompt_for_style(prompt, negative_prompt, style_name)

        gen_images = pipe(
            prompt,
            image = controlnet_image,
            ip_adapter_image_embeds = face_embedding, 
            negative_prompt = n_prompt,
            num_inference_steps = num_steps,
            guidance_scale = guidance_scale,
            width = width,
            height = height,
            controlnet_conditioning_scale=identitynet_strength_ratio,
            ip_adapter_scale=adapter_strength_ratio,
            generator=torch.Generator(device='cuda').manual_seed(seed),
            clip_skip=0,
            num_images_per_prompt=1,
        )[0]

        return gen_images[0], gr.update(visible=True)

    ### Description
    title = r"""
    <h1 align="center">InstantID for StableDiffusion 1.5</h1>
    """

    description = r"""
    <b>UNOfficial 🤗 Gradio demo</b> for <a href='https://github.com/TheDenk/InstantID-SD1.5' target='_blank'><b>InstantID for SD 1.5</b></a> based on <a href="https://github.com/InstantID/InstantID">InstantID: Zero-shot Identity-Preserving Generation in Seconds</a>.<br>

    How to use:<br>
    1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
    3. Enter a text prompt, as done in normal text-to-image models.
    4. Click the <b>Submit</b> button to begin customization.
    """

    article = r"""
    ---
    THIS WORK BASED ON <a href="https://github.com/InstantID/InstantID">InstantID: Zero-shot Identity-Preserving Generation in Seconds</a>:
    <br>
    ```bibtex
    @article{wang2024instantid,
    title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
    author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
    journal={arXiv preprint arXiv:2401.07519},
    year={2024}
    }
    ```
    """

    tips = r"""
    ### Usage tips of InstantID
    1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."    
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
    3. If you find that text control is not as expected, decrease Adapter strength.
    4. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
    """

    css = '''
    .gradio-container {width: 85% !important}
    '''
    with gr.Blocks(css=css) as demo:

        # description
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                
                # upload face image
                face_file = gr.Image(label="Upload a photo of your face", type="filepath")

                # optional: upload a reference pose image
                pose_file = gr.Image(label="Upload a reference pose image (optional)", type="filepath")
           
                # prompt
                prompt = gr.Textbox(label="Prompt",
                        info="Give simple prompt is enough to achieve good face fidelity",
                        placeholder="A photo of a person",
                        value="")
                
                submit = gr.Button("Submit", variant="primary")
                
                style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
                
                # strength
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                
                with gr.Accordion(open=False, label="Advanced Options"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt", 
                        placeholder="low quality",
                        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                    )
                    num_steps = gr.Slider( 
                        label="Number of sample steps",
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=25,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=7,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Column():
                gallery = gr.Image(label="Generated Images")
                usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips ,visible=False)

            submit.click(
                fn=randomize_seed_fn,
                inputs=[seed, randomize_seed],
                outputs=seed,
                queue=False,
                api_name=False,
            ).then(
                fn=generate_image,
                inputs=[face_file, pose_file, prompt, negative_prompt, style, num_steps, identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed],
                outputs=[gallery, usage_tips]
            )
        gr.Markdown(article)
    demo.launch(share=True)


if __name__ == "__main__":
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
    args = parser.parse_args()
    main(args)