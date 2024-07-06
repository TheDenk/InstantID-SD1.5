#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data.dataset import Dataset
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from instantid.ip_adapter.resampler import Resampler
from instantid.ip_adapter.utils import set_ip_adapter_processors
from instantid.pipelines.pipeline_instantid import StableDiffusionInstantPipeline
from instantid.data.dataset import InstantDataset

if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def print_trainable_params(model, model_name):
    total_params = list(model.parameters())
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    total_scale = sum(p.numel() for p in total_params) / 1e6
    trainable_scale = sum(p.numel() for p in trainable_params) / 1e6
    
    logger.info(f"[ {model_name} count trainable: {len(trainable_params)} | total: {len(total_params)} ]")
    logger.info(f"[ {model_name} scale trainable: {trainable_scale:.3f} M | total: {total_scale:.3f} M ]")
    logger.info("")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_models(accelerator, unet, controlnet, image_proj_model, epoch, global_step, args):
    save_folder = os.path.join(args.output_dir, f"checkpoints")
    os.makedirs(save_folder, exist_ok=True)
    
    ip_state_dict = {}
    unwrapped_unet = unwrap_model(accelerator, unet)
    for name, param in unwrapped_unet.named_parameters():
        if 'to_k_ip' in name or 'to_v_ip' in name:
            ip_state_dict[name] = param
    
    ip_save_info = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": ip_state_dict,
    }
    torch.save(ip_save_info, os.path.join(save_folder, f"ip-state-{global_step}.ckpt"))

    unwrapped_controlnet = unwrap_model(accelerator, controlnet)
    controlnet_save_info = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": unwrapped_controlnet.state_dict(),
    }
    torch.save(controlnet_save_info, os.path.join(save_folder, f"controlnet-{global_step}.ckpt"))

    unwrapped_image_proj_model = unwrap_model(accelerator, image_proj_model)
    image_proj_model_save_info = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": unwrapped_image_proj_model.state_dict(),
    }
    torch.save(image_proj_model_save_info, os.path.join(save_folder, f"image_proj-{global_step}.ckpt"))

    logging.info(f"Saved state to {save_folder} (global_step: {global_step})")


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, image_proj_model, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    unet = unwrap_model(accelerator, unet)
    controlnet = unwrap_model(accelerator, controlnet)
    image_proj_model = unwrap_model(accelerator, image_proj_model)

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
    )
    # pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_image_proj_model(image_proj_model, image_emb_dim=512)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_images = args.validation_image
    validation_prompts = args.validation_prompt
    validation_embeddings = args.valid_embeddings
    validation_negative_prompts = args.validation_negative_prompt
    

    if all(isinstance(x, str) for x in [validation_images, validation_prompts, validation_embeddings, validation_negative_prompts]):
        validation_images = [validation_images]
        validation_prompts = [validation_prompts]
        validation_embeddings = [validation_embeddings]
        validation_negative_prompts = [validation_negative_prompts]            
    elif all(isinstance(x, list) for x in [validation_images, validation_prompts, validation_embeddings, validation_negative_prompts]):
        pass
    else:
        logger.info(validation_prompts)
        logger.info(validation_negative_prompts)
        logger.info(validation_images)
        logger.info(validation_embeddings)
        raise Exception('Validation input data is not correct.')

    image_logs = []

    for validation_prompt, validation_image, validation_embedding, validation_negative_prompt in zip(validation_prompts, validation_images, validation_embeddings, validation_negative_prompts):
        validation_image = Image.open(validation_image).convert('RGB')
        validation_embedding = torch.load(validation_embedding)['embeddings']

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast('cuda'):
                image = pipeline(
                    validation_prompt,
                    negative_prompt=validation_negative_prompt,
                    image=validation_image,
                    ip_adapter_image_embeds=validation_embedding.to(device=accelerator.device),
                    num_inference_steps=50, 
                    guidance_scale=7.5, 
                    width=args.resolution,
                    height=args.resolution,
                    generator=generator,
                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "validation_image": validation_image.resize((args.resolution, args.resolution)), 
                "images": images, 
                "validation_prompt": validation_prompt
            }
        )

    images_folder = os.path.join(args.output_dir, 'images')
    os.makedirs(images_folder, exist_ok=True)

    for i, item in enumerate(image_logs):
        imgs_comb = np.hstack([np.array(x) for x in item['images']])
        image_path = os.path.join(images_folder, f'{step}-step-{i}-images.jpg')
        Image.fromarray(imgs_comb).save(image_path)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)
                
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset root.",
    )
    parser.add_argument(
        "--controlnet_ckpt_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model checkpoint with state dict.",
    )
    parser.add_argument(
        "--image_proj_ckpt_path",
        type=str,
        default=None,
        help="Path to pretrained image_proj_model model checkpoint with state dict.",
    )
    parser.add_argument(
        "--ip_adapter_ckpt_path",
        type=str,
        default=None,
        help="Path to pretrained ip_adapter model checkpoint with state dict.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates."
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--valid_embeddings",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_instantid15",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


class DummyDataset(Dataset):
    def __init__(self, dataset_root, tokenizer, image_size=512):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.dataset_root = dataset_root

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "image_pixels": torch.rand(3, 512, 512),
            "conditioning_pixel_values": torch.rand(3, 512, 512),
            "input_ids": torch.ones(77).long(),
            "embeddings": torch.rand(1, 512),
        }


def collate_fn(examples):
    image_pixels = torch.stack([example["image_pixels"] for example in examples])
    image_pixels = image_pixels.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])
    embeddings = torch.stack([example["embeddings"] for example in examples])

    return {
        "image_pixels": image_pixels,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "embeddings": embeddings,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    controlnet = ControlNetModel.from_unet(unet)
    logger.info("Controlnet weights have been initialized from unet.\n")
    
    adapter_modules = set_ip_adapter_processors(
        unet, num_tokens=16, scale=1.0, ignore_motion=True
    )
    image_proj_model = Resampler(
        dim=1280,
        dim_head=64,
        heads=20,
        depth=4,
        num_queries=16,
        embedding_dim=512,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )
    
    if args.controlnet_ckpt_path is not None:
        controlnet_ckpt = torch.load(args.controlnet_ckpt_path, map_location='cpu')
        controlnet_state_dict = controlnet_ckpt['state_dict'] if 'state_dict' in controlnet_ckpt else controlnet_ckpt
        m, u = controlnet.load_state_dict(controlnet_state_dict)
        logger.info(f'[CONTROLNET MODEL HAS BEEN LOADED] MISSING: {len(m)} | UNEXPECTED: {len(u)} |\n')

    if args.image_proj_ckpt_path is not None:
        image_proj_ckpt = torch.load(args.image_proj_ckpt_path, map_location='cpu')
        image_proj_state_dict = image_proj_ckpt['state_dict'] if 'state_dict' in image_proj_ckpt else image_proj_ckpt
        m, u = image_proj_model.load_state_dict(image_proj_state_dict)
        logger.info(f'[IMAGE PROJ MODEL HAS BEEN LOADED] MISSING: {len(m)} | UNEXPECTED: {len(u)} |\n')

    if args.ip_adapter_ckpt_path is not None:
        adapter_ckpt = torch.load(args.ip_adapter_ckpt_path, map_location='cpu')
        adapter_ckpt_state_dict = adapter_ckpt['state_dict'] if 'state_dict' in adapter_ckpt else adapter_ckpt
        adapter_sd = {}
        for name, param in adapter_ckpt['state_dict'].items():
            adapter_sd[name.replace('module.', '')] = param
        m, u = unet.load_state_dict(adapter_sd, strict=False)
        logger.info(f'[ADAPTER LAYERS HAVE BEEN LOADED] MISSING: {len(m)} | UNEXPECTED: {len(u)} |\n')

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    for key, value in unet.attn_processors.items():
        for n, p in value.named_parameters():
            p.requires_grad = True

    unet.train()
    controlnet.train()
    image_proj_model.train()

    print_trainable_params(unet, 'UNET')
    print_trainable_params(controlnet, 'CONTROLNET')
    print_trainable_params(image_proj_model, 'IMAGE PROJ MODEL')


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(accelerator, controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = itertools.chain(controlnet.parameters(), image_proj_model.parameters(),  adapter_modules.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # train_dataset = DummyDataset(None, None)
    params = {
        'dataset_root': args.dataset_root,
        'tokenizer': tokenizer,
        'image_size': args.resolution, 
    }
    train_dataset = InstantDataset(**params)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    unet, controlnet, image_proj_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, image_proj_model, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        tracker_config_args = {
            k:v for k, v in tracker_config.items() if any([isinstance(v, available_dtype) for available_dtype in [int, float, str, bool, torch.Tensor]])
        }
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config_args)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet) as _, accelerator.accumulate(image_proj_model) as _, accelerator.accumulate(adapter_modules) as _:
                # Convert images to latent space
                latents = vae.encode(batch["image_pixels"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                proj_embeds = image_proj_model(batch["embeddings"])

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=proj_embeds,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                encoder_hidden_states = torch.cat([encoder_hidden_states, proj_embeds], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_models(accelerator, unet, controlnet, image_proj_model, epoch, global_step, args)

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        _ = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            image_proj_model,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_models(accelerator, unet, controlnet, image_proj_model, epoch, global_step, args)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)