import os
import glob

import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from transformers import CLIPTokenizer


class InstantDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        tokenizer,
        image_size=512, 
        t_drop_rate=0.05, 
        i_drop_rate=0.05, 
        ti_drop_rate=0.05,
    ):
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        
        self.tokenizer = tokenizer
        self.images_folder = os.path.join(dataset_root, 'extracted_images')
        self.keypoints_folder = os.path.join(dataset_root, 'extracted_keypoints')

        csv_paths = glob.glob(os.path.join(dataset_root, 'csv', '*.csv'))[:20]
        self.df = pd.read_csv(csv_paths[0])
        for csv_path in csv_paths[1:]:
            self.df = pd.concat([self.df, pd.read_csv(csv_path)])

        embeddings_paths = glob.glob(os.path.join(dataset_root, 'embeddings', '*.pt'))[:20]
        self.embeddings = torch.load(embeddings_paths[0])
        for embeddings_path in embeddings_paths[1:]:
            self.embeddings.update(torch.load(embeddings_path))

        self.pixel_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(image_size),
        ])

    def encode_prompt(self, prompt, with_uncond=False):
        input_ids = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            ).input_ids

        uncond_input_ids = None
        if with_uncond:
            uncond_input_ids = self.tokenizer(
                    '',
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors='pt',
                ).input_ids
        return input_ids, uncond_input_ids

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pixels = torch.from_numpy(image)
        image_pixels = image_pixels.permute(2, 0, 1).contiguous()
        image_pixels = image_pixels / 255.0
        image_pixels = 2.0 * image_pixels - 1.0
        return image_pixels
        
    def get_batch(self, idx):
        row = self.df.iloc[idx]
        text = row.text
        basename = row.image_name.replace('.jpg', '')
        embeddings = self.embeddings[basename]['embeddings']

        rand_num = np.random.random()
        if rand_num < self.i_drop_rate:
            embeddings = torch.zeros_like(embeddings)
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ''
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ''
            embeddings = torch.zeros_like(embeddings)
            
        input_ids, uncond_input_ids = self.encode_prompt(text)
        
        image_path = os.path.join(self.images_folder, row.subfolder, row.image_name)
        keypoints_path = os.path.join(self.keypoints_folder, row.subfolder, row.image_name)
        
        image_pixels = self.preprocess_image(image_path)
        keypoints_pixels = self.preprocess_image(keypoints_path)
        
        return image_pixels, keypoints_pixels, embeddings, input_ids, uncond_input_ids

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        while True:
            try:
                image_pixels, keypoints_pixels, embeddings, input_ids, _ = self.get_batch(idx)
                break
            except Exception as e:
                index = np.random.randint(0, self.df.shape[0])
                return self.__getitem__(index)

        image_pixels = self.pixel_transforms(image_pixels)
        keypoints_pixels = self.pixel_transforms(keypoints_pixels)
        
        return {
            'input_ids': input_ids,
            'embeddings': embeddings.unsqueeze(0),
            'image_pixels': image_pixels,
            'conditioning_pixel_values': keypoints_pixels,
        }


if __name__ == '__main__':
    params = {
        'dataset_root': '/home/raid/datasets/LAION-Face',
        'tokenizer': CLIPTokenizer.from_pretrained(
            'runwayml/stable-diffusion-v1-5', 
            cache_dir='/home/raid/hf_cache',
            subfolder='tokenizer',
            torch_dtype=torch.float16,
        ),
        'image_size': 512, 
    }

    dataset = InstantDataset(**params)
    print(f'Dataset len: {len(dataset)}')
