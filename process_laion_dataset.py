# python scripts/process_laion_dataset_0.py
import argparse
import os
import io
import glob
import tarfile
from collections import defaultdict

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from insightface.app import FaceAnalysis
from denku import do_multiprocess, split_on_chunks

from instantid.utils.keypoints_utils import draw_2d_kps, draw_five_kps


'''
python scripts/process_laion_dataset.py \
    --data_root={DATASET_ROOT}/LAION-Face/laion_face_data/split_00001 \
    --split_name=split_00001 \
    --n_jobs=4
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Script for preparation laion-face dataset files")
    parser.add_argument(
        "--data_root", type=str, default=None, required=True, help="Path to laion files directory. Example: ../LAION-Face/"
    )
    parser.add_argument(
        "--split_name", type=str, default=None, required=True, help="Split folder name (folder with *.tar files). Example: split_00001"
    )
    parser.add_argument(
        "--antelopev2_name", type=str, default="antelopev2", required=False, help="Name of dir for antilopev2 onnx models."
    )
    parser.add_argument(
        "--antelopev2_root", type=str, default="./", required=False, help="Path of root dir for antilopev2 onnx models. *root_dir*/models/*name_dir*"
    )
    parser.add_argument(
        "--min_h", type=int, default=512, required=False, help="Minmal image height."
    )
    parser.add_argument(
        "--min_w", type=int, default=512, required=False, help="Minimal image width."
    )
    parser.add_argument(
        "--min_head_coef", type=float, required=False, default=0.3, help="Head size have to be more than IMG_H * min_head_coef and more than IMG_W * min_head_coef"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, required=False, help="Process count."
    )
    args = parser.parse_args()
    return args


def get_unique_names(names):
    unique_names = [os.path.basename(x).split('.')[0] for x in names]
    unique_names = dict().fromkeys(unique_names)
    unique_names = list(unique_names.keys())
    return unique_names
    

def parallel_process(params):
    tar_paths, args, chunk_index = params

    out_images_folder = os.path.join(args.data_root, 'extracted_images')
    out_keypoints_folder = os.path.join(args.data_root, 'extracted_keypoints')
    
    os.makedirs(out_images_folder, exist_ok=True)
    os.makedirs(out_keypoints_folder, exist_ok=True)
    
    app = FaceAnalysis(name=args.antelopev2_name, root=args.antelopev2_root, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    min_h, min_w = args.min_h, args.min_w
    min_head_coef = args.min_head_coef
    
    out_df = defaultdict(list)
    
    for tar_path in tqdm(tar_paths):
        try:
            chunk_folder = tar_path.split('/')[-2]
            tar_name = os.path.basename(tar_path).replace('.tar', '')
            
            os.makedirs(os.path.join(out_images_folder, chunk_folder), exist_ok=True)
            os.makedirs(os.path.join(out_keypoints_folder, chunk_folder), exist_ok=True)
            
            out_info = defaultdict(dict)
            
            t_file = tarfile.open(tar_path)
            names = t_file.getnames()
            unique_names = get_unique_names(names)
            for name_index, unique_name in enumerate(unique_names):
                ### READ IMAGE FILE
                image_file = t_file.extractfile(f'{unique_name}.jpg').read()
                image = Image.open(io.BytesIO(image_file))
                img_w, img_h = image.size
                if img_w < min_w or img_h < min_h:
                    continue
            
                image = np.array(image)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    
                face_info = app.get(image)
                
                if len(face_info):
                    face_info = sorted(face_info, key = lambda x:(x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[-1]  # only use the maximum face
                    face_w, face_h = face_info['bbox'][2] - face_info['bbox'][0],  face_info['bbox'][3] - face_info['bbox'][1]
                    img_h, img_w = image.shape[:2]
                    if (face_h / img_h < min_head_coef) or (face_w / img_w < min_head_coef):
                        continue
                    
                    face_points = draw_five_kps(image, face_info['kps'])
                    face_points = draw_2d_kps(face_points, face_info['landmark_2d_106'], overlap_image=True)
                else:
                    continue
                
                ### READ TXT FILE
                text_file = t_file.extractfile(f'{unique_name}.txt').read()
                text = io.BytesIO(text_file).read().decode('utf-8')
                
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                face_points = cv2.cvtColor(face_points, cv2.COLOR_RGB2BGR)
        
                out_image_path = os.path.join(out_images_folder, chunk_folder, f'{unique_name}.jpg')
                cv2.imwrite(out_image_path, image)
                out_keypoint_path = os.path.join(out_keypoints_folder, chunk_folder, f'{unique_name}.jpg')
                cv2.imwrite(out_keypoint_path, face_points)
                ### SAVE INFO ABOUT EMBEDDINGS
                out_info[unique_name]['text'] = text
                out_info[unique_name]['kps'] = face_info['kps'] if face_info is not None else None
                out_info[unique_name]['ages'] = face_info['age'] if face_info is not None else None
                out_info[unique_name]['boxes'] = torch.from_numpy(face_info['bbox'].astype(np.int32)) if face_info is not None else None
                out_info[unique_name]['genders'] = face_info['gender'] if face_info is not None else None
                out_info[unique_name]['embeddings'] = torch.from_numpy(face_info['embedding'].astype(np.float16)) if face_info is not None else None
                out_info[unique_name]['landmark_2d_106'] = torch.from_numpy(face_info['landmark_2d_106'].astype(np.float16)) if face_info is not None else None
                ### SAVE INFO ABOUT EMBEDDINGS
                out_df['image_name'].append(f'{unique_name}.jpg')
                out_df['subfolder'].append(chunk_folder)
                out_df['text'].append(text)
                # break
            
            torch.save(
                dict(out_info), 
                os.path.join(args.data_root, 'embeddings', f'{chunk_folder}_{tar_name}_{chunk_index}.pt'),
            )
            pd.DataFrame(data=out_df).to_csv(
                os.path.join(args.data_root, 'csv', f'{chunk_folder}_{tar_name}_{chunk_index}.csv'),
            )
        except Exception as e:
            print(f'\n\n{tar_path}')
            print(f'\n{e}\n\n')


def main():
    args = parse_args()

    tar_paths = glob.glob(os.path.join(args.data_root, 'laion_face_data', args.split_name, '*.tar'))
    tar_paths = list(sorted(tar_paths))
    tar_chunks = split_on_chunks(tar_paths, args.n_jobs)
    
    print(f'\n[ INFO | N_JOBS: {args.n_jobs} | CHUNK_COUNT: {len(tar_chunks)} | TOTAL TARS: {len(tar_paths)} ]\n')
    params = [(tar_chunk, args, chunk_index) for chunk_index, tar_chunk in enumerate(tar_chunks)]
    do_multiprocess(parallel_process, params, args.n_jobs)

    
if __name__ == '__main__':
    main()