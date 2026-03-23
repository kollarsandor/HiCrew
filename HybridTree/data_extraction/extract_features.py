from PIL import Image
import requests
import torch
import transformers
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import os
from pprint import pprint
import pdb

import numpy as np
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoConfig
from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode



def save_image_features(img_feats, name_ids, save_folder):
    """
    Save image features to a .pt file in a specified folder.

    Args:
    - img_feats (torch.Tensor): Tensor containing image features
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    torch.save(img_feats, filepath)


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)



# image_path = "CLIP.png"
model_name_or_path = "BAAI/EVA-CLIP-8B" # or /path/to/local/EVA-CLIP-8B
image_size = 224

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


def clip_es():
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()

    base_path = Path('./egoschema_frames')
    save_folder = './frame_features'
    
    # 确保保存目录存在
    os.makedirs(save_folder, exist_ok=True)
    
    # 获取已经处理过的文件列表
    existing_files = set(f.stem for f in Path(save_folder).glob('*.pt'))
    print(f"已处理的视频数量: {len(existing_files)}")
    
    # # 读取subset_anno.json (不再需要)
    # with open('./data/egoschema/subset_anno.json', 'r') as file:
    #     json_data = json.load(file)    
    # subset_names_list = list(json_data.keys())
    
    # 获取所有视频路径
    example_path_list = list(base_path.iterdir())
    print(f"egoschema_frames目录中的视频总数: {len(example_path_list)}")
    
    # 筛选出需要处理的视频（所有未处理的视频）
    remaining_videos = [p for p in example_path_list 
                       if p.name not in existing_files]
    
    print(f"\n需要处理的新视频数量: {len(remaining_videos)}")
    if len(remaining_videos) == 0:
        print("\n所有视频都已处理完成！")
        return

    pbar = tqdm(remaining_videos)

    # 直接遍历remaining_videos
    for example_path in pbar:
        try:
            pbar.set_description(f"处理视频: {example_path.name}")
            
            image_paths = list(example_path.iterdir())
            image_paths.sort(key=lambda x: int(x.stem))
            img_feature_list = []
            
            for image_path in image_paths:
                image = Image.open(str(image_path))
                input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to('cuda')

                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(input_pixels)
                    img_feature_list.append(image_features)
                
                # 及时释放内存
                image.close()
                torch.cuda.empty_cache()

            img_feature_tensor = torch.stack(img_feature_list)
            img_feats = img_feature_tensor.squeeze(1)

            name_ids = example_path.name
            save_image_features(img_feats, name_ids, save_folder)

            # 清理内存
            del img_feature_list
            del img_feature_tensor
            del img_feats
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"处理视频 {example_path.name} 时出错: {str(e)}")
            continue

    pbar.close()

if __name__ == '__main__':
    clip_es()

