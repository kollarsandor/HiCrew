# 从zjq_temp中拷贝过来的
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
import torch.nn.functional as F

### VLMs

# blip2
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

### LLMs

# llama2
from transformers import AutoTokenizer
# flanT5
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scipy.cluster.hierarchy import linkage, fcluster

# 子类和子子类调整为最大是2
def hierarchical_clustering_with_external_primary(segment_features, cluster_ids, relevance_scores, num_subclusters=2,
                                                  num_subsubclusters=2):
    # 确保relevance_scores不是None
    if relevance_scores is None:
        relevance_scores = []
        
    clusters = {i: {} for i in range(2)}

    for cluster_id in set(cluster_ids):
        # 得到全局帧索引
        primary_indices = [i for i, x in enumerate(cluster_ids) if x == cluster_id]

        if cluster_id < len(relevance_scores):
            score = relevance_scores[cluster_id]
        else:
            score = 3

        if len(primary_indices) < 2:
            continue

        sub_features = segment_features[primary_indices]

        linked_sub = linkage(sub_features, method='ward')
        sub_cluster_labels = fcluster(linked_sub, num_subclusters, criterion='maxclust')
        sub_cluster_labels = sub_cluster_labels - 1

        for subcluster_id in range(0, num_subclusters):
            sub_indices = np.where(sub_cluster_labels == subcluster_id)[0]
            if len(sub_indices) < 2:
                continue

            subsub_features = sub_features[sub_indices]
            linked_subsub = linkage(subsub_features, method='ward')
            subsub_cluster_labels = fcluster(linked_subsub, num_subsubclusters, criterion='maxclust')
            subsub_cluster_labels = subsub_cluster_labels - 1

            clusters[cluster_id][subcluster_id] = {}
            for subsubcluster_id in range(0, num_subsubclusters):
                final_indices = sub_indices[
                    np.where(subsub_cluster_labels == subsubcluster_id)[0]]  # Correctly index into sub_indices
                original_indices = [primary_indices[i] for i in final_indices]
                clusters[cluster_id][subcluster_id][subsubcluster_id] = original_indices

    return clusters


def cosine_similarity(points, centroid):
    """
    Calculate cosine similarity between points and centroid.
    Returns the cosine distances (1 - similarity).
    """
    points_normalized = F.normalize(points, dim=1)
    centroid_normalized = F.normalize(centroid.unsqueeze(0), dim=1)
    return 1 - torch.mm(points_normalized, centroid_normalized.T).squeeze()


def find_closest_points_in_temporal_order_subsub(x, clusters, relevance_scores):
    # 确保relevance_scores不是None
    if relevance_scores is None:
        relevance_scores = []

    # 删掉了很多分类讨论的代码，只讨论评分为3，存在子子聚类的内容并提出最深层的代表帧展平
    closest_points_indices = []

    for cluster_id, cluster_data in clusters.items():

        if cluster_id < len(relevance_scores):
            relevance = relevance_scores[cluster_id]
        else:
            relevance = 3

        if isinstance(cluster_data, dict):  # Handle subclusters and sub-subclusters
            if relevance == 1 or relevance == 2:
                 return []
            else:
                # Include primary cluster representative
                primary_indices = []
                for subcluster_data in cluster_data.values():
                    if isinstance(subcluster_data, dict):
                        for sub_data in subcluster_data.values():
                            if len(sub_data) > 0:
                                primary_indices.append(sub_data)
                for subcluster_id, subclusters in cluster_data.items():
                    if isinstance(subclusters, dict):  # Sub-subclusters
                        for subsubcluster_id, indices in subclusters.items():
                            if len(indices) == 0:
                                continue  # Skip empty sub-subclusters
                            indices_tensor = torch.tensor(indices, dtype=torch.long)
                            points_in_subsubcluster = x[indices_tensor]
                            subsubcluster_centroid = points_in_subsubcluster.mean(dim=0)
                            distances = cosine_similarity(points_in_subsubcluster, subsubcluster_centroid)
                            if distances.numel() > 0:
                                closest_idx_in_subsubcluster = torch.argmin(distances).item()
                                closest_global_idx = indices[closest_idx_in_subsubcluster]
                                closest_points_indices.append(int(closest_global_idx))
    closest_points_indices.sort()  # Ensure the points are in temporal order
    return closest_points_indices


def load_image_features(name_ids, save_folder):
    """
    Load image features from a .pt file.

    Args:
    - filename (str): Name of the .pt file to load

    Returns:
    - img_feats (torch.Tensor): Loaded image features
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    img_feats = torch.load(filepath)
    return img_feats


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def depth_expansion():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_base_path = Path('./clip_es')
    output_base_path.mkdir(parents=True, exist_ok=True)
    base_path = Path('./egoschema_frames')
    save_folder = './frame_features'

    rel_path = './dynamic_width_expansion/relevance_score.json'
    with open(rel_path, 'r') as file:
        cap_score_data = json.load(file)

    width_res_path = './dynamic_width_expansion/width_res.json'
    with open(width_res_path, 'r') as file:
        width_res_data = json.load(file)
    width_cluster_id_dict={}
    for item in width_res_data:
        if width_cluster_id_dict.get(item['video_uid']):
            width_cluster_id_dict[item['video_uid']][item["tree_node"][0]]=item['cluster_ids_x']
        else:
            width_cluster_id_dict[item['video_uid']]={item["tree_node"][0]:item['cluster_ids_x']}
    # sbd切片文件加载
    with open("./dynamic_width_expansion/segment_sbd.json", 'r') as file:
        segment_sbd_data = json.load(file)
    all_data = []

    # with open('path/data/egoschema/subset_answers.json', 'r') as file:
    #     json_data = json.load(file)
    # subset_names_list = list(json_data.keys())
    # print("subset_names_list",subset_names_list)

    example_path_list = list(base_path.iterdir())

    pbar = tqdm(total=len(example_path_list))

    i = 0
    max = 1

    # 获取relevance_score.json中存在的视频ID列表
    available_video_ids = set(cap_score_data.keys())

    for example_path in example_path_list:
        name_ids = example_path.name

        # 只处理在relevance_score.json中存在的视频
        if name_ids not in available_video_ids:
            pbar.update(1)
            print(f"跳过视频 {name_ids}: 在relevance_score.json中找不到相关数据")
            continue

        # 检查该视频是否在width_cluster_id_dict中
        if name_ids not in width_cluster_id_dict:
            pbar.update(1)
            print(f"跳过视频 {name_ids}: 在width_res.json中找不到相关数据")
            continue

        img_feats = load_image_features(name_ids, save_folder)
        # 评分文件，该视频所有片段的对应评分
        segments_scores = cap_score_data[name_ids]['segments']
        # sbd切片文件
        segment_sbd = segment_sbd_data[name_ids]
        # 视频各片段节点所属
        primary_cluster_ids = width_cluster_id_dict.get(name_ids, None)
        if primary_cluster_ids is None:
            pbar.update(1)
            print(f"跳过视频 {name_ids}: width_cluster_id_dict中没有相关数据")
            continue
            
        for seg_id, segment in segments_scores.items():
            # 片段评分列表
            relevance_scores = segment.get('pred')
            
            # 检查relevance_scores是否有效
            if relevance_scores is None:
                print(f"跳过片段 {seg_id}: 评分数据为None")
                continue
            # 片段开始时间按，计算全局帧索引
            seg_time = next((s for s in segment_sbd if s["uid"] == int(seg_id)), None)
            if not seg_time:
                continue
            start_time = seg_time["start_time"]
            end_time = seg_time["end_time"]
            # 从视频特征中提取片段特征
            segment_feats = img_feats[int(start_time):int(end_time) + 1]
            segment_feats = segment_feats.cpu()
            # 片段聚类
            segment_cluster_ids = primary_cluster_ids.get(int(seg_id), None)
            if segment_cluster_ids is None:
                print(f"跳过片段 {seg_id}: 在primary_cluster_ids中找不到相关数据")
                continue
                
            print(seg_id)
            print(segment_cluster_ids)
            clusters_info = hierarchical_clustering_with_external_primary(segment_feats, segment_cluster_ids, relevance_scores,
                                                                      num_subclusters=2, num_subsubclusters=2)

            closest_points_temporal_subsub = find_closest_points_in_temporal_order_subsub(segment_feats, clusters_info,
                                                                                      relevance_scores)
            # print("closest_points_temporal_subsub",closest_points_temporal_subsub)
            closest_points_temporal_subsub=[i+start_time for i in closest_points_temporal_subsub]
            all_data.append(
            {"name": example_path.name,
             "segment": int(seg_id),
             "sorted_values": closest_points_temporal_subsub,
             "relevance": relevance_scores})

        pbar.update(1)

    save_json(all_data, './depth_expansion_res.json')

    pbar.close()


if __name__ == '__main__':
    depth_expansion()
