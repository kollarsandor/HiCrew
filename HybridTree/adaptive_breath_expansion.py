# 这个加上了是空的处理逻辑
# 2025.7.24只跑全集中不在子集的部分
import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
from kmeans_pytorch import kmeans
from model2 import get_model
from prompts import PromptFactory
from dataset import get_dataset
from eval import *
from util import *

def load_frame_features(name_ids, save_folder):
    filename = f"{name_ids}.pt"
    filepath = os.path.join(save_folder, filename)
    img_feats = torch.load(filepath, weights_only=True)
    return img_feats

def load_caption_data(caption_file_path):
    with open(caption_file_path, "r", encoding="utf-8") as f:
        caption_data = json.load(f)
    return caption_data
# 修改为只查找一个聚类的代表帧
def find_closest_points(x, cluster_ids, cluster_centers):
    closest_points = []
    for cluster_id in range(len(cluster_centers)):
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0]
        points_in_cluster = x[indices_in_cluster]
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1)
        if distances.numel() > 0:
            closest_idx_in_cluster = torch.argmin(distances).item()
            closest_global_idx = indices_in_cluster[closest_idx_in_cluster].item()
            closest_points.append(closest_global_idx)
    return closest_points

def launch():
    args = parse_args()
    pprint(args)

    # 结果存储路径
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)
    output_width_res_path = os.path.join(args.output_base_path, "width_res_2.json")
    frame_feat_path = args.frame_feat_path
    # args.caption_file_path
    caption_file_path = "./data/egoschema/blip2_fullset.json"  # Caption 文件路径

    # 读取分段数据和caption数据
    with open("segment_sbd.json", "r", encoding="utf-8") as f:
        segmented_data = json.load(f)

    caption_data = load_caption_data(caption_file_path)
    
    # 新增：读取subset_anno.json，获取需要处理的视频ID集合
    with open("data/egoschema/subset_anno.json", "r", encoding="utf-8") as f:
        subset_anno = json.load(f)
    subset_video_ids = set(subset_anno.keys())
    # 新增：读取已经跑过的视频，获取需要处理的视频ID集合
    with open("dynamic_width_expansion/relevance_score.json", "r", encoding="utf-8") as f:
        reached = json.load(f)
    reached_video_ids = set(reached.keys())
    # 处理已存在的结果
    processed = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if 'data' in processed:
            processed = processed['data']

    # 获取数据集
    quids_to_exclude = set(processed.keys())
    dataset = get_dataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=args.num_examples_to_run)

    # 配置 Prompt 生成器
    prompter = PromptFactory().get(args.prompt_type)

    # 加载模型
    model = get_model(args)
    model.set_post_process_fn(prompter.post_process_fn)

    # 结果保存
    all_width_res = []

    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        video_uid = item.get("quid", item.get("uid"))
        
        #这里跑的是子集，全集把下面三行代码注释掉
        # 新增：只处理subset_anno.json中有的视频
        # if video_uid not in subset_video_ids:
        #     print(f"Skipping {video_uid}: Not in subset_anno.json.")
        #     continue
        if video_uid in reached_video_ids:
            print(f"Skipping {video_uid}: Already in subset_anno.json.")
            pbar.update(1)
            continue

        if video_uid not in segmented_data:
            print(f"Skipping {video_uid}: No segmentation data found.")
            continue
        print(video_uid)
        # 加载视频的帧特征
        frame_feats = load_frame_features(video_uid, frame_feat_path)
        print(segmented_data[video_uid])
        # 逐个处理每个分段
        for segment in segmented_data[video_uid]:
            start_frame, end_frame = int(segment["start_time"]), int(segment["end_time"])
            print("segment")
            # 获取该分段的帧特征
            segment_feats = frame_feats[start_frame:end_frame + 1]
  
            if segment_feats.shape[0] == 0:
                print(f"Skipping segment {segment['uid']} of {video_uid}: No frames available.")
                continue
            print("here")
            # 使用K-means聚类
            cluster_ids_x, cluster_centers = kmeans(
                X=segment_feats,
                num_clusters=2,  # 每个分段仅一个聚类中心
                distance="cosine",
                device=torch.device("cuda:0")
            )
            # 取出该分段聚类编号和聚类中心
            cluster_ids_x = cluster_ids_x.to("cuda")
            cluster_centers = cluster_centers.to("cuda")

            # 找到该分段的代表性帧索引
            closest_points = find_closest_points(segment_feats, cluster_ids_x, cluster_centers)
            tree_node = sorted([i for i in closest_points])

            # 获取每帧的描述
            segment_caption = caption_data.get(video_uid, [])
            segment_caption_text = " ".join(segment_caption[start_frame:end_frame + 1])  # 获取对应分段的描述

            # 生成 Prompt
            prompt = prompter.fill(
                **item,
                fps=args.fps,
                clip_length=int(1 / args.fps) if args.fps < 1 else 1 / args.fps,
                num_words=args.num_words_in_sum,
                examplars=[],
                # loc_pred = tree_node
                loc_pred=(segment["uid"], tree_node),
                segment_caption=segment_caption_text  # 将描述传递给Prompt
            )
            print("prompt OK")
            pred, info = model.forward(prompter.head, prompt)
            ukey_name = "quid" if "quid" in item else "uid"

            # 存储结果
            processed.setdefault(video_uid, {})
            processed[video_uid].setdefault("segments", {})
            processed[video_uid]["segments"][segment["uid"]] = {
                "tree_node": (segment["uid"], tree_node),
                "prompt": prompt,
                "response": info["response"],
                "pred": pred
            }

            all_width_res.append({
                "video_uid": video_uid,
                "tree_node": (segment["uid"], tree_node),
                "cluster_ids_x": cluster_ids_x.tolist()
            })

        if i % 10 == 0:
            save_json(all_width_res, output_width_res_path)
            save_json(processed, output_path)
        pbar.update(1)
    # 保存所有视频的结果
    save_json(all_width_res, output_width_res_path)
    save_json(processed, output_path)

if __name__ == "__main__":
    launch()
