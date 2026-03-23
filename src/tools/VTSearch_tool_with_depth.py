import json
import os
import cv2
from pathlib import Path
import pandas as pd

DATA_DIR = Path("/root/autodl-tmp/VideoTree")
ANNO_PATH = DATA_DIR / "data/nextqa/val.csv"
RELEVANCE_PATH = DATA_DIR / "dynamic_width_expansion/relevance_score.json"
WIDTH_RES_PATH = DATA_DIR / "dynamic_width_expansion/width_res.json"
DEPTH_RES_PATH = DATA_DIR / "dynamic_width_expansion/depth_expansion_res.json"
SEGMENT_SBD_PATH = DATA_DIR / "sbd_new/nextqa_segment_sbd.json"
VIDEO_DIR = DATA_DIR / "data/nextqa/nextvideo"
FRAMES_DIR = DATA_DIR / "nextqa_frames"
VIDEOS_FRAMES_CAPTIONS_DIR = DATA_DIR /"data"/ "nextqa"/"llava1.5_fps1.json"
DURATION = DATA_DIR / "data/nextqa/durations.json"
def load_json(path):

    with open(path, 'r') as f:
        return json.load(f)


def getVideoQA(video_uid):

    df = pd.read_csv('/root/autodl-tmp/VideoTree/data/nextqa/val.csv')
    videos=list(df['video'].astype(str).unique())
    video_id, qid = video_uid.split("_")

    if video_id not in videos:
        return f"can't find {video_uid} in the dataset"

    df = pd.read_csv('/root/autodl-tmp/VideoTree/data/nextqa/val.csv')
    qa_data = df[(df['video'] == int(video_id)) & (df['qid'] == int(qid))].iloc[0]
    question = qa_data["question"]
    options = [qa_data[f"a{i}"] for i in range(5)]

    result = f"question: {question}\n\noptions:\n"
    for i, option in enumerate(options):
        result += f"{chr(65 + i)}: {option}\n"
    print(result)
    return result


def getMostRelevant(video_uid):

    relevance_data = load_json(RELEVANCE_PATH)
    segment_data = load_json(SEGMENT_SBD_PATH)
    caption_data = load_json(VIDEOS_FRAMES_CAPTIONS_DIR)
    depth_data = load_json(DEPTH_RES_PATH)  # 2

    if video_uid not in relevance_data:
        return {"error": f"未找到视频 {video_uid} 的相关度数据"}

    video_segments = relevance_data[video_uid]["segments"]
    video_segment_times = segment_data[video_uid]

    segment_scores = {}
    for seg_id, seg_info in video_segments.items():
        pred_scores = seg_info["pred"]
        avg_score = sum(pred_scores) / len(pred_scores) if pred_scores else 0
        segment_scores[seg_id] = avg_score

    high_relevance_count = sum(1 for score in segment_scores.values() if score >= 2.5)
    total_segments = len(segment_scores)
    is_global = high_relevance_count >= (total_segments * 0.4)

    high_relevance_frames = {}
    if is_global:
        for seg_id, seg_info in video_segments.items():
            int_seg_id = int(seg_id)

            seg_time = next((s for s in video_segment_times if s["uid"] == int_seg_id), None)
            if not seg_time:
                continue

            start_time = seg_time["start_time"]
            center_frames = seg_info["tree_node"][1]
            pred_scores = seg_info["pred"] if seg_info["pred"] else [1]

            for i, score in enumerate(pred_scores):
                if score == 3 and i < len(center_frames):
                    center_idx = center_frames[i]
                    frame_second = int(start_time + center_idx)
                    frame_num = frame_second * 30
                    frame_path = str(f"{video_uid}/{frame_num}.jpg")
                    frame_info = []
                    frame_info.append(frame_path)
                    frame_info.append(caption_data[video_uid][frame_second])
                    high_relevance_frames[frame_second] = frame_info
    else:
        video_depth = {}
        for item in depth_data:
            if video_depth.get(item['name']):
                video_depth[item['name']][item["segment"]] = item['sorted_values']
            else:
                video_depth[item['name']] = {item["segment"]: item['sorted_values']}
        if video_uid not in video_depth:
            return {"error": f"未找到视频 {video_uid} 的深度扩展数据"}
        for seg_id, seg_info in video_segments.items():
            int_seg_id = int(seg_id)

            seg_time = next((s for s in video_segment_times if s["uid"] == int_seg_id), None)
            if not seg_time:
                continue
            start_time = seg_time["start_time"]
            center_frames = video_depth[video_uid][int_seg_id]
            if not center_frames:
                continue
            for frame_second in center_frames:
                frame_num = int(frame_second) * 30
                frame_path = str(f"{video_uid}/{frame_num}.jpg")
                frame_info = []
                frame_info.append(frame_path)
                frame_info.append(caption_data[video_uid][int(frame_second)])
                high_relevance_frames[frame_second] = frame_info
    result = {
        "is_global_question": is_global,
        "relevant_frames": high_relevance_frames
    }

    frame_times = list(high_relevance_frames.keys())
    if frame_times:
        center_time = int(sum(frame_times) / len(frame_times))
        containing_segment = next(
            (s for s in video_segment_times if s["start_time"] <= center_time < s["end_time"]), None
        )
        if containing_segment:
            clip_start = max(containing_segment["start_time"], center_time - 10)
            clip_end = min(containing_segment["end_time"], center_time + 10)
            if clip_end - clip_start > 20:
                midpoint = (clip_start + clip_end) / 2
                clip_start = midpoint - 10
                clip_end = midpoint + 10

            video_clip_info = {
                "video_uid": video_uid,
                "start_time": clip_start,
                "end_time": clip_end,
                "output_path": str(SHORT_VIDEOS_DIR / f"{video_uid}.mp4")
            }

            video_path = str(VIDEO_DIR / f"{video_uid}.mp4")
            output_path = str(video_clip_info["output_path"])
            success = extract_video_clip(video_path, clip_start, clip_end, output_path)

            if success:
                result["video_clip"] = video_clip_info

    return result


def extract_video_clip(video_path, start_time, end_time, output_path):

    try:
        if not os.path.exists(video_path):
            print(f"not exit: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"can't open: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if start_time < 0 or end_time > duration or start_time >= end_time:
            print(f"invalid: start_time={start_time}, end_time={end_time}, duration={duration}")
            return False

        start_frame = round(start_time * fps)
        end_frame = round(end_time * fps)

        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                print("fail to read frame")
                break
            out.write(frame)

        cap.release()
        out.release()

        print("succeed to extract clip")
        return True
    except Exception as e:
        print(f"error: {e}")
        return False

