import json
import os
import cv2
from datetime import timedelta
import numpy as np

blk_size = 16

def edge_detector(ref):
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    high_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    low_thresh = 0.2 * high_thresh
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    mat = np.matrix(edges) / 255
    h, w = mat.shape
    blk_mean_arr = [mat[i:i + blk_size, j:j + blk_size].mean()
                    for i in range(0, h, blk_size)
                    for j in range(0, w, blk_size)]
    return blk_mean_arr

def edge_diff(ref_img_blk_mean_arr, curr_img, img_h, img_w):
    curr_img_blk_mean_arr = edge_detector(curr_img)
    diff = [abs(curr - ref) for curr, ref in zip(curr_img_blk_mean_arr, ref_img_blk_mean_arr)]
    diff2 = [round(val, 1) if val > 0.1 else 0 for val in diff]
    return round(sum(diff2), 2), curr_img_blk_mean_arr

def zero_runs(a):
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def cal_edge_diff2(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = int(total_frame / fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    scale_h = 128
    scale_w = int(size[0] * scale_h / size[1])

    ref_img_blk_mean_arr = None
    diff_val_arr = []

    for i in range(0, total_frame, fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        flag, frame = cap.read()
        if not flag:
            break
        frame = cv2.resize(frame, (scale_w, 128))
        if ref_img_blk_mean_arr is None:
            ref_img_blk_mean_arr = edge_detector(frame)
            diff_val_arr.append(0)
        else:
            diff_val, ref_img_blk_mean_arr = edge_diff(ref_img_blk_mean_arr, frame, scale_h, scale_w)
            diff_val_arr.append(0 if diff_val < 6 else diff_val)

    runs = zero_runs(diff_val_arr)
    shot_start_arr = [0 if i == 0 else runs[i][0] - 1 for i in range(len(runs))]
    shot_start_arr.append(total_sec)
    return shot_start_arr

def scene_check(video_path):
    shot_starts = cal_edge_diff2(video_path)
    segments = []
    for i in range(len(shot_starts) - 1):
        start = shot_starts[i]
        end = shot_starts[i + 1]
        segments.append({
            "uid": i + 1,
            "start_time": float(start),
            "end_time": float(end),
            "duration": float(end - start)
        })
    return segments

def merge_small_segments(segments):
    i = 0
    while i < len(segments) - 1:
        while segments[i]["duration"] < 10:
            segments[i]["end_time"] = segments[i + 1]["end_time"]
            segments[i]["duration"] = segments[i]["end_time"] - segments[i]["start_time"]
            del segments[i + 1]
        else:
            i += 1
    return segments

# 主逻辑
video_folder_path = "/workspace/media/test"
video_uids_in_folder = {
    filename[:-4] for filename in os.listdir(video_folder_path) if filename.endswith(".mp4")
}

output_data = {}

for video_uid in video_uids_in_folder:
    video_path = os.path.join(video_folder_path, f"{video_uid}.mp4")
    if not os.path.exists(video_path):
        print(f"Skipping {video_uid}: Video file does not exist.")
        continue

    segments = scene_check(video_path)
    merge_small_segments(segments)

    new_segments = []
    for segment in segments:
        if segment["duration"] > 20:
            num_splits = int(segment["duration"] // 10)
            for i in range(num_splits):
                start = segment["start_time"] + i * 10
                end = min(segment["start_time"] + (i + 1) * 10, segment["end_time"])
                new_segments.append({
                    "uid": len(new_segments) + 1,
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start
                })
        else:
            new_segments.append(segment)

    new_segments.sort(key=lambda x: x["start_time"])
    for idx, seg in enumerate(new_segments):
        seg["uid"] = idx + 1

    output_data[video_uid] = new_segments

with open("segment_only_pysecenedetect.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("All video segmentation results have been successfully saved.")
print("Saved to:", os.path.abspath("segment_only_pysecenedetect.json"))
