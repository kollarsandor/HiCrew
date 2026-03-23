import json
import os
from datetime import timedelta
import cv2
from sbd import cal_edge_diff2

##--------------------------文件加载---------------------------------
# 指定视频文件夹路径
video_folder_path = "./Egoschema_videos"

# 获取视频文件夹中的 MP4 文件名（去掉 .mp4 后缀）
video_uids_in_folder = {
    filename[:-4] for filename in os.listdir(video_folder_path) if filename.endswith(".mp4")
}
# # 加载 JSON 数据（转换为字典）
# with open("./data/egoschema/blip2_fullset.json", "r", encoding="utf-8") as f:
#     frame_data = json.load(f)
# # print(frame_data[video_uids_in_folder])
# print({uid: frame_data.get(uid) for uid in video_uids_in_folder})


##--------------------------短片段合并函数---------------------------------
def merge_small_segments(segments):
    i = 0
    while i < len(segments) - 1:
        # 检查当前片段与下一个片段合并后的时长
        while i < len(segments) - 1 and segments[i]["duration"]< 10 and segments[i + 1]["duration"]< 10:
            # 合并当前片段和下一个片段
            segments[i]["end_time"] = segments[i + 1]["end_time"]
            segments[i]["duration"] = segments[i]["end_time"] - segments[i]["start_time"]
            # 删除下一个片段
            del segments[i + 1]
        else:
            i += 1
    return segments
##--------------------------场景检测函数---------------------------------
def scene_check(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_uid = os.path.splitext(os.path.basename(video_path))[0]
    shot_start_arr = cal_edge_diff2(video_path[:-4])  # 去掉.mp4后缀传给sbd

    segments = []
    for i in range(len(shot_start_arr) - 1):
        start_time = shot_start_arr[i]
        end_time = shot_start_arr[i + 1]
        duration = end_time - start_time
        segment = {
            "uid": int(i + 1),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "duration": float(duration)
        }

        segments.append(segment)
    return segments

##--------------------------时间格式转换---------------------------------
def seconds_to_time_format(seconds):
    return str(timedelta(seconds=seconds))

##--------------------------视频按场景分割---------------------------------
# 处理符合条件的视频
output_data = {}
# 按场景检测并分析每个视频
for video_uid in video_uids_in_folder:
    video_path = os.path.join(video_folder_path, f"{video_uid}.mp4")
    if not os.path.exists(video_path):
        print(f"Skipping {video_uid}: Video file does not exist.")
        continue
    #------基础分割------#
    segments = scene_check(video_path)
    #------短片段合并------#
    merge_small_segments(segments)
    #------长片段拆分------#
    new_segments = []
    for segment in segments:
        if segment["duration"] > 20:
            # 计算分割成的子片段数
            num_splits = int(segment["duration"]) // 10
            sublen = int(segment["duration"]) / num_splits
            for i in range(num_splits):
                start_time = segment["start_time"] + i * sublen
                if i<num_splits-1:
                    end_time = segment["start_time"] + (i + 1) * sublen
                else:
                    end_time = segment["end_time"]
                new_segments.append({
                    "uid": len(new_segments) + 1,  # 新的片段UID
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time
                })
        else:
            new_segments.append(segment)  # 如果不是长片段，则保持原样

    segments = new_segments

    # 按开始时间对片段排序并更新uid
    segments.sort(key=lambda x: x["start_time"])
    for idx, segment in enumerate(segments):
        segment["uid"] = idx + 1  # 根据排序重新生成uid
    output_data[video_uid] = segments

# 保存所有视频片段的 JSON 文件
with open("segment_sbd.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("All video segmentation results have been successfully saved.")
