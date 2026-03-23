import cv2
from pathlib import Path
from tqdm import tqdm
import json
import os


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0.0
    cap.release()
    return duration


def extract_es():
    # /path/to/egoschema/videos
    input_base_path = Path('./Egoschema_videos/videos/videos')
    # /path/to/data/egoschema_frames
    output_base_path = Path('./egoschema_frames')
    fps = 1
    pbar = tqdm(total=len(list(input_base_path.iterdir())))
    for video_fp in input_base_path.iterdir():
        output_path = output_base_path / video_fp.stem
        file_count = 0
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isfile(item_path):
                file_count += 1
        if file_count >= 180:
            continue
        print(f"帧数量为{file_count}，不足继续执行")
        duration = get_video_duration(video_fp)
        print(f"视频长度为{duration}")
        vidcap = cv2.VideoCapture(str(video_fp))
        count = 0
        success = True
        fps_ori = int(vidcap.get(cv2.CAP_PROP_FPS))   
        frame_interval = int(1 / fps * fps_ori)
        while success:
            success, image = vidcap.read()
            if not success:
                break
            if count % (frame_interval) == 0 :
                cv2.imwrite(f'{output_path}/{count}.jpg', image)
            count+=1
        file_count = 0
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isfile(item_path):
                file_count += 1
        print(f"{video_fp.stem}帧数量为{file_count}")       
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    extract_es()