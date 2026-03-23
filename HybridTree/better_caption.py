import json
import os
import pandas as pd
from pathlib import Path
import VideoCrew.videoanalyze_image.src.videoanalyze.tools.VTSearch_tool_with_depth as vts
import base64
import requests
from openai import OpenAI

IMAGES = Path("/root/autodl-tmp/VideoTree/egoschema_frames")
SUB = Path("/root/autodl-tmp/VideoTree/data/egoschema/subset_anno.json")
ANNO_PATH = Path("/root/autodl-tmp/VideoTree/data/egoschema/fullset_anno.json")
RES = Path("/root/autodl-tmp/good_cap")

def load_video_ids(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        video_ids = [item['q_uid'] for item in data.values()]
        return video_ids
    except Exception as e:
        return []

def getKeyWords(video_uid):
    qa = vts.getVideoQA(video_uid)
    base_url = 'https://opus.gptuu.com'
    api_key = "sk-46KivuHvn6nIN1r6LlUr8B3k781oB3VaLZif2UIugY1BmqRq"
    model = "gpt-4o"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url + '/v1'
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"Extract keywords from this text: \n {qa}. \n This is a set of video Q&A questions and options. I only want vocabulary that might be related to the actual video content."}
                ]
            }
        ],
        temperature=0.7,
    )
    return chat_completion.choices[0].message.content

def getCaption(img,key_words):
    print(str(img))
    with open(img, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    base_url = 'https://opus.gptuu.com'
    api_key = "sk-46KivuHvn6nIN1r6LlUr8B3k781oB3VaLZif2UIugY1BmqRq"
    model = "gpt-4o"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url + '/v1'
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You must describe the image objectively in 1-2 sentences. Focus only on clearly visible elements without speculation (avoid 'might', 'seems', 'probably'). Do not mention what is absent. Prioritize using keywords from {key_words} but do not necessarily need to use all of them. Keep the description accurate and concise, omitting non-essential background details."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.5,
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    videos = load_video_ids(ANNO_PATH)
    count = 4
    better_captions = {videos[i]: {} for i in range(len(videos))}
    reached = load_video_ids(SUB)
    df = pd.read_json("cap_err.jsonl", lines=True)
    reached = reached + list(df.columns)
    df = pd.read_json("retry_depth_cap.jsonl", lines=True)
    reached = reached + list(df.columns)
    for i, video_uid in enumerate(videos):
        if video_uid in reached:
            continue
        try:
            # 调用 getMostRelevant 获取最相关内容
            relevant_result = vts.getMostRelevant(video_uid)
            if relevant_result["is_global_question"]:
                continue
            # 获得关键词
            key_words = getKeyWords(video_uid)
            print("关键词：" + key_words)
            count+=1
            if "error" in relevant_result:
                print(relevant_result["error"])
            else:
                # 处理空的情况
                if relevant_result["relevant_frames"]:
                    with open(f"./good_cap/{video_uid}.json", 'r', encoding = "utf-8") as f:
                        better_captions[video_uid]=json.load(f)
                        print(better_captions[video_uid])
                    for second in relevant_result["relevant_frames"].keys():
                        better_captions[video_uid][int(second)] = getCaption(IMAGES/f'{video_uid}/{str(int(second)*30)}.jpg', key_words)
                        print(f'视频{video_uid}的{int(second)*30}帧的字幕:{better_captions[video_uid][int(second)]}')
            # for i in range(0,180,10):
            #     if not better_captions[video_uid].get(i):
            #         better_captions[video_uid][i] = getCaption(IMAGES / f'{video_uid}/{str(i * 30)}.jpg', key_words)
            #         print(f'视频{video_uid}的{i * 30}帧的字幕:{better_captions[video_uid][i]}')
            # if not better_captions[video_uid].get(179):
            #     better_captions[video_uid][179] = getCaption(IMAGES / f'{video_uid}/{str(179 * 30)}.jpg', key_words)
            #     print(f'视频{video_uid}的{179 * 30}帧的字幕:{better_captions[video_uid][179]}')
            with open(f"./good_cap/{video_uid}.json", 'w', encoding = "utf-8") as f:
                json.dump(better_captions[video_uid], f)
            with open("retry_depth_cap.jsonl", 'a', encoding = "utf-8") as f:
                json.dump({video_uid: count}, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
        except Exception as e:
            with open("cap_err.jsonl", 'a', encoding = "utf-8") as f:
                json.dump({video_uid: str(e)}, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
            continue