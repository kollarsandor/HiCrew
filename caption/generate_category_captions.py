import json
import os
import pandas as pd
from pathlib import Path
import base64
from openai import OpenAI
import cv2
import sys
import argparse

IMAGES = Path("/root/autodl-tmp/VideoTree/nextqa_frames")
VAL_CSV = Path("/root/autodl-tmp/VideoTree/data/nextqa/val.csv")
DUR = Path("/root/autodl-tmp/VideoTree/data/nextqa/durations.json")
VID = Path('/root/autodl-tmp/VideoTree/data/nextqa/nextvideo')
CATEGORY_QUESTIONS = Path("/root/autodl-tmp/VideoTree/data/video_category_questions.json")
OUTPUT = Path("/root/autodl-tmp/VideoTree/data/per_second_category_captions")
PROMPT_LOG = Path("/root/autodl-tmp/VideoTree/data/prompt.jsonl")
ERROR_LOG = Path("/root/autodl-tmp/VideoTree/data/err.jsonl")


def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(fn, data):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_vlm_prompt(category, questions, category_types, api_key):

    category_descriptions = {
        'causal': 'Causal questions - Focus on reasons (why) and methods (how) of actions',
        'temporal': 'Temporal questions - Focus on action sequences (before/after/current)',
        'descriptive': 'Descriptive questions - Focus on scene information (location/objects/counting)'
    }

    type_descriptions = {
        'CH': 'Causal How - How an action is performed',
        'CW': 'Causal Why - Why an action is performed',
        'TN': 'Temporal Next - What happens after',
        'TP': 'Temporal Previous - What happened before',
        'TC': 'Temporal Current - What is happening now',
        'DL': 'Descriptive Location - Where is this',
        'DO': 'Descriptive Object - Objects/relationships',
        'DC': 'Descriptive Counting - Count of objects/people'
    }

    questions_text = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])
    types_text = "\n".join([f"- {t}: {type_descriptions.get(t, t)}" for t in category_types])

    meta_prompt = f"""You are a prompt engineer. Your task is to generate a clear, focused instruction for a Vision-Language Model (VLM) to describe video frames.

**Context**:
- Category: {category} ({category_descriptions.get(category, category)})
- Question Types in this category:
{types_text}

**Questions that need to be answered** (all from the same video):
{questions_text}

**Your Task**:
Generate a concise instruction (3-5 bullet points) for the VLM to describe video frames in a way that helps answer these specific questions.

**Requirements**:
1. Focus on what visual information the VLM should prioritize
2. Be specific about details needed (e.g., "count people precisely", "describe action methods", "note temporal order")
3. Tailor to the actual questions - if questions ask about specific objects/actions, mention them
4. For causal category (why/how questions): **Describe only observable visual facts, do NOT make speculative inferences** 
5. Keep it actionable and clear
6. Output ONLY the instruction text, no meta-commentary
7. Be precise ！
**Output Format**:
Start with "**Focus on [CATEGORY] information:**" and then list 3-5 specific bullet points.

Example output:
**Focus on CAUSAL information:**
- Describe HOW the man demonstrates the watch (hand gestures, positioning)
- Note WHY people might be gathering (observable purposes)
- Capture action execution details that show intention
"""

    base_url = 'your url'
    model = "gpt-4o"

    client = OpenAI(api_key=api_key, base_url=base_url + '/v1')

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.7,
        )

        # 检查response
        if not response.choices:
            raise ValueError(f"API returned empty choices for category: {category}")

        message_content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        # 检查content是否为None
        if message_content is None:
            raise ValueError(f"API returned None content. Finish reason: {finish_reason}, Category: {category}")

        generated_prompt = message_content.strip()
        return generated_prompt

    except Exception as e:
        print(f"     ❌ Error generating VLM prompt for {category}: {str(e)}")
        raise


def generate_category_caption(img_path, category, vlm_instruction, api_key, current_second, total_seconds):

    with open(img_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    base_url = 'https://opus.gptuu.com'
    model = "gpt-4o"

    client = OpenAI(api_key=api_key, base_url=base_url + '/v1')

    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Describe this video frame for question-answering purposes.

**Temporal Context**: This frame is at {current_second}s / {total_seconds}s (total video duration).

{vlm_instruction}

Provide 1-2 sentences. Be objective and precise. Focus only on clearly visible elements without speculation (avoid 'might', 'seems', 'probably'...). Do not mention what is absent."""},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=0.3,
    )
    return chat_completion.choices[0].message.content


def main(start_idx, end_idx, api_key):

    print(f"[Worker {start_idx}-{end_idx}] Loading data...")
    durations = load_json(DUR)
    category_questions_map = load_json(CATEGORY_QUESTIONS)
    df = pd.read_csv(VAL_CSV)
    subset_video_ids = list(df['video'].astype(str).unique())

    os.makedirs(OUTPUT, exist_ok=True)

    target_videos = subset_video_ids[start_idx:end_idx]
    processed_count = 0

    print(f"[Worker {start_idx}-{end_idx}] Processing {len(target_videos)} videos...")

    for video_id in target_videos:
        print(f"\n{'=' * 60}")
        print(f"Processing video: {video_id}")

        try:
            if video_id not in category_questions_map:
                print(f"  ⚠️  No questions found, skipping...")
                continue

            output_file = OUTPUT / f"{video_id}.json"
            if output_file.exists():
                print(f"  ✓ Already processed, skipping...")
                continue

            if video_id not in durations:
                print(f"  ⚠️  No duration data, skipping...")
                continue

            video_len = durations[video_id]
            video_path = VID / f"{video_id}.mp4"

            if not video_path.exists():
                print(f"  ⚠️  Video file not found, skipping...")
                continue

            vidcap = cv2.VideoCapture(str(video_path))
            fps_ori = int(vidcap.get(cv2.CAP_PROP_FPS))
            vidcap.release()

            print(f"  Duration: {video_len}s, FPS: {fps_ori}")

            video_data = category_questions_map[video_id]
            categories = video_data.get('categories', [])

            print(f"  Categories: {', '.join(categories)}")
            print(f"  Will generate {len(categories)} captions per second")

            category_vlm_prompts = {}
            print(f"\n  🔧 Generating VLM prompts for each category...")

            for category in categories:
                category_data = video_data[category]
                questions = category_data['questions']
                types = category_data['types']

                print(f"     Generating prompt for {category}...")
                vlm_prompt = generate_vlm_prompt(category, questions, types, api_key)
                category_vlm_prompts[category] = vlm_prompt
                print(f"     ✓ {category}: {vlm_prompt[:80]}...")

            prompt_log_entry = {
                'video_id': video_id,
                'categories': categories,
                'prompts': category_vlm_prompts,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            with open(PROMPT_LOG, 'a', encoding='utf-8') as f:
                json.dump(prompt_log_entry, f, ensure_ascii=False)
                f.write('\n')
            print(f"  📝 Logged prompts to {PROMPT_LOG}")

            per_second_captions = {}

            for second in range(0, video_len, 2):
                frame_num = second * fps_ori
                frame_path = IMAGES / f'{video_id}/{frame_num}.jpg'

                if not frame_path.exists():
                    print(f"  ⚠️  Frame {second}s not found, skipping...")
                    continue

                second_captions = {}

                for category in categories:
                    vlm_instruction = category_vlm_prompts[category]

                    caption = generate_category_caption(
                        frame_path,
                        category,
                        vlm_instruction,
                        api_key,
                        current_second=second,
                        total_seconds=video_len
                    )

                    second_captions[category] = caption

                per_second_captions[str(second)] = second_captions


                if second % 10 == 0:
                    print(f"  Progress: {second}/{video_len}s")

            result = {
                video_id: per_second_captions
            }


            save_json(output_file, result)
            processed_count += 1
            print(f"\n  ✅ Saved {len(per_second_captions)} seconds × {len(categories)} categories")
            print(f"     Total captions: {len(per_second_captions) * len(categories)}")

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"  ❌ Error processing video {video_id}: {error_msg}")

            error_log_entry = {
                'video_id': video_id,
                'error': error_msg,
                'error_type': error_type,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            with open(ERROR_LOG, 'a', encoding='utf-8') as f:
                json.dump(error_log_entry, f, ensure_ascii=False)
                f.write('\n')
            print(f"  📝 Logged error to {ERROR_LOG}")

            continue

    print(f"\n{'=' * 60}")
    print(f"[Worker {start_idx}-{end_idx}] ✅ Processing complete! Processed {processed_count} videos")
    print(f"📁 Output directory: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate category captions for videos')
    parser.add_argument('--start', type=int, required=True, help='Start video index')
    parser.add_argument('--end', type=int, required=True, help='End video index')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')

    args = parser.parse_args()
    main(args.start, args.end, args.api_key)
