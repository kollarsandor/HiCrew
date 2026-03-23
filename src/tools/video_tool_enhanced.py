#!/usr/bin/env python
# -*- coding: utf-8 -*-


from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import json
from typing import Type, Optional
from pathlib import Path
import cv2
from dashscope import MultiModalConversation


DATA_DIR = Path("/root/autodl-tmp/VideoTree")
SHORT_VIDEOS_DIR = DATA_DIR / "nextqa_video_short"
VIDEO_DIR = DATA_DIR / "data/nextqa/nextvideo"
SEGMENT_SUMMARIES_FILE = DATA_DIR / "data/segment_summaries.jsonl"


class AnalyzeSegmentVideoToolInput(BaseModel):
    video_uuid: str = Field(..., description="The UUID of the video (e.g., '2400171624')")
    segment_id: int = Field(..., description="The segment ID (1-based index) to analyze")
    prompt: str = Field(..., description="Prompt text to guide segment video analysis")



class AnalyzeSegmentVideoTool(BaseTool):

    name: str = "analyze_segment_video"
    description: str = """Analyze a video segment by segment_id.

This tool automatically retrieves the segment boundaries from segment_summaries.jsonl
and analyzes the corresponding video clip using VLM.

Input:
- video_uuid: Video identifier (e.g., "2400171624")
- segment_id: Segment index (1-based, e.g., 1 for first segment)
- prompt: Analysis prompt focused on the question

Use this when you need to visually analyze a specific segment of the video.

Returns: Detailed VLM analysis of the video segment.
"""
    args_schema: Type[BaseModel] = AnalyzeSegmentVideoToolInput

    _segment_data: dict = None

    def _load_segment_data(self) -> dict:
        """从 segment_summaries.jsonl 加载segment数据"""
        if self._segment_data is None:
            self._segment_data = {}
            if SEGMENT_SUMMARIES_FILE.exists():
                try:
                    with open(SEGMENT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                video_id = entry.get('video_id')
                                if video_id:
                                    self._segment_data[video_id] = entry.get('segments', [])
                except Exception as e:
                    print(f"Error loading segment summaries: {e}")
        return self._segment_data

    def _extract_video_clip(self, video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
        try:
            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                return False

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


            start_time = max(0, start_time)
            end_time = min(end_time, duration)

            if start_time >= end_time:
                print(f"time invalid: start={start_time}, end={end_time}")
                return False

            start_frame = round(start_time * fps)
            end_frame = round(end_time * fps)
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            cap.release()
            out.release()
            return True

        except Exception as e:
            print(f"error: {e}")
            return False

    def _run(self, video_uuid: str, segment_id: int, prompt: str) -> str:

        segment_data = self._load_segment_data()

        if video_uuid not in segment_data:
            return json.dumps({
                "error": f"No segment data found for video {video_uuid}",
                "video_uuid": video_uuid,
                "segment_id": segment_id
            })

        segments = segment_data[video_uuid]

        if segment_id >= len(segments):
            return json.dumps({
                "error": f"Segment {segment_id} not found. Video has {len(segments)} segments.",
                "video_uuid": video_uuid,
                "available_segments": len(segments)
            })

        segment_info = segments[segment_id]
        start_time = segment_info.get('start_time', 0)
        end_time = segment_info.get('end_time', 0)

        input_path = os.path.join(VIDEO_DIR, f"{video_uuid}.mp4")

        if not os.path.exists(input_path):
            return json.dumps({
                "error": f"Video file not found: {input_path}",
                "video_uuid": video_uuid
            })

        output_path = os.path.join(SHORT_VIDEOS_DIR, f"{video_uuid}_seg{segment_id}_{start_time}_{end_time}.mp4")

        if not self._extract_video_clip(input_path, start_time, end_time, output_path):
            return json.dumps({
                "error": "Failed to extract video clip",
                "video_uuid": video_uuid,
                "segment_id": segment_id,
                "time_range": [start_time, end_time]
            })

        try:
            local_video_path = f"file://{output_path}"

            completion = MultiModalConversation.call(
                model="qwen-vl-max",
                api_key="sk-3d2f2fcf8d3541cdaf9d646161407b3a",
                messages=[
                    {
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": f"""You are a video analysis assistant. Analyze this video segment carefully.
                            
Focus on: {prompt}

This is segment {segment_id} of the video, spanning from {start_time}s to {end_time}s.
Provide factual observations only. Do not hallucinate or make assumptions beyond what is visible."""
                        }]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"video": local_video_path},
                            {"text": f"Analyze this video segment ({start_time}s to {end_time}s). Focus on: {prompt}"}
                        ]
                    }
                ]
            )

            if completion and completion.get("output"):
                analysis = completion["output"]["choices"][0]["message"]["content"][0]["text"]
            else:
                analysis = "VLM analysis failed - no response received"

            result = {
                "video_uuid": video_uuid,
                "segment_id": segment_id,
                "time_range": {"start": start_time, "end": end_time},
                "analysis": analysis
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "error": f"VLM analysis failed: {str(e)}",
                "video_uuid": video_uuid,
                "segment_id": segment_id
            })
        finally:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass



