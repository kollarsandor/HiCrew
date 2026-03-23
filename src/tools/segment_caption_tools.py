#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segment Caption Tools
"""

import json
from pathlib import Path
from typing import Type, Dict, Any, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


DATA_DIR = Path("/root/autodl-tmp/VideoTree/data")
PER_SECOND_CAPTIONS_DIR = DATA_DIR / "per_second_category_captions"
SEGMENT_SUMMARIES_FILE = DATA_DIR / "segment_summaries.jsonl"



class GetAvailableTimestampsInput(BaseModel):

    video_id: str = Field(..., description="Video ID (e.g., '2400171624')")


class GetCaptionTimestampsInput(BaseModel):

    video_id: str = Field(..., description="Video ID (e.g., '2400171624')")
    second: int = Field(..., description="The second timestamp to get caption for (e.g., 10 for 10th second)")
    caption_type: str = Field(
        default="all",
        description="Caption type: 'causal', 'temporal', 'descriptive', or 'all' for all types"
    )


class GetSegmentCaptionTypeInput(BaseModel):

    video_id: str = Field(..., description="Video ID (e.g., '2400171624')")
    segment_id: int = Field(..., description="Segment ID (1-based index)")
    caption_type: str = Field(
        ...,
        description="Caption type to retrieve: 'causal', 'temporal', or 'descriptive'"
    )


class AnalyzeSegmentVideoInput(BaseModel):

    video_id: str = Field(..., description="Video ID (e.g., '2400171624')")
    segment_id: int = Field(..., description="Segment ID (1-based index) to analyze")
    prompt: str = Field(..., description="Analysis prompt/question for the video segment")


class GetAvailableTimestampsTool(BaseTool):

    name: str = "get_available_timestamps"
    description: str = """Get all available caption timestamps for a specific video.

Use this tool FIRST before using get_caption_timestamps to know which seconds have captions available.
This helps you understand the video's caption coverage and choose appropriate timestamps to query.

Input:
- video_id: The video identifier (e.g., "2400171624")

Returns: A list of all available seconds that have captions, along with the total count and time range.
"""
    args_schema: Type[BaseModel] = GetAvailableTimestampsInput

    def _run(self, video_id: str) -> str:

        pure_video_id = video_id.rsplit('_', 1)[0] if '_' in video_id else video_id

        caption_file = PER_SECOND_CAPTIONS_DIR / f"{pure_video_id}.json"

        if not caption_file.exists():
            return json.dumps({
                "error": f"Caption file not found for video {pure_video_id}",
                "video_id": pure_video_id,
                "available_timestamps": []
            })

        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # {video_id: {second: {causal:..., temporal:..., descriptive:...}}}
            video_data = data.get(pure_video_id, {})

            if not video_data:
                return json.dumps({
                    "error": f"No caption data found for video {pure_video_id}",
                    "video_id": pure_video_id,
                    "available_timestamps": []
                })

            available_seconds = sorted([int(s) for s in video_data.keys()])

            result = {
                "video_id": pure_video_id,
                "total_timestamps": len(available_seconds),
                "min_second": min(available_seconds) if available_seconds else None,
                "max_second": max(available_seconds) if available_seconds else None,
                "available_timestamps": available_seconds,
                "caption_types_available": ["causal", "temporal", "descriptive"]
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "error": f"Error reading caption file: {str(e)}",
                "video_id": pure_video_id,
                "available_timestamps": []
            })


class GetCaptionTimestampsTool(BaseTool):

    name: str = "get_caption_timestamps"
    description: str = """Get detailed caption description for a specific second in a video.

Use this tool to retrieve pre-generated captions for any timestamp in the video.
The captions include three types of information:
- causal: Information about WHY and HOW actions happen
- temporal: Information about action sequences and timing
- descriptive: Information about scene, objects, and counts

Input:
- video_id: The video identifier (e.g., "2400171624")
- second: The timestamp in seconds (e.g., 10 for the 10th second)
- caption_type: "causal", "temporal", "descriptive", or "all"

Returns: Caption text for the specified timestamp and type.
"""
    args_schema: Type[BaseModel] = GetCaptionTimestampsInput

    def _run(self, video_id: str, second: int, caption_type: str = "all") -> str:

        pure_video_id = video_id.rsplit('_', 1)[0] if '_' in video_id else video_id

        caption_file = PER_SECOND_CAPTIONS_DIR / f"{pure_video_id}.json"

        if not caption_file.exists():
            return json.dumps({
                "error": f"Caption file not found for video {pure_video_id}",
                "video_id": pure_video_id,
                "requested_second": second
            })

        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # {video_id: {second: {causal:..., temporal:..., descriptive:...}}}
            video_data = data.get(pure_video_id, {})

            if not video_data:
                return json.dumps({
                    "error": f"No caption data found for video {pure_video_id}",
                    "video_id": pure_video_id,
                    "requested_second": second
                })

            requested_second = second
            second_str = str(second)
            actual_second = None

            if second_str in video_data:
                actual_second = second
            else:

                available_seconds = sorted([int(s) for s in video_data.keys()])

                if available_seconds:

                    closest = min(available_seconds, key=lambda x: abs(x - second))
                    actual_second = closest
                    second_str = str(closest)

            if actual_second is None or second_str not in video_data:
                return json.dumps({
                    "error": f"No caption found near second {second}",
                    "video_id": pure_video_id,
                    "requested_second": second,
                    "available_seconds_sample": list(video_data.keys())[:10]
                })

            caption_data = video_data[second_str]

            used_different_second = (actual_second != requested_second)

            if caption_type == "all":
                result = {
                    "video_id": pure_video_id,
                    "requested_second": requested_second,
                    "actual_second": actual_second,
                    "second_note": f"Using caption from {actual_second}s (requested {requested_second}s)" if used_different_second else f"Exact match at {actual_second}s",
                    "captions": caption_data
                }
            else:
                caption_text = caption_data.get(caption_type, f"No {caption_type} caption available")
                result = {
                    "video_id": pure_video_id,
                    "requested_second": requested_second,
                    "actual_second": actual_second,
                    "second_note": f"Using caption from {actual_second}s (requested {requested_second}s)" if used_different_second else f"Exact match at {actual_second}s",
                    "caption_type": caption_type,
                    "caption": caption_text
                }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps({
                "error": f"Error reading caption: {str(e)}",
                "video_id": pure_video_id,
                "requested_second": second
            })


class GetSegmentCaptionTypeTool(BaseTool):

    name: str = "get_segment_caption_type"
    description: str = """Get a specific type of caption summary for a video segment.

Use this tool to retrieve segment-level summaries that provide overview information.
Each segment has three types of summaries:
- causal: Summary focusing on cause-effect relationships in the segment
- temporal: Summary focusing on temporal sequence of events
- descriptive: Summary of visual elements, objects, and scene

Input:
- video_id: The video identifier (e.g., "2400171624")
- segment_id: The segment index (0-based, e.g., 0 for first segment)
- caption_type: "causal", "temporal", or "descriptive"

Returns: The summary text for the specified segment and type, including time range.
"""
    args_schema: Type[BaseModel] = GetSegmentCaptionTypeInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用实例变量而非类变量，避免跨实例污染
        self._segment_data: Dict[str, Any] = {}

    def _load_segment_summaries(self) -> Dict[str, Any]:
        """加载segment summaries数据（带缓存）- 支持新的 jsonl 结构"""
        if not self._segment_data:
            if SEGMENT_SUMMARIES_FILE.exists():
                try:
                    with open(SEGMENT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                video_id = entry.get('video_id')
                                if video_id:
                                    # 新结构: {"video_id": "xxx", "segments": [...]}
                                    self._segment_data[video_id] = entry.get('segments', [])
                except Exception as e:
                    print(f"Error loading segment summaries: {e}")
        return self._segment_data

    def _run(self, video_id: str, segment_id: int, caption_type: str) -> str:
        pure_video_id = video_id.rsplit('_', 1)[0] if '_' in video_id else video_id

        segment_data = self._load_segment_summaries()

        segments = segment_data.get(pure_video_id, [])

        if not segments:
            return json.dumps({
                "error": f"No segment data found for video {pure_video_id}",
                "video_id": pure_video_id,
                "segment_id": segment_id
            })

        # segment_id 是 0-based，对应 segments 列表索引
        if segment_id >= len(segments):
            return json.dumps({
                "error": f"Segment {segment_id} not found. Video has {len(segments)} segments.",
                "video_id": pure_video_id,
                "available_segments": len(segments)
            })

        segment_info = segments[segment_id]

        time_range = {
            "start": segment_info.get('start_time', 'unknown'),
            "end": segment_info.get('end_time', 'unknown')
        }

        summaries = segment_info.get('summaries', {})
        caption_text = summaries.get(caption_type, f"No {caption_type} summary available")

        result = {
            "video_id": pure_video_id,
            "segment_id": segment_id,
            "segment_uid": segment_info.get('uid', segment_id),
            "caption_type": caption_type,
            "time_range": time_range,
            "summary": caption_text
        }

        return json.dumps(result, ensure_ascii=False)


class SegmentKnowledgeManager:

    QUESTION_TYPE_MAPPING = {
        # Causal
        'CH': 'causal',   # Causal How
        'CW': 'causal',   # Causal Why
        # Temporal
        'TN': 'temporal', # Temporal Next
        'TC': 'temporal', # Temporal Current
        'TP': 'temporal', # Temporal Previous
        # Descriptive
        'DL': 'descriptive', # Descriptive Location
        'DO': 'descriptive', # Descriptive Object
        'DC': 'descriptive', # Descriptive Counting
    }

    def __init__(self):
        self._segment_data: Dict[str, List[Dict[str, Any]]] = {}
        self._loaded = False

    def _load_data(self):

        if self._loaded:
            return

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

        self._loaded = True

    def get_knowledge_source_content(self, video_id: str, question_type: str) -> str:
        self._load_data()

        # 提取纯视频ID
        pure_video_id = video_id.rsplit('_', 1)[0] if '_' in video_id else video_id

        caption_type = self.QUESTION_TYPE_MAPPING.get(question_type, 'descriptive')

        segments = self._segment_data.get(pure_video_id, [])

        if not segments:
            return f"No segment information available for video {pure_video_id}"

        content_parts = [
            f"=== SEGMENT KNOWLEDGE SOURCE ===",
            f"Video ID: {pure_video_id}",
            f"Question Type: {question_type} (Focus: {caption_type})",
            f"Total Segments: {len(segments)}",
            f"",
            f"--- Segment Summaries ({caption_type} focus) ---"
        ]

        for i, segment_info in enumerate(segments):
            summaries = segment_info.get('summaries', {})

            start_time = segment_info.get('start_time', 'unknown')
            end_time = segment_info.get('end_time', 'unknown')
            duration = segment_info.get('duration', 'unknown')

            main_summary = summaries.get(caption_type, "No summary available")

            content_parts.append(f"")
            content_parts.append(f"[Segment {i}] ({start_time}s - {end_time}s, duration: {duration}s)")
            content_parts.append(f"  {caption_type.upper()}: {main_summary}")

        content_parts.append(f"")
        content_parts.append(f"=== END SEGMENT KNOWLEDGE ===")

        return "\n".join(content_parts)

    def get_all_segments_info(self, video_id: str) -> List[Dict[str, Any]]:

        self._load_data()

        pure_video_id = video_id.rsplit('_', 1)[0] if '_' in video_id else video_id

        segments = self._segment_data.get(pure_video_id, [])

        result = []
        for i, segment_info in enumerate(segments):
            result.append({
                "segment_id": i,
                "uid": segment_info.get('uid', i),
                "start_time": segment_info.get('start_time', 0),
                "end_time": segment_info.get('end_time', 0),
                "duration": segment_info.get('duration', 0),
                "summaries": segment_info.get('summaries', {})
            })

        return result

segment_knowledge_manager = SegmentKnowledgeManager()
