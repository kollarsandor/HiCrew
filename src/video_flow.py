#!/usr/bin/env python
"""
Video comprehension module using CrewAI for analyzing video content.
"""
import os
import logging
import json
import uuid
from typing import Dict
import shutil
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import sys

from crewai.flow import Flow, listen, start
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crews.task_analyze_andgenerate_crew.task_analyze_and_generate_crew import task_analyze_and_generate_Crew
from crews.video_comprehension_crew.video_comprehension_crew import Video_Comprehension_Crew

# 导入新的知识源管理器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.segment_caption_tools import SegmentKnowledgeManager
import tools.VTSearch_tool_with_depth as VTSearch_tool
from listeners.tool_listener import ToolErrorListener

DATA_DIR = Path("/root/autodl-tmp/VideoTree")
VIDEO_DIR = DATA_DIR /"data/nextqa/nextvideo"
DURATION_FILE = DATA_DIR / "data/nextqa/durations.json"
VAL_CSV = DATA_DIR / "data/nextqa/val.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultManager:

    def __init__(self, result_path: str = "./nextqa_results.jsonl"):
        self.result_path = result_path
        self._answered_uids: set = set()  # 存储已回答的video_uid集合
        self._load_existing()

    def _load_existing(self):

        if os.path.exists(self.result_path):
            try:
                with open(self.result_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                video_uid = entry.get('video_uid')
                                if video_uid:
                                    self._answered_uids.add(video_uid)
                            except json.JSONDecodeError as e:
                                logger.warning(f"跳过无效的JSON行: {e}")
                logger.info(f"已加载 {len(self._answered_uids)} 个已有结果")
            except Exception as e:
                logger.warning(f"无法加载已有结果文件: {e}，将从头开始")
                self._answered_uids = set()
        else:
            logger.info("未找到已有结果文件，将从头开始")

    def add_result(self, video_uid: str, answer: str, question: str = "", question_type: str = ""):

        if video_uid in self._answered_uids:
            logger.warning(f"问题 {video_uid} 已存在，跳过重复保存")
            return

        result_entry = {
            "video_uid": video_uid,
            "answer": answer,
            "question": question,
            "question_type": question_type,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        try:
            directory = os.path.dirname(self.result_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(self.result_path, 'a', encoding='utf-8') as f:
                json.dump(result_entry, f, ensure_ascii=False)
                f.write('\n')

            self._answered_uids.add(video_uid)
            logger.info(f"✓ saved result: {video_uid}")

        except Exception as e:
            logger.error(f"fail {video_uid}: {e}")

    def has_result(self, video_uid: str) -> bool:
        return video_uid in self._answered_uids

    def get_answered_count(self) -> int:

        return len(self._answered_uids)

    def __len__(self):
        return len(self._answered_uids)


_result_manager: 'ResultManager | None' = None


def get_result_manager(result_path: str = "./nextqa_results.jsonl") -> ResultManager:

    global _result_manager
    if _result_manager is None:
        _result_manager = ResultManager(result_path)
    return _result_manager

# Load video durations at module level for efficiency
_video_durations = None

def load_video_durations():
    """
    Load video durations from JSON file.
    
    Returns:
        Dictionary mapping video IDs to their durations in seconds
    """
    global _video_durations
    if _video_durations is None:
        try:
            with open(DURATION_FILE, 'r', encoding='utf-8') as f:
                _video_durations = json.load(f)
            logger.info(f"Loaded {len(_video_durations)} video durations from {DURATION_FILE}")
        except Exception as e:
            logger.error(f"Error loading video durations: {e}")
            _video_durations = {}
    return _video_durations


def get_question_type(video_uid: str) -> str:
    """
    Get question type from val.csv based on video_id and qid.
    
    Args:
        video_uid: Video UID in format "video_qid" (e.g., "4010069381_6")
    
    Returns:
        Question type string (e.g., "CH", "CW", "TN") or empty string if not found
    """
    try:
        video_id, qid = video_uid.rsplit('_', 1)
        df = pd.read_csv(VAL_CSV)
        row = df[(df['video'].astype(str) == video_id) & (df['qid'].astype(str) == qid)]
        if not row.empty:
            question_type = row.iloc[0]['type']
            logger.info(f"Question type for {video_uid}: {question_type}")
            return question_type
        else:
            logger.warning(f"Question type not found for {video_uid}")
            return ""
    except Exception as e:
        logger.error(f"Error getting question type for {video_uid}: {e}")
        return ""

def format_question_type(question_type: str) -> str:
    """
    Format question type code to descriptive text for YAML template.
    
    Args:
        question_type: Question type code (e.g., "CH", "CW", "TN")
    
    Returns:
        Formatted question type text or empty string
    """
    if not question_type:
        return ""
    
    type_mapping = {
        "CH": "Causal How - Focus on METHOD/TECHNIQUE of action execution",
        "CW": "Causal Why - Focus on CAUSE/PURPOSE (bidirectional: before=trigger, after=goal)",
        "TN": "Temporal Next - Focus on what happens AFTER the reference event",
        "TC": "Temporal Current - Focus on CONCURRENT/SIMULTANEOUS events",
        "TP": "Temporal Previous - Focus on what happened BEFORE the reference event",
        "DC": "Descriptive Counting - Focus on COUNTING objects/people across segments",
        "DL": "Descriptive Location - Focus on SCENE/ENVIRONMENT identification",
        "DO": "Descriptive Object - Focus on OBJECT/RELATIONSHIP identification"
    }
    
    type_description = type_mapping.get(question_type, f"Unknown Type: {question_type}")
    return f"Question Type: {question_type} ({type_description})"

def get_video_duration(video_uid: str) -> int:
    """
    Get duration for a specific video.
    
    Args:
        video_uid: Video UID in format "video_qid" (e.g., "2435100235_7")
        
    Returns:
        Duration in seconds, or None if not found
    """
    durations = load_video_durations()
    # Extract video ID from video_uid (format: "video_qid" -> "video")
    video_id = video_uid.rsplit('_', 1)[0]
    duration = durations.get(video_id)
    if duration is None:
        logger.warning(f"Duration not found for video {video_id}")
    return duration


def set_crewai_storage_dir(video_id):

    storage_dir = f"VideoCrew/nextqa_temp_storage/{video_id}"

    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    os.makedirs(storage_dir, exist_ok=True)

    os.environ["CREWAI_STORAGE_DIR"] = storage_dir
    logger.info(f"Set CREWAI_STORAGE_DIR to {storage_dir}")

    return storage_dir


class HashableKnowledgeSourceWrapper:
    """Wrapper to make StringKnowledgeSource objects hashable."""

    def __init__(self, source: StringKnowledgeSource):
        self.source = source
        self.id = id(source)  # Use object memory address as unique identifier

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, HashableKnowledgeSourceWrapper):
            return self.id == other.id
        return False


class VideoComprehensionState(BaseModel):
    """State model for VideoComprehensionFlow."""
    question: str = ""
    uuid: str = ""
    answer: str = ""
    video_key_frames_captions: Dict[str, str] = {}


class VideoComprehensionFlow(Flow[VideoComprehensionState]):
    """Flow for video comprehension tasks using CrewAI."""
    def __init__(self, video_uid: str, question: str, video_caption_source=None,
                 duration: int = None, full_video_uid: str = None, question_type: str = ""):
        super().__init__()
        self.state.uuid = video_uid
        self.state.question = question
        # self.video_caption_source = video_caption_source
        self.segment_knowledge_source = None
        self.duration = duration
        self.full_video_uid = full_video_uid if full_video_uid else video_uid
        self.question_type = question_type

        self.knowledge_manager = SegmentKnowledgeManager()


    @start()
    def load_segment_knowledge(self):

        logger.info(f"Loading segment knowledge for video: {self.state.uuid}")
        logger.info(f"Question type: {self.question_type}")

        knowledge_content = self.knowledge_manager.get_knowledge_source_content(
            video_id=self.state.uuid,
            question_type=self.question_type
        )
        print(knowledge_content)

        self.segment_knowledge_source = HashableKnowledgeSourceWrapper(
            StringKnowledgeSource(content=knowledge_content)
        )

        logger.info(f"Segment knowledge loaded successfully")

    @listen(load_segment_knowledge)
    def generate_tasks(self):
        logger.info("Task Analyze for video")

        question_type_text = format_question_type(self.question_type)

        task_crew = task_analyze_and_generate_Crew(question_type=self.question_type)

        (
            task_crew
            .task_generate_crew()
            .kickoff(inputs={"uuid": self.state.uuid, "question": self.state.question, "duration": self.duration, "question_type": question_type_text})
        )

    @listen(generate_tasks)
    def video_comprehension(self):

        question_type_text = format_question_type(self.question_type)

        video_crew = Video_Comprehension_Crew(question_type=self.question_type)
        ans = (
            video_crew
            .video_comprehension_crew(self.segment_knowledge_source)
            .kickoff(inputs={"uuid": self.state.uuid, "question": self.state.question, "duration": self.duration, "question_type": question_type_text})
        )

        result_manager = get_result_manager()
        result_manager.add_result(
            video_uid=self.full_video_uid,
            answer=ans.raw,
            question=self.state.question,
            question_type=self.question_type
        )
        print(f"\n{'='*60}")
        print(f"✓  {self.full_video_uid} saved")
        print(f"answer: {ans.raw}")
        print(f"{'='*60}\n")



def load_video_ids(csv_path: str, caption_dir: str = "/root/autodl-tmp/good_cap", min_duration: int = 60):
    """
    Load video IDs from the CSV file and filter by available caption files and video duration.
    
    For NExT-QA dataset, video_uid format is "video_qid" (e.g., "4010069381_6")

    Args:
        csv_path: Path to the CSV file
        caption_dir: Directory containing caption JSON files
        min_duration: Minimum video duration in seconds (default: 60)

    Returns:
        List of video UIDs in format "video_qid" that have caption files and meet duration requirement
    """
    try:
        import pandas as pd
        import os
        from pathlib import Path
        
        df = pd.read_csv(csv_path)
        
        # Load video durations
        durations = load_video_durations()
        
        # Get all available caption files
        caption_files = set()
        if os.path.exists(caption_dir):
            caption_files = {
                f.stem for f in Path(caption_dir).glob("*.json")
            }
        
        # Combine video and qid to create video_uid, filter by available captions and duration
        video_uids = []
        skipped_no_caption = 0
        skipped_short_duration = 0
        for _, row in df.iterrows():
            video_id = str(row['video'])
            video_uid = f"{video_id}_{row['qid']}"
            
            # Check if caption file exists
            if video_uid not in caption_files:
                skipped_no_caption += 1
                continue
            
            # Check if video duration meets minimum requirement
            duration = durations.get(video_id)
            if duration is None or duration < min_duration:
                skipped_short_duration += 1
                continue
            
            video_uids.append(video_uid)
        
        logger.info(f"Loaded {len(video_uids)} video UIDs from {csv_path}")
        logger.info(f"Skipped {skipped_no_caption} video UIDs without caption files")
        logger.info(f"Skipped {skipped_short_duration} video UIDs with duration < {min_duration} seconds")
        return video_uids
    except Exception as e:
        logger.error(f"Error loading video UIDs from {csv_path}: {e}")
        return []


def save_result_to_json(dictionary, filepath):

    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=4, ensure_ascii=False)

    print(f"saved in {filepath}")
    return True


def load_err_video_ids(jsonl_path: str):
    """
    Load video IDs from the JSONL file.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of video IDs
    """
    try:
        video_ids = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                video_ids.append(data['video_uuid'])
        
        logger.info(f"Loaded {len(video_ids)} video IDs from {jsonl_path}")
        return video_ids
    except Exception as e:
        logger.error(f"Error loading video IDs from {jsonl_path}: {e}")
        return []


def main():
    """Main function to execute the video comprehension flow."""
    # Initialize with specific video UID
    tool_error_listener = ToolErrorListener()

    result_path = "./nextqa_results.jsonl"
    result_manager = get_result_manager(result_path)

    logger.info(f"已加载 {result_manager.get_answered_count()} 个已回答的问题")

    # Load video UIDs from CSV file (format: "video_qid")
    csv_path = r"/root/autodl-tmp/VideoTree/data/nextqa/val.csv"
    video_ids = load_video_ids(csv_path)

    
    logger.info(f"total {len(video_ids)} questions")

    processed_count = 0
    skipped_count = 0

    for i, video_uid in enumerate(video_ids):
        if result_manager.has_result(video_uid):
            skipped_count += 1
            continue
        
        processed_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(video_ids)}] : {video_uid}")
        logger.info(f"{'='*60}")

        storage_id = f"{video_uid}_{uuid.uuid4().hex[:8]}"
        set_crewai_storage_dir(storage_id)

        # Get question for this video
        question = VTSearch_tool.getVideoQA(video_uid)
        logger.info(f"Question: {question}")

        # # Get video caption source
        # video_caption_source = get_video_caption_source(video_uid)

        # Get video duration
        duration = get_video_duration(video_uid)
        logger.info(f"Video duration: {duration} seconds")

        # Get question type for knowledge source selection
        question_type = get_question_type(video_uid)
        logger.info(f"Question type: {question_type}")

        video_id, qid = video_uid.rsplit('_', 1)
        try:
            # Initialize and run the flow
            video_comprehension_flow = VideoComprehensionFlow(
                video_uid=video_id,
                question=question,
                # video_caption_source=video_caption_source,
                duration=duration,
                full_video_uid=video_uid,
                question_type=question_type
            )
            video_comprehension_flow.kickoff()

            logger.info(f"✓ {video_uid}")

        except Exception as e:
            error_data = {
                "video_uid": video_uid,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            with open("./nextqa_err.jsonl", 'a', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False)
                f.write('\n')
            logger.error(f"✗  {video_uid} fail: {e}")

        folder = f"VideoCrew/nextqa_temp_storage/{storage_id}"
        shutil.rmtree(folder) if Path(folder).is_dir() else None

    logger.info(f"\n{'='*60}")
    logger.info(f"finished!")
    logger.info(f"total question: {len(video_ids)}")
    logger.info(f"already answer: {result_manager.get_answered_count()}")
    logger.info(f"result file: {result_path}")
    logger.info(f"{'='*60}\n")

    result_manager.save()


if __name__ == "__main__":
    main()