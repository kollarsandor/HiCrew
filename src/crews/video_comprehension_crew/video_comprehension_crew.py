from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew
from dotenv import load_dotenv
import os
import sys
import yaml
from pathlib import Path

from tools.video_tool_enhanced import AnalyzeSegmentVideoTool
from tools.segment_caption_tools import GetCaptionTimestampsTool
from tools.segment_caption_tools import GetSegmentCaptionTypeTool
from tools.segment_caption_tools import GetAvailableTimestampsTool

VALID_QUESTION_TYPES = {'CH', 'CW', 'TN', 'TC', 'TP', 'DL', 'DO', 'DC'}


def load_agents_config_by_type(question_type: str) -> dict:
    crew_dir = Path(__file__).parent

    q_type = question_type.upper() if question_type else 'CH'
    if q_type not in VALID_QUESTION_TYPES:
        q_type = 'CH'

    config_file = crew_dir / "config" / "agents_by_type" / f"agents_{q_type}.yaml"

    if not config_file.exists():
        config_file = crew_dir / "config" / "agents.yaml"

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@CrewBase
class Video_Comprehension_Crew:
    default_llm1 = LLM(        
        base_url="your api url",
        api_key="your api key",
        model="gpt-4o", 
        temperature=0.3
    )
    default_llm2 = LLM(        
        base_url="your api url",
        api_key="your api key",
        model="gpt-4o", 
        temperature=0.6
    )
    embedder_config = {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v3",
            "api_key": "your api key",
            "api_base": "your api url" }
    }

    def __init__(self, question_type: str = None):
        """
        Args:
            question_type: 问题类型 (CH, CW, TN, TC, TP, DL, DO, DC)
        """
        super().__init__()
        self._question_type = None
        self._dynamic_agents_config = None

        if question_type:
            self._question_type = question_type.upper()
            self._dynamic_agents_config = load_agents_config_by_type(question_type)

    def _get_agents_config(self):
        return self._dynamic_agents_config

    @agent
    def Video_Caption_Analysis_Agent(self) -> Agent:
        caption_tools =  [
            GetAvailableTimestampsTool(),
            GetCaptionTimestampsTool(),
            GetSegmentCaptionTypeTool()
        ] 

        config = self._get_agents_config()
        if isinstance(config, dict):
            agent_config = config["Video_Caption_Analysis_Agent"]

        return Agent(
            config=agent_config,
            verbose=True,
            llm=self.default_llm1,
            tools=caption_tools,
            max_iter=15,
        )


    @agent
    def Short_Video_Analysis_Agent(self) -> Agent:
        video_tools = [
            AnalyzeSegmentVideoTool() 
        ] 

        config = self._get_agents_config()
        if isinstance(config, dict):
            agent_config = config["Short_Video_Analysis_Agent"]

        return Agent(
            config=agent_config,
            verbose=True,
            llm=self.default_llm2,
            tools=video_tools,  # 视频分析工具
            max_iter=5,
        )

    @agent
    def Information_Integration_Agent(self) -> Agent:
        config = self._get_agents_config()
        if isinstance(config, dict):
            agent_config = config["Information_Integration_Agent"]
        return Agent(
            config=agent_config,
            verbose=True,
            llm=self.default_llm2,
        )

    @agent
    def Answer_Agent(self) -> Agent:
        config = self._get_agents_config()
        if isinstance(config, dict):
            agent_config = config["Answer_Agent"]
        else:
            agent_config = self.agents_config["Answer_Agent"]

        return Agent(
            config=agent_config,
            verbose=True,
            llm=self.default_llm1,
        )

    def _load_tasks(self):
        tasks = []

        for task_name, task_info in self.tasks_config.items():
            task_instance = Task(
                config={
                    "name": task_name,
                    "description": task_info['description'],
                    "expected_output": task_info['expected_output'],
                    "agent": task_info['agent'],
                }
            )
            tasks.append(task_instance)

        return tasks

    @crew
    def video_comprehension_crew(self,segment_knowledge_source) -> Crew:
        tasks = self._load_tasks()
        segment_knowledge_source = segment_knowledge_source.source
        
        return Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=False,
            knowledge_sources=[segment_knowledge_source],
            embedder=self.embedder_config,
        )
