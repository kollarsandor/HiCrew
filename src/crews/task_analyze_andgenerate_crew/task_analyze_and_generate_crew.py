from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from dotenv import load_dotenv
import os
import sys
import yaml
from pathlib import Path

from tools.task_to_yaml_tool import save_yaml_callback

VALID_QUESTION_TYPES = {'CH', 'CW', 'TN', 'TC', 'TP', 'DL', 'DO', 'DC'}


def load_agents_config_by_type(question_type: str) -> dict:

    current_dir = Path(__file__).parent

    q_type = question_type.upper() if question_type else 'CH'
    if q_type not in VALID_QUESTION_TYPES:
        q_type = 'CH'

    config_file = current_dir / "config" / "agents_by_type" / f"agents_{q_type}.yaml"

    if not config_file.exists():
        config_file = current_dir / "config" / "agents.yaml"

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@CrewBase
class task_analyze_and_generate_Crew:
    default_llm = LLM(        
        base_url="your api url",
        api_key="your api key",
        model="gpt-4o", 
        temperature=0.2
    )
    embedder_config = {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v3",
            "api_key": "your api key",
            "api_base": "your api url"
        }
    }

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, question_type: str = None):

        super().__init__()

        self._question_type = None
        self._dynamic_agents_config = None

        if question_type:
            self._question_type = question_type.upper()
            self._dynamic_agents_config = load_agents_config_by_type(question_type)

    def _get_agents_config(self):

        if self._dynamic_agents_config:
            return self._dynamic_agents_config
        return self.agents_config

    @agent
    def Video_QA_Analyzer(self) -> Agent:
        config = self._get_agents_config()
        if isinstance(config, dict):
            agent_config = config["Video_QA_Analyzer"]
        else:
            agent_config = self.agents_config["Video_QA_Analyzer"]
        return Agent(
            config=agent_config,
            verbose=True,
            llm=self.default_llm,
        )

    @agent
    def Assignment_Verification_and_Format_Converter(self) -> Agent:
        config = self._get_agents_config()
        if isinstance(config, dict):
            agent_config = config["Assignment_Verification_and_Format_Converter"]
        else:
            agent_config = self.agents_config["Assignment_Verification_and_Format_Converter"]
        return Agent(
            config=agent_config,
            verbose=True,
            llm=self.default_llm,
        )

    @task
    def Video_Question_Analyze(self) -> Task:
        return Task(
            config=self.tasks_config["Video_Question_Analyze"],
        )

    @task
    def Assignment_Verification_and_Format_Convert(self) -> Task:
        return Task(
            config=self.tasks_config["Assignment_Verification_and_Format_Convert"],
            callback=save_yaml_callback
        )

    @crew
    def task_generate_crew(self) -> Crew:
        
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
            embedder=self.embedder_config,
        )