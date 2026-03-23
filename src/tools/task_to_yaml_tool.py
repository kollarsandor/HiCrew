# tools/answer_to_yaml_tool.py

import os
import re
import yaml
from crewai.tools import BaseTool


# Standard task.yaml template - returned when extracted content does not match YAML format
STANDARD_TASK_YAML_TEMPLATE = """

subtask_1:
  agent: Video_Caption_Analysis_Agent
  description: |
    Analyze video captions to locate timestamps of key events.
    Choose appropriate search strategy based on question type:
    - Causal (CH/CW): Use caption_type="causal" to find causal descriptions
    - Temporal (TN/TC/TP): Build timeline, locate events before/after reference event
    - Descriptive (DL/DO/DC): Search for target objects, locations, or quantity information
  expected_output: |
        OUTPUT FORMAT (JSON):
    {
      "answer": "option number (A-E)",
      "reason": "detailed reasoning about the METHOD of execution",
      "confidence": 0.0-1.0,
      "evidence": ["timestamp/segment references"],
      "method_identified": "specific technique or manner observed"
    }

subtask_2:
  agent: Short_Video_Analysis_Agent
  description: |
    Based on caption analysis results, select relevant video segments for visual analysis.
    Focus on:
    - Specific execution method of actions
    - Temporal relationships of events
    - Visual features of objects
  expected_output: |
        OUTPUT FORMAT (JSON):
    {
      "answer": "option number (A-E)",
      "reason": "detailed reasoning based on visual METHOD observation",
      "confidence": 0.0-1.0,
      "segments_analyzed": [segment_ids],
      "method_observed": "specific technique visually confirmed"
    }

subtask_3:
  agent: Information_Integration_Agent
  description: |
    Integrate evidence from caption analysis and visual analysis.
    Verify information consistency and resolve potential conflicts.
    Adjust weights based on question type:
    - Causal How: Caption 40% + Visual 60%
    - Causal Why: Caption 40% + Visual 60%
    - Temporal: Caption 50% + Visual 50%
    - Descriptive: Adjust based on specific question
  expected_output: |
       OUTPUT FORMAT (JSON):
    {
      "integrated_answer": "option number (A-E)",
      "final_confidence": 0.0-1.0,
      "reasoning_process": {
        "caption_method": "method from caption analysis",
        "visual_method": "method from visual analysis",
        "agreement": true/false,
        "weight_applied": "Caption 40% + Visual 60%"
      },
      "final_evidence": "synthesized method description"
    }

subtask_4:
  agent: Answer_Agent
  description: |
    Based on integrated evidence, select the best answer from given options.
    Ensure the answer is fully consistent with the evidence chain.
  expected_output: |
    "option" like "X" where X is A/B/C/D/E
"""


class YamlToFileTool(BaseTool):
    name: str = "Answer to YAML and Save Tool"
    description: str = "Extracts YAML block from answer and saves it as a .yaml file."

    def _run(self, raw_answer, file_path: str = "") -> str:
        try:
            # Support raw_answer as object with .raw attribute or direct string
            raw_text = getattr(raw_answer, "raw", raw_answer) if hasattr(raw_answer, "raw") else str(raw_answer)

            match = re.search(r"```yaml(.*?)```", raw_text, re.DOTALL)
            if not match:
                return self._return_standard_template("No valid ```yaml ... ``` block found in the answer.")

            yaml_block = match.group(1).strip()

            # Try to parse YAML, if fails, attempt to fix common issues
            try:
                parsed_yaml = yaml.safe_load(yaml_block)
            except yaml.YAMLError as e:
                # Try to fix common YAML formatting issues
                fixed_yaml_block = self._fix_yaml_formatting(yaml_block)
                try:
                    parsed_yaml = yaml.safe_load(fixed_yaml_block)
                except yaml.YAMLError as e2:
                    return self._return_standard_template(f"YAML parsing error: {str(e)}")


            file_path = (
                    "/root/autodl-tmp/VideoCrew/videoanalyze_video_listener/src/crews/video_comprehension_crew/config/tasks.yaml"
            )

            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(parsed_yaml, f, allow_unicode=True)

            return f"YAML extracted and saved to: {file_path}"

        except Exception as e:
            return f"YAML conversion failed: {str(e)}"


    def _return_standard_template(self, error_reason: str) -> str:
        """
        Return standard task.yaml template when YAML format doesn't match
        """
        return (
            f"Error: {error_reason}\n\n"
            "The extracted content does not match the expected task.yaml format.\n"
            "Please provide a YAML matching the following standard template:\n\n"
            "```yaml"
            f"{STANDARD_TASK_YAML_TEMPLATE}"
            "```\n"
        )

    def _fix_yaml_formatting(self, yaml_block: str) -> str:
        """
        Attempt to fix common YAML formatting issues:
        1. Unquoted strings containing special characters like quotes, colons
        2. Values with "X" patterns that need proper quoting
        3. Multi-word values after colons that should be quoted
        """
        lines = yaml_block.split('\n')
        fixed_lines = []

        for line in lines:
            # Skip empty lines or comments
            if not line.strip() or line.strip().startswith('#'):
                fixed_lines.append(line)
                continue

            # Check if line has a key-value pattern (key: value)
            # Match pattern: spaces + key + colon + space + value
            kv_match = re.match(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', line)

            if kv_match:
                indent = kv_match.group(1)
                key = kv_match.group(2)
                value = kv_match.group(3).strip()

                # Check if value needs quoting
                needs_quoting = False

                # Already properly quoted - skip
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    fixed_lines.append(line)
                    continue

                # Value starts with | or > (block scalar) - skip
                if value in ['|', '>', '|-', '>-']:
                    fixed_lines.append(line)
                    continue

                # Check for problematic patterns that need quoting:
                # 1. Contains unescaped quotes
                # 2. Contains special YAML characters: : { } [ ] , & * # ? | - < > = ! % @ `
                # 3. Looks like "X" pattern (quoted letter in middle of string)
                # 4. Contains multiple spaces that might be significant

                problematic_patterns = [
                    r'"[^"]*"',           # Contains quoted substring like "X" or "option"
                    r"'[^']*'",           # Contains single-quoted substring
                    r':\s',               # Contains colon followed by space (embedded mapping)
                    r'[{}\[\],&*#?|<>=!%@`]',  # Special YAML characters
                ]

                for pattern in problematic_patterns:
                    if re.search(pattern, value):
                        needs_quoting = True
                        break

                if needs_quoting:
                    # Escape existing double quotes and wrap in double quotes
                    escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
                    fixed_line = f'{indent}{key}: "{escaped_value}"'
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)


def create_yaml_converter_tool():
    return YamlToFileTool()


def save_yaml_callback(result: str):
    tool = create_yaml_converter_tool()
    outcome = tool._run(raw_answer=result)
    print(f"[Callback] {outcome}")

