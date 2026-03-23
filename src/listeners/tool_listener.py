import json
from crewai.utilities.events import (
    ToolUsageErrorEvent,
    ToolExecutionErrorEvent
)
from crewai.utilities.events.base_event_listener import BaseEventListener

class ToolErrorListener(BaseEventListener):
    def __init__(self):
        super().__init__()
        self.error_log_file = "/root/autodl-tmp/VideoCrew/videoanalyze_video_listener/src/listeners/tool_err.jsonl"

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_usage_error(source, event: ToolExecutionErrorEvent):
            print(f"tool err: {event.error}")
            print(f"tool input err: {event.tool_args}")
            error_data = {str(event.tool_args):str(event.error)}
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False)
                f.write('\n')

