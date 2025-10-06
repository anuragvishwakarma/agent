# agents/workload_manager.py
from .base_agent import BaseAgent

class WorkloadManagerAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are a Workload Manager expert. You optimize resource allocation and workload distribution."""