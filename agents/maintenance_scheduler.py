# agents/maintenance_scheduler.py
from .base_agent import BaseAgent

class MaintenanceSchedulerAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are a Maintenance Scheduler expert. You create maintenance plans, schedules, and optimize maintenance operations."""