# agents/field_support.py
from .base_agent import BaseAgent

class FieldSupportAgent(BaseAgent):
    def _get_system_prompt(self) -> str:
        return """You are a Field Support expert. You provide technical support and troubleshooting for field operations."""