# agents/multi_agent_system.py
from langgraph.graph import Graph
from typing import Dict, Any, List
from .maintenance_scheduler import MaintenanceSchedulerAgent
from .field_support import FieldSupportAgent
from .workload_manager import WorkloadManagerAgent
from data_loader.document_processor import DocumentProcessor

class MultiAgentSystem:
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.agents = {
            "maintenance_scheduler": MaintenanceSchedulerAgent("Maintenance Scheduler", document_processor),
            "field_support": FieldSupportAgent("Field Support", document_processor),
            "workload_manager": WorkloadManagerAgent("Workload Manager", document_processor)
        }
        self.graph = self._create_workflow()
    
    def _create_workflow(self) -> Graph:
        """Create LangGraph workflow for multi-agent collaboration"""
        workflow = Graph()
        
        # Define nodes for each agent
        workflow.add_node("maintenance_scheduler", self._invoke_maintenance_scheduler)
        workflow.add_node("field_support", self._invoke_field_support)
        workflow.add_node("workload_manager", self._invoke_workload_manager)
        workflow.add_node("orchestrator", self._orchestrate_response)
        
        # Define workflow edges
        workflow.set_entry_point("orchestrator")
        workflow.add_edge("orchestrator", "maintenance_scheduler")
        workflow.add_edge("orchestrator", "field_support")
        workflow.add_edge("orchestrator", "workload_manager")
        workflow.add_edge("maintenance_scheduler", "orchestrator")
        workflow.add_edge("field_support", "orchestrator")
        workflow.add_edge("workload_manager", "orchestrator")
        
        return workflow.compile()
    
    def _invoke_maintenance_scheduler(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        response = self.agents["maintenance_scheduler"].invoke(query)
        return {"maintenance_response": response}
    
    def _invoke_field_support(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        response = self.agents["field_support"].invoke(query)
        return {"field_support_response": response}
    
    def _invoke_workload_manager(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        response = self.agents["workload_manager"].invoke(query)
        return {"workload_response": response}
    
    def _orchestrate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate and combine responses from all agents"""
        combined_response = {
            "maintenance_scheduler": state.get("maintenance_response", ""),
            "field_support": state.get("field_support_response", ""),
            "workload_manager": state.get("workload_response", ""),
            "timestamp": state.get("timestamp"),
            "query": state.get("query")
        }
        
        # Create final consolidated response
        final_response = self._create_final_response(combined_response)
        combined_response["final_response"] = final_response
        
        return combined_response
    
    def _create_final_response(self, responses: Dict[str, Any]) -> str:
        """Create a consolidated final response"""
        prompt = f"""
        Consolidate the following agent responses into a comprehensive, actionable final response:
        
        ORIGINAL QUERY: {responses['query']}
        
        MAINTENANCE SCHEDULER:
        {responses['maintenance_scheduler']}
        
        FIELD SUPPORT:
        {responses['field_support']}
        
        WORKLOAD MANAGER:
        {responses['workload_manager']}
        
        Please provide a well-structured final response that:
        1. Summarizes key insights from all agents
        2. Highlights any conflicts or synergies between recommendations
        3. Provides prioritized actionable steps
        4. Suggests implementation timeline
        5. Identifies potential risks and mitigations
        
        Format the response for executive review with clear sections.
        """
        
        return self.agents["maintenance_scheduler"].aws_models.invoke_nova_pro(prompt)
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """Invoke the multi-agent system with a query"""
        initial_state = {
            "query": query,
            "timestamp": "2024-01-01"  # You can use actual timestamp
        }
        
        return self.graph.invoke(initial_state)