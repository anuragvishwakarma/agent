# agents/multi_agent_system.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict, List
from .maintenance_scheduler import MaintenanceSchedulerAgent
from .field_support import FieldSupportAgent
from .workload_manager import WorkloadManagerAgent
from data_loader.document_processor import DocumentProcessor

# Define the state structure
class AgentState(TypedDict):
    query: str
    maintenance_response: str
    field_support_response: str
    workload_response: str
    final_response: str
    all_responses: List[str]

class MultiAgentSystem:
    def __init__(self, document_processor: DocumentProcessor):
        if document_processor is None:
            raise ValueError("DocumentProcessor cannot be None")
            
        self.document_processor = document_processor
        self.agents = {
            "maintenance_scheduler": MaintenanceSchedulerAgent("Maintenance Scheduler", document_processor),
            "field_support": FieldSupportAgent("Field Support", document_processor),
            "workload_manager": WorkloadManagerAgent("Workload Manager", document_processor)
        }
        self.graph = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph StateGraph workflow for multi-agent collaboration"""
        workflow = StateGraph(AgentState)
        
        # Define nodes for each agent
        workflow.add_node("maintenance_scheduler", self._invoke_maintenance_scheduler)
        workflow.add_node("field_support", self._invoke_field_support)
        workflow.add_node("workload_manager", self._invoke_workload_manager)
        workflow.add_node("orchestrator", self._orchestrate_response)
        
        # Set the entry point
        workflow.set_entry_point("orchestrator")
        
        # Define conditional edges from orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_to_agents,
            {
                "maintenance_scheduler": "maintenance_scheduler",
                "field_support": "field_support", 
                "workload_manager": "workload_manager",
                "final": END
            }
        )
        
        # Define edges from agent nodes back to orchestrator
        workflow.add_edge("maintenance_scheduler", "orchestrator")
        workflow.add_edge("field_support", "orchestrator")
        workflow.add_edge("workload_manager", "orchestrator")
        
        return workflow.compile()
    
    def _route_to_agents(self, state: AgentState) -> str:
        """Route to appropriate agents based on state"""
        # If we haven't called any agents yet, start with maintenance scheduler
        if not state.get("maintenance_response"):
            return "maintenance_scheduler"
        elif not state.get("field_support_response"):
            return "field_support"
        elif not state.get("workload_response"):
            return "workload_manager"
        else:
            # All agents have responded, we're done
            return "final"
    
    def _invoke_maintenance_scheduler(self, state: AgentState) -> AgentState:
        """Invoke maintenance scheduler agent"""
        query = state.get("query", "")
        try:
            response = self.agents["maintenance_scheduler"].invoke(query)
            return {"maintenance_response": response}
        except Exception as e:
            return {"maintenance_response": f"Error: {str(e)}"}
    
    def _invoke_field_support(self, state: AgentState) -> AgentState:
        """Invoke field support agent"""
        query = state.get("query", "")
        try:
            response = self.agents["field_support"].invoke(query)
            return {"field_support_response": response}
        except Exception as e:
            return {"field_support_response": f"Error: {str(e)}"}
    
    def _invoke_workload_manager(self, state: AgentState) -> AgentState:
        """Invoke workload manager agent"""
        query = state.get("query", "")
        try:
            response = self.agents["workload_manager"].invoke(query)
            return {"workload_response": response}
        except Exception as e:
            return {"workload_response": f"Error: {str(e)}"}
    
    def _orchestrate_response(self, state: AgentState) -> AgentState:
        """Orchestrate and combine responses from all agents"""
        # Collect all responses
        maintenance_response = state.get("maintenance_response", "")
        field_support_response = state.get("field_support_response", "")
        workload_response = state.get("workload_response", "")
        
        # If we have all responses, create final consolidated response
        if maintenance_response and field_support_response and workload_response:
            final_response = self._create_final_response({
                "maintenance_scheduler": maintenance_response,
                "field_support": field_support_response,
                "workload_manager": workload_response,
                "query": state.get("query", "")
            })
            return {"final_response": final_response}
        
        # Otherwise, just pass through the state
        return state
    
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
        
        try:
            return self.agents["maintenance_scheduler"].aws_models.invoke_nova_pro(prompt)
        except Exception as e:
            return f"Error creating final response: {str(e)}"
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """Invoke the multi-agent system with a query"""
        if not query or not query.strip():
            return {"error": "Query cannot be empty"}
            
        initial_state = AgentState(
            query=query.strip(),
            maintenance_response="",
            field_support_response="", 
            workload_response="",
            final_response="",
            all_responses=[]
        )
        
        try:
            result = self.graph.invoke(initial_state)
            return dict(result)  # Convert to regular dict for easier handling
        except Exception as e:
            return {"error": f"Multi-agent system invocation failed: {str(e)}"}