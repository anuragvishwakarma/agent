# agents/multi_agent_system.py
import concurrent.futures
from typing import Dict, Any
from .maintenance_scheduler import MaintenanceSchedulerAgent
from .field_support import FieldSupportAgent
from .workload_manager import WorkloadManagerAgent
from langchain_aws import ChatBedrock

class MultiAgentSystem:
    def __init__(self, document_processor):
        self.document_processor = document_processor
        self.agents = {
            "maintenance": MaintenanceSchedulerAgent("Maintenance", document_processor),
            "field": FieldSupportAgent("Field", document_processor),
            "workload": WorkloadManagerAgent("Workload", document_processor)
        }
        # LLM for final consolidation
        self.llm = ChatBedrock(
            model_id="amazon.nova-pro-v1:0", 
            region_name="us-west-2",
            model_kwargs={"temperature": 0.1, "max_tokens": 2000}
        )
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """Run all agents and consolidate responses"""
        def run_agent(agent_name):
            try:
                return self.agents[agent_name].invoke(query)
            except Exception as e:
                return f"Agent error: {str(e)}"
        
        # Run agents in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(run_agent, "maintenance"): "maintenance",
                executor.submit(run_agent, "field"): "field", 
                executor.submit(run_agent, "workload"): "workload"
            }
            
            results = {}
            for future in concurrent.futures.as_completed(futures):
                agent_name = futures[future]
                results[f"{agent_name}_response"] = future.result()
        
        # Create simple consolidated response
        results["final_response"] = self._create_simple_final(query, results)
        return results
    
    def _create_simple_final(self, query: str, responses: Dict[str, Any]) -> str:
        """Create a simple consolidated response"""
        try:
            prompt = f"""
            Question: {query}
            
            Maintenance Expert: {responses.get('maintenance_response', 'No response')}
            Field Support: {responses.get('field_response', 'No response')}
            Workload Manager: {responses.get('workload_response', 'No response')}
            
            Combine these perspectives into one helpful answer.
            """
            
            from langchain.schema import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except:
            # Fallback
            return " | ".join([f"{k}: {v[:100]}..." for k, v in responses.items()])