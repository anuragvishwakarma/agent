# agents/multi_agent_system.py
import concurrent.futures
from typing import Dict, Any
from .maintenance_scheduler import MaintenanceSchedulerAgent
from .field_support import FieldSupportAgent
from .workload_manager import WorkloadManagerAgent

class MultiAgentSystem:
    def __init__(self, document_processor):
        self.agents = {
            "maintenance_scheduler": MaintenanceSchedulerAgent("Maintenance Scheduler", document_processor),
            "field_support": FieldSupportAgent("Field Support", document_processor),
            "workload_manager": WorkloadManagerAgent("Workload Manager", document_processor)
        }
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """Run all agents in parallel"""
        def run_agent(agent_name):
            try:
                return self.agents[agent_name].invoke(query)
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Run agents in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(run_agent, "maintenance_scheduler"): "maintenance_scheduler_response",
                executor.submit(run_agent, "field_support"): "field_support_response", 
                executor.submit(run_agent, "workload_manager"): "workload_manager_response"
            }
            
            results = {}
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                results[key] = future.result()
        
        # Create final response
        results["final_response"] = self._create_final_response(query, results)
        return results
    
    def _create_final_response(self, query: str, responses: Dict[str, Any]) -> str:
        """Combine agent responses into one coherent answer"""
        prompt = f"""
        Query: {query}
        
        Maintenance Agent: {responses.get('maintenance_scheduler_response', 'No response')}
        Field Support: {responses.get('field_support_response', 'No response')}
        Workload Manager: {responses.get('workload_manager_response', 'No response')}
        
        Please provide a comprehensive, coordinated response that combines insights from all three specialists.
        """
        
        try:
            return self.agents["maintenance_scheduler"].aws_models.invoke_nova_pro(prompt)
        except:
            # Fallback: just combine the responses
            main_parts = []
            for key, response in responses.items():
                if "response" in key and "No response" not in str(response):
                    main_parts.append(str(response))
            
            return "\n\n".join(main_parts) if main_parts else "No responses from agents"