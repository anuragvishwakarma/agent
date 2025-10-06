# agents/multi_agent_system.py
import concurrent.futures
from typing import Dict, Any
from .maintenance_scheduler import MaintenanceSchedulerAgent
from .field_support import FieldSupportAgent
from .workload_manager import WorkloadManagerAgent

class MultiAgentSystem:
    def __init__(self, document_processor):
        self.document_processor = document_processor
        self.agents = {
            "maintenance_scheduler": MaintenanceSchedulerAgent("Maintenance Scheduler", document_processor),
            "field_support": FieldSupportAgent("Field Support", document_processor),
            "workload_manager": WorkloadManagerAgent("Workload Manager", document_processor)
        }
    
    def invoke(self, query: str) -> Dict[str, Any]:
        """Run all agents in parallel and create consolidated response"""
        def run_agent(agent_name):
            try:
                return self.agents[agent_name].invoke(query)
            except Exception as e:
                return f"Error from {agent_name}: {str(e)}"
        
        # Run all agents in parallel
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
        
        # Create final consolidated response using Nova Pro
        results["final_response"] = self._create_final_response(query, results)
        results["query"] = query
        
        return results
    
    def _create_final_response(self, query: str, responses: Dict[str, Any]) -> str:
        """Use Nova Pro to create a coordinated final response"""
        # Prepare the consolidation prompt
        consolidation_prompt = f"""
        You are an expert coordinator synthesizing responses from three specialized agents.

        ORIGINAL USER QUERY: {query}

        AGENT RESPONSES:

        MAINTENANCE SCHEDULER (Planning & Scheduling):
        {responses.get('maintenance_scheduler_response', 'No response from maintenance scheduler')}

        FIELD SUPPORT (Technical & Operational):
        {responses.get('field_support_response', 'No response from field support')}

        WORKLOAD MANAGER (Resource & Efficiency):
        {responses.get('workload_manager_response', 'No response from workload manager')}

        TASK: Synthesize these responses into one comprehensive, actionable final recommendation.

        Please provide a well-structured response that:
        1. Starts with a brief executive summary
        2. Integrates key insights from all three perspectives
        3. Highlights any synergies or conflicts between recommendations
        4. Provides prioritized, actionable steps with clear ownership
        5. Includes timeline suggestions and resource requirements
        6. Identifies risks and mitigation strategies
        7. Ends with clear next steps

        Format the response for operational leadership review.
        """
        
        try:
            # Use any agent's Nova Pro invocation method
            return self.agents["maintenance_scheduler"].invoke_nova_pro(consolidation_prompt, max_tokens=2500)
        except Exception as e:
            # Fallback: create a simple combined response
            return f"Final coordination unavailable. Please review individual agent responses. Error: {str(e)}"