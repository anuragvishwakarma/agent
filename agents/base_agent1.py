# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_aws import BedrockLLM
from data_loader.document_processor import DocumentProcessor

class BaseAgent(ABC):
    def __init__(self, name: str, document_processor: DocumentProcessor):
        self.name = name
        self.document_processor = document_processor
        
        # Use BedrockLLM instead of ChatBedrock - often more reliable
        self.llm = BedrockLLM(
            model_id="us.amazon.nova-pro-v1:0",
            region_name="us-west-2",
            model_kwargs={
                "temperature": 0.1,
                "max_tokens": 2000,
                "top_p": 0.9
            }
        )
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        pass
    
    def get_context(self, query: str) -> str:
        """Get relevant context from documents"""
        try:
            relevant_docs = self.document_processor.search_documents(query, k=2)
            if not relevant_docs:
                return ""
            
            context = ""
            for doc_info in relevant_docs[:2]:
                doc = doc_info['document']
                context += f"Source: {doc['source']}\nContent: {doc['content'][:500]}\n\n"
            
            return context
        except:
            return ""
    
    def invoke(self, query: str) -> str:
        """Invoke the agent with the given query"""
        try:
            context = self.get_context(query)
            
            prompt = f"""
            {self.system_prompt}
            
            {f"CONTEXT FROM DOCUMENTS:\n{context}" if context else "No specific context available."}
            
            USER QUESTION: {query}
            
            Please provide a helpful response based on the available information.
            """
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            return f"Error invoking agent: {str(e)}"