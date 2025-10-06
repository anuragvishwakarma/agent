# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from models.aws_models import AWSModels
from data_loader.document_processor import DocumentProcessor

class BaseAgent(ABC):
    def __init__(self, name: str, document_processor: DocumentProcessor):
        self.name = name
        self.aws_models = AWSModels()
        self.document_processor = document_processor
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        pass
    
    def get_context(self, query: str) -> str:
        """Get relevant context from documents"""
        relevant_docs = self.document_processor.search_documents(query)
        context = "Relevant document context:\n"
        
        for doc_info in relevant_docs:
            doc = doc_info['document']
            context += f"Source: {doc['source']}\n"
            context += f"Content: {doc['content'][:500]}...\n\n"
        
        return context
    
    def invoke(self, query: str, **kwargs) -> str:
        """Invoke the agent with the given query"""
        context = self.get_context(query)
        
        full_prompt = f"""
        {self.system_prompt}
        
        CONTEXT:
        {context}
        
        USER QUERY: {query}
        
        ADDITIONAL PARAMETERS: {kwargs}
        
        Please provide a comprehensive response based on the context and your expertise.
        """
        
        return self.aws_models.invoke_nova_pro(full_prompt)