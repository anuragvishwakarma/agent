# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, SystemMessage
from data_loader.document_processor import DocumentProcessor

class BaseAgent(ABC):
    def __init__(self, name: str, document_processor: DocumentProcessor):
        self.name = name
        self.document_processor = document_processor
        self.llm = ChatBedrock(
            model_id="amazon.nova-pro-v1:0",
            region_name="us-west-2",
            model_kwargs={
                "temperature": 0.1,
                "max_tokens": 2000
            }
        )
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        pass
    
    def get_context(self, query: str) -> str:
        """Get relevant context from documents"""
        relevant_docs = self.document_processor.search_documents(query, k=2)
        if not relevant_docs:
            return "No relevant context found."
        
        context = "Relevant document context:\n"
        for doc_info in relevant_docs[:2]:  # Limit to 2 most relevant
            doc = doc_info['document']
            context += f"From {doc['source']}:\n{doc['content'][:300]}...\n\n"
        
        return context
    
    def invoke(self, query: str) -> str:
        """Invoke the agent with the given query"""
        context = self.get_context(query)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {query}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"