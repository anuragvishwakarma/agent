# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import boto3
import json
from data_loader.document_processor import DocumentProcessor

class BaseAgent(ABC):
    def __init__(self, name: str, document_processor: DocumentProcessor):
        self.name = name
        self.document_processor = document_processor
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Define the agent's specialized system prompt"""
        pass
    
    def get_context(self, query: str) -> str:
        """Get relevant context from documents"""
        relevant_docs = self.document_processor.search_documents(query, k=3)
        context = "Relevant document context:\n"
        
        for doc_info in relevant_docs:
            doc = doc_info['document']
            context += f"Source: {doc['source']}\n"
            context += f"Content: {doc['content'][:500]}...\n\n"
        
        return context if context.strip() else "No relevant context found in documents."
    
    def invoke_nova_pro(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Invoke Amazon Nova Pro model via Bedrock"""
        try:
            # Format for Nova Pro model
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            body = json.dumps({
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": max_tokens,
                "anthropic_version": "bedrock-2023-05-31"
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId="us.amazon.nova-pro-v1:0",
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract the response text
            if 'content' in response_body:
                return response_body['content'][0]['text']
            else:
                return "No response generated."
            
        except Exception as e:
            return f"Error invoking Nova Pro: {str(e)}"
    
    def invoke(self, query: str, **kwargs) -> str:
        """Invoke the agent with the given query"""
        context = self.get_context(query)
        
        full_prompt = f"""
        {self.system_prompt}
        
        CONTEXT FROM DOCUMENTS:
        {context}
        
        USER QUERY: {query}
        
        Please provide a comprehensive response based on the context and your expertise.
        """
        
        return self.invoke_nova_pro(full_prompt)