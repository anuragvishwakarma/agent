# models/aws_models.py
import boto3
import json
from typing import List
import os

class AWSModels:
    def __init__(self):
        # SageMaker automatically handles Bedrock access
        # No explicit credentials needed in SageMaker environment
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
    def get_nova_pro_embedding(self, text: str) -> List[float]:
        """Get embeddings using Amazon Titan Embeddings"""
        try:
            body = json.dumps({
                "inputText": text
            })
            response = self.bedrock_runtime.invoke_model(
                body=body,
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            return response_body.get('embedding', [])
        except Exception as e:
            print(f"Embedding error: {e}")
            # Fallback to sentence transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(text).tolist()
    
    def invoke_nova_pro(self, prompt: str, temperature: float = 0.1) -> str:
        """Invoke Amazon Nova Pro model"""
        try:
            # Updated for Nova Pro model format
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": 2048
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId="us.amazon.nova-pro-v1:0",
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('content', [{}])[0].get('text', '')
            
        except Exception as e:
            return f"Error invoking Nova Pro: {str(e)}"