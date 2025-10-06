# data_loader/document_processor.py
import PyPDF2
import pandas as pd
import faiss
import numpy as np
from typing import List, Dict, Any
import os
from models.aws_models import AWSModels

class DocumentProcessor:
    def __init__(self):
        self.aws_models = AWSModels()
        self.index = None
        self.documents = []
    
    def load_pdf_documents(self, pdf_folder: str) -> List[Dict[str, Any]]:
        """Load and process PDF documents"""
        pdf_docs = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(pdf_folder, filename)
                try:
                    with open(filepath, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        
                        pdf_docs.append({
                            'source': filename,
                            'content': text,
                            'type': 'pdf'
                        })
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        return pdf_docs
    
    def load_csv_documents(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load and process CSV documents"""
        try:
            df = pd.read_csv(csv_path)
            # Convert dataframe to text representation
            csv_content = f"CSV Data with {len(df)} rows and {len(df.columns)} columns:\n"
            csv_content += df.head(10).to_string()  # First 10 rows as sample
            
            return [{
                'source': os.path.basename(csv_path),
                'content': csv_content,
                'type': 'csv',
                'dataframe': df
            }]
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return []
    
    def create_vector_store(self, documents: List[Dict[str, Any]]):
        """Create FAISS vector store from documents"""
        texts = [doc['content'] for doc in documents]
        embeddings = []
        
        for text in texts:
            embedding = self.aws_models.get_nova_pro_embedding(text[:1000])  # First 1000 chars
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        self.documents = documents
    
    def search_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if self.index is None:
            return []
        
        query_embedding = self.aws_models.get_nova_pro_embedding(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'score': float(distances[0][i])
                })
        
        return results