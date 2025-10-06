# data_loader/document_processor.py
import PyPDF2
import pandas as pd
import faiss
import numpy as np
from typing import List, Dict, Any
import os
import json
import boto3
import pickle
import time
from datetime import datetime

class DocumentProcessor:
    def __init__(self, storage_dir: str = "vector_store"):
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.storage_dir = storage_dir
        self.index = None
        self.documents = []
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def get_bedrock_embedding(self, text: str) -> List[float]:
        """Get embeddings using Amazon Titan Embeddings via Bedrock"""
        try:
            # Clean and prepare text
            clean_text = text.replace('\n', ' ').strip()
            if len(clean_text) > 10000:  # Bedrock limit
                clean_text = clean_text[:10000]
            
            body = json.dumps({
                "inputText": clean_text
            })
            
            response = self.bedrock_runtime.invoke_model(
                body=body,
                modelId="amazon.titan-embed-text-v1",
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding', [])
            
            if not embedding:
                raise ValueError("Empty embedding received")
                
            return embedding
            
        except Exception as e:
            print(f"Bedrock embedding error: {e}")
            # Fallback to local embeddings if Bedrock fails
            return self._get_fallback_embedding(text)
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding using sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text).tolist()
            print("Using fallback embeddings")
            return embedding
        except Exception as e:
            print(f"Fallback embedding also failed: {e}")
            # Return zero vector as last resort
            return [0.0] * 384  # Standard dimension for all-MiniLM-L6-v2
    
    def load_pdf_documents(self, pdf_folder: str) -> List[Dict[str, Any]]:
        """Load and process PDF documents with chunking for better embeddings"""
        pdf_docs = []
        
        if not os.path.exists(pdf_folder):
            print(f"PDF folder {pdf_folder} does not exist")
            return pdf_docs
            
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(pdf_folder, filename)
                try:
                    with open(filepath, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        
                        for page_num, page in enumerate(pdf_reader.pages):
                            page_text = page.extract_text()
                            if page_text:
                                text += f"Page {page_num + 1}: {page_text}\n"
                        
                        if text.strip():
                            # Chunk the document for better embeddings
                            chunks = self._chunk_text(text, chunk_size=1000, overlap=200)
                            
                            for i, chunk in enumerate(chunks):
                                pdf_docs.append({
                                    'source': f"{filename}_chunk_{i+1}",
                                    'content': chunk,
                                    'type': 'pdf',
                                    'original_file': filename,
                                    'chunk_id': i,
                                    'file_path': filepath,
                                    'total_chunks': len(chunks)
                                })
                        else:
                            print(f"No text extracted from {filename}")
                            
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
        print(f"Loaded {len(pdf_docs)} PDF chunks from {pdf_folder}")
        return pdf_docs
    
    def load_csv_documents(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load and process CSV documents with intelligent chunking"""
        csv_docs = []
        
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} does not exist")
            return csv_docs
            
        try:
            df = pd.read_csv(csv_path)
            
            # Create multiple representations of the CSV data
            representations = []
            
            # 1. Overall summary
            summary_content = f"""
            CSV Data Summary:
            - File: {os.path.basename(csv_path)}
            - Rows: {len(df)}
            - Columns: {len(df.columns)}
            - Columns: {', '.join(df.columns.tolist())}
            - Data Types: {df.dtypes.to_dict()}
            - Missing Values: {df.isnull().sum().to_dict()}
            """
            representations.append(summary_content)
            
            # 2. Column-wise descriptions
            for column in df.columns:
                col_summary = f"""
                Column: {column}
                Type: {df[column].dtype}
                Unique Values: {df[column].nunique()}
                Sample Values: {df[column].dropna().head(5).tolist()}
                """
                representations.append(col_summary)
            
            # 3. Data chunks for larger CSVs
            if len(df) > 50:
                chunk_size = 50
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk_content = f"""
                    Data Chunk {i//chunk_size + 1} (Rows {i+1}-{min(i+chunk_size, len(df))}):
                    {chunk.to_string()}
                    """
                    representations.append(chunk_content)
            else:
                # Full data for small CSVs
                full_data_content = f"""
                Complete Data:
                {df.to_string()}
                """
                representations.append(full_data_content)
            
            # Create document entries for each representation
            for i, content in enumerate(representations):
                csv_docs.append({
                    'source': f"{os.path.basename(csv_path)}_rep_{i+1}",
                    'content': content,
                    'type': 'csv',
                    'original_file': os.path.basename(csv_path),
                    'representation_type': ['summary', 'column_wise', 'data_chunk'][min(i, 2)],
                    'file_path': csv_path,
                    'total_rows': len(df),
                    'total_columns': len(df.columns)
                })
                
            print(f"Loaded {len(csv_docs)} CSV representations from {csv_path}")
            
        except Exception as e:
            print(f"Error processing CSV {csv_path}: {e}")
            
        return csv_docs
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better context"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
                
            start = end - overlap  # Overlap for context continuity
            
        return chunks
    
    def create_vector_store(self, documents: List[Dict[str, Any]], save_locally: bool = True):
        """Create FAISS vector store from documents using Bedrock embeddings"""
        if not documents:
            print("No documents to process")
            return
            
        print(f"Creating vector store with {len(documents)} documents...")
        
        embeddings = []
        valid_documents = []
        
        for i, doc in enumerate(documents):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing document {i+1}/{len(documents)}")
                
            try:
                embedding = self.get_bedrock_embedding(doc['content'])
                
                if embedding and len(embedding) > 0:
                    embeddings.append(embedding)
                    valid_documents.append(doc)
                else:
                    print(f"Skipping document {doc['source']} - empty embedding")
                    
            except Exception as e:
                print(f"Error generating embedding for {doc['source']}: {e}")
        
        if not embeddings:
            print("No valid embeddings generated")
            return
            
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        self.documents = valid_documents
        
        print(f"✅ Vector store created with {len(valid_documents)} documents")
        print(f"📊 Embedding dimension: {dimension}")
        
        # Save to local storage
        if save_locally:
            self._save_vector_store()
    
    def _save_vector_store(self):
        """Save the vector store to local storage"""
        try:
            # Save FAISS index
            index_path = os.path.join(self.storage_dir, "faiss_index.index")
            faiss.write_index(self.index, index_path)
            
            # Save documents metadata
            documents_path = os.path.join(self.storage_dir, "documents.pkl")
            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save metadata
            metadata = {
                "total_documents": len(self.documents),
                "embedding_dimension": self.index.d if self.index else 0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "document_types": self._get_document_types(),
                "sources": self._get_sources()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Vector store saved to {self.storage_dir}")
            print(f"   - Index: {index_path}")
            print(f"   - Documents: {documents_path}")
            print(f"   - Metadata: {self.metadata_file}")
            
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def load_vector_store(self) -> bool:
        """Load the vector store from local storage"""
        try:
            index_path = os.path.join(self.storage_dir, "faiss_index.index")
            documents_path = os.path.join(self.storage_dir, "documents.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(documents_path):
                print("Vector store files not found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents
            with open(documents_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    print(f"📁 Loaded vector store with {metadata['total_documents']} documents")
                    print(f"🕐 Created: {metadata['created_at']}")
            
            print(f"✅ Vector store loaded from {self.storage_dir}")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using Bedrock embeddings"""
        if self.index is None or len(self.documents) == 0:
            print("Vector store not initialized or empty")
            return []
        
        try:
            # Get query embedding
            query_embedding = self.get_bedrock_embedding(query)
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_array, min(k, len(self.documents)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    # Convert distance to similarity score (higher is better)
                    similarity_score = 1 / (1 + distances[0][i]) if distances[0][i] > 0 else 1.0
                    
                    results.append({
                        'document': self.documents[idx],
                        'score': similarity_score,
                        'distance': float(distances[0][i])
                    })
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"🔍 Found {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def _get_document_types(self) -> Dict[str, int]:
        """Get document type statistics"""
        doc_types = {}
        for doc in self.documents:
            doc_type = doc.get('type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        return doc_types
    
    def _get_sources(self) -> Dict[str, int]:
        """Get source file statistics"""
        sources = {}
        for doc in self.documents:
            source = doc.get('original_file', doc.get('source', 'unknown'))
            sources[source] = sources.get(source, 0) + 1
        return sources
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded documents"""
        if not self.documents:
            return {"total_documents": 0, "vector_store_ready": False}
        
        return {
            "total_documents": len(self.documents),
            "document_types": self._get_document_types(),
            "sources": self._get_sources(),
            "vector_store_ready": self.index is not None,
            "embedding_dimension": self.index.d if self.index else 0,
            "storage_location": self.storage_dir
        }
    
    def is_vector_store_loaded(self) -> bool:
        """Check if vector store is loaded"""
        return self.index is not None and len(self.documents) > 0
    
    def clear_vector_store(self):
        """Clear the current vector store from memory"""
        self.index = None
        self.documents = []
        print("Vector store cleared from memory")
    
    def delete_local_store(self):
        """Delete the local vector store files"""
        try:
            import shutil
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)
                print(f"🗑️  Local vector store deleted: {self.storage_dir}")
            else:
                print("Local vector store directory does not exist")
        except Exception as e:
            print(f"Error deleting local store: {e}")