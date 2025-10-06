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
import random
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
    
    def get_bedrock_embedding(self, text: str, max_retries: int = 5) -> List[float]:
        """Get embeddings using Amazon Titan Embeddings via Bedrock with retry logic"""
        for attempt in range(max_retries):
            try:
                # Clean and prepare text
                clean_text = text.replace('\n', ' ').strip()
                if len(clean_text) > 10000:  # Bedrock limit
                    clean_text = clean_text[:10000]
                
                body = json.dumps({
                    "inputText": clean_text
                })
                
                # Add exponential backoff with jitter
                if attempt > 0:
                    base_delay = min(2 ** attempt + random.uniform(0, 1), 60)  # Max 60 seconds
                    print(f"🔄 Bedrock retry attempt {attempt + 1}/{max_retries}, waiting {base_delay:.2f}s")
                    time.sleep(base_delay)
                
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
                
                # Small delay between successful calls to avoid throttling
                time.sleep(0.2)
                return embedding
                    
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Bedrock embedding attempt {attempt + 1} failed: {error_msg}")
                
                # Check if it's a throttling error
                if "ThrottlingException" in error_msg and attempt < max_retries - 1:
                    continue
                elif "AccessDenied" in error_msg:
                    print("🔐 Access denied - check Bedrock model access permissions")
                    break
                else:
                    print(f"❌ Non-retryable error: {error_msg}")
                    break
        
        # If all retries failed, use fallback
        print("🔄 Switching to fallback embeddings after Bedrock failures")
        return self._get_fallback_embedding(text)
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding using sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a smaller model for faster loading
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text).tolist()
            print("✅ Using local fallback embeddings")
            return embedding
        except Exception as e:
            print(f"❌ Fallback embedding failed: {e}")
            # Return zero vector as last resort
            return [0.0] * 384
    
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
        """Load and process CSV documents with robust error handling"""
        csv_docs = []
        
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} does not exist")
            return csv_docs
            
        try:
            # Try different CSV reading strategies
            df = None
            strategies = [
                # Strategy 1: Standard read
                lambda: pd.read_csv(csv_path),
                # Strategy 2: Handle different separators
                lambda: pd.read_csv(csv_path, sep=None, engine='python'),
                # Strategy 3: Skip bad lines
                lambda: pd.read_csv(csv_path, on_bad_lines='skip', engine='python'),
                # Strategy 4: Manual parsing as last resort
                lambda: self._manual_csv_parse(csv_path)
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    df = strategy()
                    print(f"✅ CSV loaded successfully with strategy {i+1}")
                    break
                except Exception as e:
                    print(f"❌ Strategy {i+1} failed: {e}")
                    continue
            
            if df is None or df.empty:
                print(f"❌ All CSV loading strategies failed for {csv_path}")
                # Try raw text fallback
                return self._load_csv_as_raw_text(csv_path)
                
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
                Sample Values: {df[column].dropna().head(3).tolist()}
                """
                representations.append(col_summary)
            
            # 3. Statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                stats_content = f"""
                Statistical Summary:
                {df[numeric_cols].describe().to_string()}
                """
                representations.append(stats_content)
            
            # 4. Data chunks for larger CSVs
            if len(df) > 20:
                chunk_size = 10
                for i in range(0, min(len(df), 50), chunk_size):  # Limit to first 50 rows
                    chunk = df.iloc[i:i+chunk_size]
                    chunk_content = f"""
                    Data Chunk {i//chunk_size + 1} (Rows {i+1}-{min(i+chunk_size, len(df))}):
                    {chunk.to_string(max_rows=10, max_cols=10)}
                    """
                    representations.append(chunk_content)
            else:
                # Full data for small CSVs
                full_data_content = f"""
                Complete Data (first 20 rows):
                {df.head(20).to_string(max_rows=20, max_cols=10)}
                """
                representations.append(full_data_content)
            
            # Create document entries for each representation
            for i, content in enumerate(representations):
                csv_docs.append({
                    'source': f"{os.path.basename(csv_path)}_rep_{i+1}",
                    'content': content.strip(),
                    'type': 'csv',
                    'original_file': os.path.basename(csv_path),
                    'representation_type': ['summary', 'column_wise', 'statistics', 'data_chunk'][min(i, 3)],
                    'file_path': csv_path,
                    'total_rows': len(df),
                    'total_columns': len(df.columns)
                })
                
            print(f"✅ Loaded {len(csv_docs)} CSV representations from {csv_path}")
            
        except Exception as e:
            print(f"❌ Error processing CSV {csv_path}: {e}")
            # Try raw text fallback
            return self._load_csv_as_raw_text(csv_path)
            
        return csv_docs

    def _manual_csv_parse(self, csv_path: str) -> pd.DataFrame:
        """Manual CSV parsing as last resort"""
        import csv
        rows = []
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i < 100:  # Limit to first 100 rows
                        rows.append(row)
                    else:
                        break
            
            # Create DataFrame from rows
            if len(rows) > 1:
                # Use first row as header, rest as data
                df = pd.DataFrame(rows[1:], columns=rows[0])
                return df
            elif len(rows) == 1:
                # Only header row
                df = pd.DataFrame(columns=rows[0])
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Manual CSV parsing failed: {e}")
            return pd.DataFrame()

    def _load_csv_as_raw_text(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load CSV as raw text when all parsing fails"""
        csv_docs = []
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)  # Read first 5000 characters
            
            csv_docs.append({
                'source': os.path.basename(csv_path),
                'content': f"Raw CSV content (first 5000 chars): {content}",
                'type': 'csv',
                'original_file': os.path.basename(csv_path),
                'file_path': csv_path,
                'representation_type': 'raw_text'
            })
            print("✅ Loaded CSV as raw text fallback")
        except Exception as fallback_error:
            print(f"❌ Even raw text fallback failed: {fallback_error}")
            
        return csv_docs

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better context"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == text_length:
                break
                
            start = end - overlap  # Overlap for context continuity
            if start < 0:
                start = 0
                
        return chunks
    
    def create_vector_store(self, documents: List[Dict[str, Any]], save_locally: bool = True, batch_delay: float = 0.5):
        """Create FAISS vector store with rate limiting"""
        if not documents:
            print("No documents to process")
            return
            
        print(f"Creating vector store with {len(documents)} documents...")
        
        embeddings = []
        valid_documents = []
        
        for i, doc in enumerate(documents):
            if i % 5 == 0:  # More frequent progress updates
                print(f"📄 Processing document {i+1}/{len(documents)}")
                
            try:
                # Add delay between batches to avoid throttling
                if i > 0 and i % 10 == 0:
                    print(f"⏳ Batch delay of {batch_delay}s to avoid throttling...")
                    time.sleep(batch_delay)
                
                embedding = self.get_bedrock_embedding(doc['content'])
                
                if embedding and len(embedding) > 0:
                    embeddings.append(embedding)
                    valid_documents.append(doc)
                else:
                    print(f"⚠️ Skipping document {doc['source']} - empty embedding")
                    
            except Exception as e:
                print(f"❌ Error generating embedding for {doc['source']}: {e}")
                # Continue with other documents even if one fails
        
        if not embeddings:
            print("❌ No valid embeddings generated")
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
        """Search for relevant documents with retry logic"""
        if self.index is None or len(self.documents) == 0:
            print("Vector store not initialized or empty")
            return []
        
        for attempt in range(3):  # Retry search up to 3 times
            try:
                # Get query embedding with retry
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
                
                print(f"🔍 Found {len(results)} relevant documents for query")
                return results
                
            except Exception as e:
                print(f"❌ Search attempt {attempt + 1} failed: {e}")
                if attempt < 2:  # Not the last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
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