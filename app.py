# app.py (updated section for vector store management)
import streamlit as st
import os
from data_loader.document_processor import DocumentProcessor
from agents.multi_agent_system import MultiAgentSystem
import time

class StreamlitApp:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.multi_agent_system = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the system with vector store management"""
        try:
            with st.spinner("🚀 Initializing AI Agents in SageMaker environment..."):
                # Try to load existing vector store first
                if self.document_processor.load_vector_store():
                    st.success("✅ Loaded existing vector store from local storage")
                else:
                    st.info("📁 No existing vector store found. Creating new one...")
                    
                    # Load documents
                    pdf_docs = self.document_processor.load_pdf_documents("data/")
                    csv_docs = self.document_processor.load_csv_documents("data/your_data.csv")  # Update path
                    
                    all_docs = pdf_docs + csv_docs
                    
                    if all_docs:
                        self.document_processor.create_vector_store(all_docs, save_locally=True)
                        st.success(f"✅ Created new vector store with {len(all_docs)} documents")
                    else:
                        st.error("❌ No documents found to process")
                        return
                
                # Initialize multi-agent system
                self.multi_agent_system = MultiAgentSystem(self.document_processor)
                
            # Show statistics
            stats = self.document_processor.get_document_stats()
            st.sidebar.success(f"📊 {stats['total_documents']} documents loaded")
            
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
    
    def render_sidebar(self):
        """Render the sidebar with vector store management"""
        st.sidebar.title("🤖 SageMaker Agent System")
        st.sidebar.markdown('<span class="sagemaker-badge">Amazon SageMaker</span>', unsafe_allow_html=True)
        
        # Vector Store Management
        st.sidebar.markdown("### 🗃️ Vector Store")
        
        if self.document_processor.is_vector_store_loaded():
            stats = self.document_processor.get_document_stats()
            st.sidebar.info(f"""
            **Status**: ✅ Loaded
            **Documents**: {stats['total_documents']}
            **Location**: `{stats['storage_location']}`
            """)
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🔄 Reload"):
                    if self.document_processor.load_vector_store():
                        st.sidebar.success("Reloaded!")
                    else:
                        st.sidebar.error("Reload failed!")
            with col2:
                if st.button("🗑️ Clear"):
                    self.document_processor.clear_vector_store()
                    st.sidebar.info("Cleared from memory")
        else:
            st.sidebar.warning("**Status**: ❌ Not Loaded")
        
        st.sidebar.markdown("### Available Agents")
        st.sidebar.info("""
        - **Maintenance Scheduler**: Plans and optimizes maintenance operations
        - **Field Support Agent**: Provides technical field support
        - **Workload Manager**: Manages resource allocation and workload
        """)
        
        # Rest of the sidebar code remains the same...