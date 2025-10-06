# app.py
import streamlit as st
import os
import time
from data_loader.document_processor import DocumentProcessor
from agents.multi_agent_system import MultiAgentSystem

# Page configuration
st.set_page_config(
    page_title="AI Agentic Operations System - SageMaker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-response {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .final-response {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sagemaker-badge {
        background: linear-gradient(45deg, #FF9900, #FFB84D);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #44ff44;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffaa44;
        margin: 1rem 0;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.document_processor = None
        self.multi_agent_system = None
        self.initialized = False
        
    def initialize_system(self):
        """Initialize the document processing and agent system"""
        try:
            with st.spinner("🚀 Initializing AI Agents in SageMaker environment..."):
                # Initialize document processor
                self.document_processor = DocumentProcessor()
                
                # Try to load existing vector store first
                if self.document_processor.load_vector_store():
                    st.success("✅ Loaded existing vector store from local storage")
                else:
                    st.info("📁 No existing vector store found. Creating new one...")
                    
                    # Load documents
                    data_dir = "data/"
                    pdf_docs = []
                    csv_docs = []
                    
                    # Check if data directory exists
                    if not os.path.exists(data_dir):
                        st.error(f"❌ Data directory '{data_dir}' not found")
                        return False
                    
                    # Load PDFs
                    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
                    if pdf_files:
                        st.info(f"📄 Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
                        pdf_docs = self.document_processor.load_pdf_documents(data_dir)
                        st.success(f"✅ Loaded {len(pdf_docs)} PDF chunks")
                    else:
                        st.warning("⚠️ No PDF files found in data directory")
                    
                    # Load CSV - try to find any CSV file
                    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                    csv_loaded = False
                    if csv_files:
                        st.info(f"📊 Found {len(csv_files)} CSV files: {', '.join(csv_files)}")
                        for csv_file in csv_files:
                            csv_path = os.path.join(data_dir, csv_file)
                            csv_docs = self.document_processor.load_csv_documents(csv_path)
                            if csv_docs:
                                st.success(f"✅ Loaded {len(csv_docs)} CSV representations from {csv_file}")
                                csv_loaded = True
                                break
                    
                    if not csv_loaded and csv_files:
                        st.warning("⚠️ CSV files found but could not be processed")
                    elif not csv_files:
                        st.warning("⚠️ No CSV files found in data directory")
                    
                    all_docs = pdf_docs + csv_docs
                    
                    if all_docs:
                        with st.spinner("🔄 Creating vector store (this may take a while due to rate limits)..."):
                            self.document_processor.create_vector_store(all_docs, save_locally=True)
                        st.success(f"✅ Created new vector store with {len(all_docs)} documents")
                    else:
                        st.error("❌ No documents found to process")
                        return False
                
                # Initialize multi-agent system
                st.info("🤖 Initializing multi-agent system...")
                self.multi_agent_system = MultiAgentSystem(self.document_processor)
                self.initialized = True
                
                # Show statistics
                stats = self.document_processor.get_document_stats()
                st.sidebar.success(f"📊 {stats['total_documents']} documents loaded")
                
                st.success("🎉 System initialized successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            st.info("💡 Check the following:")
            st.info("   - Ensure 'data/' directory exists with PDF/CSV files")
            st.info("   - Check AWS Bedrock access permissions")
            st.info("   - Verify network connectivity")
            return False
    
    def render_sidebar(self):
        """Render the sidebar with agent information and controls"""
        st.sidebar.title("🤖 AI Agentic Operations")
        st.sidebar.markdown('<span class="sagemaker-badge">Amazon SageMaker</span>', unsafe_allow_html=True)
        
        # System Status
        st.sidebar.markdown("### 🔧 System Status")
        if self.initialized and self.multi_agent_system:
            st.sidebar.success("🟢 System Ready")
            
            # Vector Store Info
            if self.document_processor:
                stats = self.document_processor.get_document_stats()
                st.sidebar.info(f"""
                **Documents Loaded**: {stats['total_documents']}
                **Vector Store**: ✅ Ready
                **Storage**: `{stats['storage_location']}`
                """)
                
                # Document types breakdown
                if stats['document_types']:
                    st.sidebar.markdown("**Document Types:**")
                    for doc_type, count in stats['document_types'].items():
                        st.sidebar.write(f"  - {doc_type}: {count}")
        else:
            st.sidebar.error("🔴 System Offline")
            if st.sidebar.button("🔄 Initialize System", use_container_width=True):
                with st.spinner("Initializing..."):
                    if self.initialize_system():
                        st.rerun()
        
        st.sidebar.markdown("---")
        
        # Agent Information
        st.sidebar.markdown("### 🤖 Available Agents")
        
        agents_info = [
            {"name": "🛠️ Maintenance Scheduler", "desc": "Plans and optimizes maintenance operations"},
            {"name": "🔧 Field Support", "desc": "Provides technical field support"},  
            {"name": "📈 Workload Manager", "desc": "Manages resource allocation and workload"}
        ]
        
        for agent in agents_info:
            with st.sidebar.expander(agent["name"]):
                st.write(agent["desc"])
        
        st.sidebar.markdown("---")
        
        # Models Information
        st.sidebar.markdown("### 🛠️ Models Used")
        st.sidebar.success("""
        🔹 **Amazon Nova Pro** - Primary LLM
        🔹 **Titan Embeddings** - Vector embeddings  
        🔹 **FAISS** - Vector store
        """)
        
        st.sidebar.markdown("---")
        
        # Query Examples
        st.sidebar.markdown("### 💡 Query Examples")
        example_queries = [
            "Schedule preventive maintenance for all critical equipment for next quarter",
            "We have a field issue with equipment X showing error code Y, what should we do?",
            "Optimize workload distribution across three teams for the upcoming project",
            "Create a maintenance plan considering current workload and field constraints"
        ]
        
        for i, example in enumerate(example_queries):
            if st.sidebar.button(f"Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.user_query = example
                st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ System Controls")
        
        if st.sidebar.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
        if st.sidebar.button("🔄 Reload Vector Store", use_container_width=True):
            if self.document_processor and self.document_processor.load_vector_store():
                st.sidebar.success("Vector store reloaded!")
            else:
                st.sidebar.error("Failed to reload vector store")
    
    def render_main_content(self):
        """Render the main content area"""
        st.markdown('<div class="main-header">🏭 AI Agentic Operations System</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666;">Powered by Amazon SageMaker & Bedrock</p>', unsafe_allow_html=True)
        
        # System status indicator
        if not self.initialized or not self.multi_agent_system:
            st.markdown("""
            <div class="warning-box">
            ⚠️ <strong>System not initialized</strong><br>
            Please check the sidebar and click "Initialize System" to start.
            </div>
            """, unsafe_allow_html=True)
            
            # Show initialization instructions
            with st.expander("📋 Setup Instructions", expanded=True):
                st.markdown("""
                1. **Ensure your data files are in the `data/` directory:**
                   - PDF files (`.pdf`) for documentation
                   - CSV files (`.csv`) for structured data
                
                2. **Check AWS permissions:**
                   - Bedrock model access (Nova Pro, Titan Embeddings)
                   - SageMaker execution role permissions
                
                3. **Click "Initialize System" in the sidebar**
                
                4. **Start querying the agents!**
                """)
            
            # Show data directory status
            data_dir = "data/"
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                if files:
                    st.success(f"✅ Data directory found with {len(files)} files")
                    st.write("Files in data directory:")
                    for file in files:
                        st.write(f"  - {file}")
                else:
                    st.warning("⚠️ Data directory exists but is empty")
            else:
                st.error(f"❌ Data directory '{data_dir}' not found")
                
            return
        
        # Query input section
        st.markdown("### 💬 Enter Your Query")
        user_query = st.text_area(
            "Describe your operational challenge or question:",
            value=st.session_state.get("user_query", ""),
            height=120,
            placeholder="e.g., Schedule maintenance for equipment X considering current workload and field team availability...\n\nOr click an example in the sidebar to get started.",
            key="query_input"
        )
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_query = st.button(
                "🚀 Process with All Agents", 
                use_container_width=True, 
                type="primary",
                disabled=not user_query.strip()
            )
        
        if process_query and user_query.strip():
            self.process_user_query(user_query.strip())
    
    def process_user_query(self, query: str):
        """Process the user query using the multi-agent system"""
        if not self.multi_agent_system:
            st.error("❌ Multi-agent system not initialized. Please check system status.")
            return
        
        # Display the query
        st.markdown(f"**Your Query:** {query}")
        st.markdown("---")
        
        try:
            with st.spinner("🤖 Agents are collaborating on your query. This may take a minute..."):
                start_time = time.time()
                response = self.multi_agent_system.invoke(query)
                processing_time = time.time() - start_time
            
            # Check for errors in response
            if "error" in response:
                st.error(f"❌ System Error: {response['error']}")
                return
            
            # Display individual agent responses
            st.markdown("## 📊 Agent Responses")
            
            col1, col2, col3 = st.columns(3)
            
            # Maintenance Scheduler Agent
            with col1:
                st.markdown("### 🛠️ Maintenance Scheduler")
                maintenance_response = response.get("maintenance_scheduler_response", "No response from Maintenance Scheduler")
                st.markdown(f'<div class="agent-response">{maintenance_response}</div>', unsafe_allow_html=True)
            
            # Field Support Agent
            with col2:
                st.markdown("### 🔧 Field Support")
                field_response = response.get("field_support_response", "No response from Field Support")
                st.markdown(f'<div class="agent-response">{field_response}</div>', unsafe_allow_html=True)
            
            # Workload Manager Agent
            with col3:
                st.markdown("### 📈 Workload Manager")
                workload_response = response.get("workload_manager_response", "No response from Workload Manager")
                st.markdown(f'<div class="agent-response">{workload_response}</div>', unsafe_allow_html=True)
            
            # Display final consolidated response
            st.markdown("---")
            st.markdown("## 🎯 Consolidated Recommendation")
            final_response = response.get("final_response", "No consolidated response available")
            st.markdown(f'<div class="final-response">{final_response}</div>', unsafe_allow_html=True)
            
            # Display performance metrics
            st.markdown("---")
            st.markdown("### 📈 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Processing Time", f"{processing_time:.2f} seconds")
            
            with col2:
                agents_used = sum(1 for key in ['maintenance_scheduler_response', 'field_support_response', 'workload_manager_response'] 
                                if response.get(key) and "No response" not in response.get(key, ""))
                st.metric("Agents Responded", agents_used)
            
            with col3:
                if processing_time > 0:
                    st.metric("Response Speed", f"{(len(query) / processing_time):.1f} chars/sec")
                else:
                    st.metric("Response Speed", "N/A")
            
            # Add feedback section
            st.markdown("---")
            st.markdown("### 💬 Feedback")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful Response", use_container_width=True):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("👎 Needs Improvement", use_container_width=True):
                    st.info("We appreciate your feedback. We'll work to improve.")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.markdown("""
            <div class="error-box">
            <strong>💡 Troubleshooting tips:</strong><br>
            - Try reinitializing the system from the sidebar<br>
            - Check if all agents are properly loaded<br>
            - Ensure Bedrock models are accessible<br>
            - Try a simpler query first
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        # Initialize session state
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        if "system_initialized" not in st.session_state:
            st.session_state.system_initialized = False
        
        # Auto-initialize on first run if not already done
        if not self.initialized and not st.session_state.system_initialized:
            # Show welcome message
            st.markdown("""
            <div class="success-box">
            <h3>🎉 Welcome to the AI Agentic Operations System!</h3>
            <p>This system uses three specialized AI agents to help with your operational challenges:</p>
            <ul>
                <li><strong>🛠️ Maintenance Scheduler</strong> - Plans and optimizes maintenance operations</li>
                <li><strong>🔧 Field Support</strong> - Provides technical field support and troubleshooting</li>
                <li><strong>📈 Workload Manager</strong> - Manages resource allocation and workload optimization</li>
            </ul>
            <p>The system is initializing automatically...</p>
            </div>
            """, unsafe_allow_html=True)
            
            if self.initialize_system():
                st.session_state.system_initialized = True
                st.rerun()
            else:
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ Automatic initialization failed</strong><br>
                Please check the sidebar and click "Initialize System" manually.
                </div>
                """, unsafe_allow_html=True)
        
        # Render the main interface
        self.render_sidebar()
        self.render_main_content()

def main():
    """Main function to run the app"""
    # Set page title
    st.title("🤖 AI Agentic Operations System")
    
    # Add some introductory text for first-time users
    if "first_visit" not in st.session_state:
        st.session_state.first_visit = True
    
    # Create and run the app
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()