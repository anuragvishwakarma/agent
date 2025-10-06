# app.py
import streamlit as st
import os
import time
import traceback
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
    .debug-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4488ff;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.document_processor = None
        self.multi_agent_system = None
        self.initialized = False
        self.debug_mode = True  # Enable debug mode
        
    def debug_log(self, message):
        """Print debug messages if debug mode is enabled"""
        if self.debug_mode:
            print(f"DEBUG: {message}")
    
    def initialize_system(self):
        """Initialize the document processing and agent system with detailed debugging"""
        try:
            self.debug_log("Starting system initialization...")
            
            with st.spinner("🚀 Initializing AI Agents in SageMaker environment..."):
                
                # Step 1: Initialize Document Processor
                self.debug_log("Step 1: Creating DocumentProcessor...")
                try:
                    self.document_processor = DocumentProcessor()
                    self.debug_log("✅ DocumentProcessor created successfully")
                except Exception as e:
                    st.error(f"❌ Failed to create DocumentProcessor: {str(e)}")
                    return False
                
                # Step 2: Try to load existing vector store
                self.debug_log("Step 2: Attempting to load existing vector store...")
                try:
                    if self.document_processor.load_vector_store():
                        st.success("✅ Loaded existing vector store from local storage")
                        self.debug_log("✅ Vector store loaded successfully")
                    else:
                        st.info("📁 No existing vector store found. Creating new one...")
                        self.debug_log("❌ No existing vector store found")
                        
                        # Step 3: Load documents
                        data_dir = "data/"
                        pdf_docs = []
                        csv_docs = []
                        
                        # Check if data directory exists
                        self.debug_log(f"Checking data directory: {data_dir}")
                        if not os.path.exists(data_dir):
                            st.error(f"❌ Data directory '{data_dir}' not found")
                            self.debug_log(f"❌ Data directory {data_dir} does not exist")
                            return False
                        
                        # List files in data directory
                        files = os.listdir(data_dir)
                        self.debug_log(f"Files in data directory: {files}")
                        
                        # Load PDFs
                        pdf_files = [f for f in files if f.endswith('.pdf')]
                        if pdf_files:
                            st.info(f"📄 Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
                            self.debug_log(f"Loading PDFs: {pdf_files}")
                            pdf_docs = self.document_processor.load_pdf_documents(data_dir)
                            st.success(f"✅ Loaded {len(pdf_docs)} PDF chunks")
                            self.debug_log(f"✅ Loaded {len(pdf_docs)} PDF chunks")
                        else:
                            st.warning("⚠️ No PDF files found in data directory")
                            self.debug_log("⚠️ No PDF files found")
                        
                        # Load CSV files
                        csv_files = [f for f in files if f.endswith('.csv')]
                        csv_loaded = False
                        if csv_files:
                            st.info(f"📊 Found {len(csv_files)} CSV files: {', '.join(csv_files)}")
                            self.debug_log(f"Loading CSVs: {csv_files}")
                            for csv_file in csv_files:
                                csv_path = os.path.join(data_dir, csv_file)
                                self.debug_log(f"Attempting to load CSV: {csv_path}")
                                csv_docs = self.document_processor.load_csv_documents(csv_path)
                                if csv_docs:
                                    st.success(f"✅ Loaded {len(csv_docs)} CSV representations from {csv_file}")
                                    self.debug_log(f"✅ Successfully loaded CSV: {csv_file}")
                                    csv_loaded = True
                                    break
                                else:
                                    self.debug_log(f"❌ Failed to load CSV: {csv_file}")
                        
                        if not csv_loaded and csv_files:
                            st.warning("⚠️ CSV files found but could not be processed")
                            self.debug_log("⚠️ CSV files found but processing failed")
                        elif not csv_files:
                            st.warning("⚠️ No CSV files found in data directory")
                            self.debug_log("⚠️ No CSV files found")
                        
                        all_docs = pdf_docs + csv_docs
                        self.debug_log(f"Total documents to process: {len(all_docs)}")
                        
                        if all_docs:
                            with st.spinner("🔄 Creating vector store (this may take a while due to rate limits)..."):
                                self.debug_log("Creating vector store...")
                                try:
                                    self.document_processor.create_vector_store(all_docs, save_locally=True)
                                    st.success(f"✅ Created new vector store with {len(all_docs)} documents")
                                    self.debug_log("✅ Vector store created successfully")
                                except Exception as e:
                                    st.error(f"❌ Failed to create vector store: {str(e)}")
                                    self.debug_log(f"❌ Vector store creation failed: {str(e)}")
                                    return False
                        else:
                            st.error("❌ No documents found to process")
                            self.debug_log("❌ No documents available for processing")
                            return False
                
                except Exception as e:
                    st.error(f"❌ Error during vector store loading: {str(e)}")
                    self.debug_log(f"❌ Vector store loading error: {str(e)}")
                    return False
                
                # Step 4: Initialize Multi-Agent System
                self.debug_log("Step 4: Initializing MultiAgentSystem...")
                try:
                    st.info("🤖 Initializing multi-agent system...")
                    self.multi_agent_system = MultiAgentSystem(self.document_processor)
                    self.initialized = True
                    self.debug_log("✅ MultiAgentSystem initialized successfully")
                except Exception as e:
                    st.error(f"❌ Failed to initialize multi-agent system: {str(e)}")
                    self.debug_log(f"❌ MultiAgentSystem initialization failed: {str(e)}")
                    traceback.print_exc()  # Print full traceback for debugging
                    return False
                
                # Step 5: Show final statistics
                self.debug_log("Step 5: Displaying final statistics...")
                try:
                    stats = self.document_processor.get_document_stats()
                    st.sidebar.success(f"📊 {stats['total_documents']} documents loaded")
                    self.debug_log(f"✅ Final stats: {stats}")
                    
                    st.success("🎉 System initialized successfully!")
                    self.debug_log("✅ System initialization completed successfully")
                    return True
                    
                except Exception as e:
                    st.error(f"❌ Error displaying statistics: {str(e)}")
                    self.debug_log(f"❌ Statistics display error: {str(e)}")
                    return False
                
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
            self.debug_log(f"❌ Overall initialization failed: {str(e)}")
            traceback.print_exc()  # Print full traceback
            
            # Show detailed error information
            with st.expander("🔍 Detailed Error Information", expanded=True):
                st.code(traceback.format_exc())
            
            st.info("💡 Troubleshooting steps:")
            st.info("1. Check if 'data/' directory exists with PDF/CSV files")
            st.info("2. Verify AWS Bedrock access permissions")
            st.info("3. Check network connectivity")
            st.info("4. Look at the debug output in the console")
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
                try:
                    stats = self.document_processor.get_document_stats()
                    st.sidebar.info(f"""
                    **Documents Loaded**: {stats['total_documents']}
                    **Vector Store**: ✅ Ready
                    **Storage**: `{stats['storage_location']}`
                    """)
                except Exception as e:
                    st.sidebar.error(f"Error getting stats: {str(e)}")
        else:
            st.sidebar.error("🔴 System Offline")
            st.sidebar.markdown("""
            **To initialize the system:**
            1. Ensure 'data/' directory exists
            2. Add PDF/CSV files to data directory
            3. Click Initialize below
            """)
            
            if st.sidebar.button("🔄 Initialize System", use_container_width=True, type="primary"):
                with st.spinner("Initializing system..."):
                    if self.initialize_system():
                        st.rerun()
                    else:
                        st.sidebar.error("Initialization failed. Check console for details.")
        
        st.sidebar.markdown("---")
        
        # Debug Section
        if self.debug_mode:
            with st.sidebar.expander("🐛 Debug Info", expanded=False):
                if self.document_processor:
                    try:
                        stats = self.document_processor.get_document_stats()
                        st.write("Document Stats:", stats)
                    except:
                        st.write("No document stats available")
                
                st.write("Initialized:", self.initialized)
                st.write("MultiAgentSystem:", "Ready" if self.multi_agent_system else "None")
                
                if st.button("Clear Vector Store Cache"):
                    if self.document_processor:
                        self.document_processor.delete_local_store()
                        st.success("Cache cleared!")
        
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
        
        # Query Examples
        st.sidebar.markdown("### 💡 Query Examples")
        example_queries = [
            "Schedule preventive maintenance for equipment",
            "We have a field issue with equipment showing error codes",
            "Optimize workload distribution across teams",
            "Create a maintenance plan considering current workload"
        ]
        
        for i, example in enumerate(example_queries):
            if st.sidebar.button(f"Example {i+1}", key=f"example_{i}", use_container_width=True):
                st.session_state.user_query = example
                st.rerun()
    
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
            
            # Show detailed setup instructions
            with st.expander("📋 Detailed Setup Instructions", expanded=True):
                st.markdown("""
                ### Prerequisites:
                
                1. **Data Directory Structure:**
                ```
                your_project/
                ├── app.py
                ├── data/
                │   ├── file1.pdf
                │   ├── file2.pdf
                │   ├── *.pdf (any PDF files)
                │   └── *.csv (any CSV files)
                ├── agents/
                │   ├── __init__.py
                │   ├── multi_agent_system.py
                │   ├── base_agent.py
                │   ├── maintenance_scheduler.py
                │   ├── field_support.py
                │   └── workload_manager.py
                └── data_loader/
                    ├── __init__.py
                    └── document_processor.py
                ```
                
                2. **Required Python Packages:**
                ```bash
                pip install streamlit langchain langgraph boto3 pypdf pandas faiss-cpu sentence-transformers
                ```
                
                3. **AWS Permissions:**
                   - Bedrock model access (Nova Pro, Titan Embeddings)
                   - SageMaker execution role permissions
                """)
            
            # Show current directory status
            st.markdown("### 🔍 Current Directory Status")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Directory Check")
                current_dir = os.getcwd()
                st.write(f"**Current Directory:** `{current_dir}`")
                
                data_dir_exists = os.path.exists("data/")
                st.write(f"**Data Directory:** {'✅ Exists' if data_dir_exists else '❌ Missing'}")
                
                if data_dir_exists:
                    files = os.listdir("data/")
                    st.write(f"**Files in data/:** {len(files)} files")
                    for file in files:
                        st.write(f"  - {file}")
            
            with col2:
                st.subheader("File Check")
                required_dirs = ["agents", "data_loader"]
                for dir_name in required_dirs:
                    exists = os.path.exists(dir_name)
                    st.write(f"**{dir_name}/:** {'✅ Exists' if exists else '❌ Missing'}")
                
                # Check for required Python files
                required_files = [
                    "agents/multi_agent_system.py",
                    "agents/base_agent.py", 
                    "data_loader/document_processor.py"
                ]
                for file_path in required_files:
                    exists = os.path.exists(file_path)
                    st.write(f"**{file_path}:** {'✅ Exists' if exists else '❌ Missing'}")
            
            return
        
        # System is initialized - show query interface
        st.markdown("### 💬 Enter Your Query")
        user_query = st.text_area(
            "Describe your operational challenge or question:",
            value=st.session_state.get("user_query", ""),
            height=120,
            placeholder="e.g., Schedule maintenance for equipment X considering current workload and field team availability...",
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
            st.error("❌ Multi-agent system not initialized.")
            return
        
        try:
            with st.spinner("🤖 Agents are collaborating on your query..."):
                start_time = time.time()
                response = self.multi_agent_system.invoke(query)
                processing_time = time.time() - start_time
            
            # Check for errors
            if "error" in response:
                st.error(f"❌ System Error: {response['error']}")
                return
            
            # Display responses
            st.markdown("## 📊 Agent Responses")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🛠️ Maintenance Scheduler")
                response_text = response.get("maintenance_scheduler_response", "No response")
                st.markdown(f'<div class="agent-response">{response_text}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🔧 Field Support")
                response_text = response.get("field_support_response", "No response")
                st.markdown(f'<div class="agent-response">{response_text}</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown("### 📈 Workload Manager")
                response_text = response.get("workload_manager_response", "No response")
                st.markdown(f'<div class="agent-response">{response_text}</div>', unsafe_allow_html=True)
            
            # Final response
            st.markdown("## 🎯 Consolidated Recommendation")
            final_response = response.get("final_response", "No consolidated response")
            st.markdown(f'<div class="final-response">{final_response}</div>', unsafe_allow_html=True)
            
            # Metrics
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            with col2:
                st.metric("Agents Used", "3")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.code(traceback.format_exc())
    
    def run(self):
        """Run the Streamlit application"""
        # Initialize session state
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        if "system_initialized" not in st.session_state:
            st.session_state.system_initialized = False
        
        # Don't auto-initialize - let user click the button
        self.render_sidebar()
        self.render_main_content()

def main():
    """Main function to run the app"""
    st.title("🤖 AI Agentic Operations System")
    
    # Create and run the app
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()