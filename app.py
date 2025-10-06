# app.py
import streamlit as st
import os
from data_loader.document_processor import DocumentProcessor
from agents.multi_agent_system import MultiAgentSystem
import time

# Page configuration
st.set_page_config(
    page_title="AI Agentic Operations System",
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
    }
    .final-response {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.document_processor = None
        self.multi_agent_system = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the document processing and agent system"""
        try:
            with st.spinner("Initializing AI Agents and loading documents..."):
                self.document_processor = DocumentProcessor()
                
                # Load documents
                pdf_docs = self.document_processor.load_pdf_documents("data/")
                csv_docs = self.document_processor.load_csv_documents("data/")  # Update with actual CSV path
                
                all_docs = pdf_docs + csv_docs
                self.document_processor.create_vector_store(all_docs)
                
                self.multi_agent_system = MultiAgentSystem(self.document_processor)
                
            st.success(f"✅ System initialized with {len(all_docs)} documents")
        except Exception as e:
            st.error(f"❌ System initialization failed: {str(e)}")
    
    def render_sidebar(self):
        """Render the sidebar with agent information and controls"""
        st.sidebar.title("🤖 Agent Configuration")
        
        st.sidebar.markdown("### Available Agents")
        st.sidebar.info("""
        - **Maintenance Scheduler**: Plans and optimizes maintenance operations
        - **Field Support Agent**: Provides technical field support
        - **Workload Manager**: Manages resource allocation and workload
        """)
        
        st.sidebar.markdown("### System Status")
        if self.multi_agent_system:
            st.sidebar.success("🟢 System Ready")
        else:
            st.sidebar.error("🔴 System Offline")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Query Examples")
        example_queries = [
            "Schedule preventive maintenance for all critical equipment for next quarter",
            "We have a field issue with equipment X showing error code Y, what should we do?",
            "Optimize workload distribution across three teams for the upcoming project",
            "Create a maintenance plan considering current workload and field constraints"
        ]
        
        for i, example in enumerate(example_queries):
            if st.sidebar.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.user_query = example
    
    def render_main_content(self):
        """Render the main content area"""
        st.markdown('<div class="main-header">🏭 AI Agentic Operations System</div>', unsafe_allow_html=True)
        
        # Query input
        user_query = st.text_area(
            "Enter your operational query:",
            value=st.session_state.get("user_query", ""),
            height=100,
            placeholder="e.g., Schedule maintenance for equipment X considering current workload and field team availability..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            process_query = st.button("🚀 Process with All Agents", use_container_width=True)
        
        if process_query and user_query:
            self.process_user_query(user_query)
    
    def process_user_query(self, query: str):
        """Process the user query using the multi-agent system"""
        try:
            with st.spinner("🤖 Agents are collaborating on your query..."):
                start_time = time.time()
                response = self.multi_agent_system.invoke(query)
                processing_time = time.time() - start_time
            
            # Display individual agent responses
            st.markdown("## 📊 Agent Responses")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🛠️ Maintenance Scheduler")
                st.markdown(f'<div class="agent-response">{response["maintenance_scheduler"]}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🔧 Field Support")
                st.markdown(f'<div class="agent-response">{response["field_support"]}</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown("### 📈 Workload Manager")
                st.markdown(f'<div class="agent-response">{response["workload_manager"]}</div>', unsafe_allow_html=True)
            
            # Display final consolidated response
            st.markdown("## 🎯 Consolidated Recommendation")
            st.markdown(f'<div class="final-response">{response["final_response"]}</div>', unsafe_allow_html=True)
            
            # Display performance metrics
            st.markdown("---")
            st.markdown(f"**Processing Time**: {processing_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
    
    def run(self):
        """Run the Streamlit application"""
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        
        self.render_sidebar()
        self.render_main_content()

if __name__ == "__main__":
    # Initialize AWS credentials (set these in your environment)
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    
    app = StreamlitApp()
    app.run()