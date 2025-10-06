# app.py
import streamlit as st
import os
import time
from data_loader.document_processor import DocumentProcessor
from agents.multi_agent_system import MultiAgentSystem

# Page config
st.set_page_config(
    page_title="AI Agents Chat",
    page_icon="🤖",
    layout="centered"
)

# Simple CSS
st.markdown("""
<style>
    .agent-message {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
    }
    .final-response {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class SimpleChatApp:
    def __init__(self):
        self.system_ready = False
        self.document_processor = None
        self.multi_agent_system = None
        
    def initialize_system(self):
        """Simple system initialization"""
        try:
            with st.spinner("🔄 Initializing AI agents..."):
                # Initialize document processor
                self.document_processor = DocumentProcessor()
                
                # Try to load existing vector store
                if not self.document_processor.load_vector_store():
                    # Create new vector store
                    data_dir = "data/"
                    if os.path.exists(data_dir):
                        pdf_docs = self.document_processor.load_pdf_documents(data_dir)
                        csv_docs = []
                        
                        # Find and load CSV files
                        for file in os.listdir(data_dir):
                            if file.endswith('.csv'):
                                csv_path = os.path.join(data_dir, file)
                                csv_docs = self.document_processor.load_csv_documents(csv_path)
                                break
                        
                        all_docs = pdf_docs + csv_docs
                        if all_docs:
                            self.document_processor.create_vector_store(all_docs, save_locally=True)
                
                # Initialize multi-agent system
                self.multi_agent_system = MultiAgentSystem(self.document_processor)
                self.system_ready = True
                return True
                
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            return False
    
    def display_chat(self):
        """Display the chat interface"""
        st.title("🤖 AI Agents Chat")
        st.markdown("Ask anything about maintenance, field support, or workload management")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-message"><strong>AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Type your question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            st.markdown(f'<div class="user-message"><strong>You:</strong> {prompt}</div>', unsafe_allow_html=True)
            
            # Get AI response
            with st.spinner("🤖 Agents are thinking..."):
                try:
                    response = self.multi_agent_system.invoke(prompt)
                    
                    # Display agent responses
                    if "error" not in response:
                        # Show individual agent responses
                        agents = {
                            "🛠️ Maintenance": response.get("maintenance_scheduler_response", ""),
                            "🔧 Field Support": response.get("field_support_response", ""),
                            "📈 Workload Manager": response.get("workload_manager_response", "")
                        }
                        
                        for agent_name, agent_response in agents.items():
                            if agent_response and "No response" not in agent_response:
                                st.markdown(f'<div class="agent-message"><strong>{agent_name}:</strong> {agent_response}</div>', unsafe_allow_html=True)
                        
                        # Show final consolidated response
                        final_response = response.get("final_response", "")
                        if final_response:
                            st.markdown(f'<div class="final-response"><strong>🎯 Final Recommendation:</strong><br>{final_response}</div>', unsafe_allow_html=True)
                            
                            # Add final response to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f"Final Recommendation: {final_response}"
                            })
                    else:
                        st.error(f"Error: {response['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    def run(self):
        """Run the simple chat app"""
        # Sidebar for system status
        with st.sidebar:
            st.title("System Status")
            
            if not self.system_ready:
                st.error("🔴 System Offline")
                if st.button("🚀 Initialize System", type="primary"):
                    if self.initialize_system():
                        st.success("🟢 System Ready!")
                        st.rerun()
                    else:
                        st.error("Initialization failed")
            else:
                st.success("🟢 System Ready")
                
                # Quick stats
                if self.document_processor:
                    stats = self.document_processor.get_document_stats()
                    st.info(f"📚 {stats['total_documents']} documents loaded")
                
                st.markdown("---")
                st.markdown("### 🤖 Available Agents")
                st.write("• 🛠️ Maintenance Scheduler")
                st.write("• 🔧 Field Support") 
                st.write("• 📈 Workload Manager")
                
                st.markdown("---")
                if st.button("Clear Chat History"):
                    st.session_state.messages = []
                    st.rerun()
        
        # Main chat interface
        if self.system_ready:
            self.display_chat()
        else:
            st.info("👋 Welcome! Click 'Initialize System' in the sidebar to start chatting with AI agents.")

# Run the app
if __name__ == "__main__":
    app = SimpleChatApp()
    app.run()