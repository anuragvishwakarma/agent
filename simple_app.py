# simple_app.py
import streamlit as st
from data_loader.document_processor import DocumentProcessor
from agents.multi_agent_system import MultiAgentSystem

st.set_page_config(page_title="AI Agents", page_icon="🤖")

# Initialize system
@st.cache_resource
def init_system():
    processor = DocumentProcessor()
    if not processor.load_vector_store():
        # Add your data loading logic here if needed
        pass
    return MultiAgentSystem(processor)

# Main app
st.title("🤖 AI Agents Chat")

# Initialize
if "system" not in st.session_state:
    with st.spinner("Loading AI agents..."):
        try:
            st.session_state.system = init_system()
            st.success("Ready!")
        except Exception as e:
            st.error(f"Failed to load: {e}")

# Chat
if "system" in st.session_state:
    question = st.text_input("Ask about maintenance, field support, or workload:")
    
    if question:
        with st.spinner("Thinking..."):
            response = st.session_state.system.invoke(question)
            
            if "final_response" in response:
                st.write("### 🤖 Response:")
                st.write(response["final_response"])
                
                # Show individual agents if curious
                with st.expander("See individual agent responses"):
                    if "maintenance_scheduler_response" in response:
                        st.write("**🛠️ Maintenance:**", response["maintenance_scheduler_response"])
                    if "field_support_response" in response:
                        st.write("**🔧 Field Support:**", response["field_support_response"])
                    if "workload_manager_response" in response:
                        st.write("**📈 Workload:**", response["workload_manager_response"])