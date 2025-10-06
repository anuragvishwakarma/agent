# app.py
import streamlit as st
import os
from data_loader.document_processor import DocumentProcessor
from agents.multi_agent_system import MultiAgentSystem

st.set_page_config(page_title="AI Agents", page_icon="🤖")

@st.cache_resource
def load_system():
    """Load the document processor and agent system"""
    processor = DocumentProcessor()
    # Try to load existing vector store
    if not processor.load_vector_store():
        st.info("No existing data found. Please add PDF/CSV files to 'data/' folder.")
    return MultiAgentSystem(processor)

def main():
    st.title("🤖 AI Agents Chat")
    st.write("Ask about maintenance, field support, or workload management")
    
    # Initialize system
    if "system" not in st.session_state:
        with st.spinner("Loading AI agents..."):
            try:
                st.session_state.system = load_system()
            except Exception as e:
                st.error(f"Failed to load: {e}")
                return
    
    # Chat interface
    question = st.text_input("Your question:")
    
    if question and st.button("Ask Agents"):
        with st.spinner("Agents are thinking..."):
            try:
                response = st.session_state.system.invoke(question)
                
                # Show final response
                st.write("### 🤖 Response:")
                st.write(response.get("final_response", "No response"))
                
                # Show individual agents in expander
                with st.expander("See individual agent responses"):
                    if "maintenance_response" in response:
                        st.write("**🛠️ Maintenance:**", response["maintenance_response"])
                    if "field_response" in response:
                        st.write("**🔧 Field Support:**", response["field_response"])
                    if "workload_response" in response:
                        st.write("**📈 Workload:**", response["workload_response"])
                        
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()