# app.py
import streamlit as st
from data_loader.document_processor import DocumentProcessor
from agents.multi_agent_system import MultiAgentSystem

st.set_page_config(page_title="AI Agents", page_icon="🤖")

def main():
    st.title("🤖 AI Agents Chat")
    
    # Initialize system
    if "system" not in st.session_state:
        with st.spinner("Loading AI agents..."):
            try:
                # Initialize document processor
                processor = DocumentProcessor()
                if not processor.load_vector_store():
                    st.info("No vector store found. Please add documents to 'data/' folder.")
                
                # Initialize multi-agent system
                st.session_state.system = MultiAgentSystem(processor)
                st.success("✅ System loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                st.info("Run the Bedrock test above to debug permissions.")
                return
    
    # Chat interface
    question = st.text_input("Your question:")
    
    if question and st.button("Ask Agents"):
        with st.spinner("Agents are thinking..."):
            try:
                response = st.session_state.system.invoke(question)
                
                # Show responses
                st.write("### 🤖 Response:")
                
                if "error" in response:
                    st.error(response["error"])
                else:
                    st.write(response.get("final_response", "No response"))
                    
                    # Show individual agents
                    with st.expander("See agent responses"):
                        agents = {
                            "🛠️ Maintenance": "maintenance_response",
                            "🔧 Field Support": "field_support_response", 
                            "📈 Workload": "workload_response"
                        }
                        
                        for name, key in agents.items():
                            if key in response:
                                st.write(f"**{name}:** {response[key]}")
                                
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()