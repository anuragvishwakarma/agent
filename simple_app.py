# simple_agent.py
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage
import streamlit as st

@st.cache_resource
def get_agent():
    return ChatBedrock(
        model_id="amazon.nova-pro-v1:0",
        region_name="us-west-2",
        model_kwargs={"temperature": 0.1, "max_tokens": 2000}
    )

def main():
    st.title("🤖 AI Assistant")
    question = st.text_input("Ask me anything:")
    
    if question and st.button("Ask"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                response = agent.invoke([HumanMessage(content=question)])
                st.write(response.content)
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()