# test_bedrock.py
import boto3
from langchain_aws import ChatBedrock, BedrockLLM
import streamlit as st

def test_bedrock_access():
    st.title("🔧 Bedrock Access Test")
    
    # Test 1: Basic boto3 client
    st.subheader("Test 1: Boto3 Client")
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')
        models = bedrock_runtime.list_foundation_models()
        st.success("✅ Boto3 Bedrock client works!")
        st.write(f"Available models: {len(models.get('modelSummaries', []))}")
    except Exception as e:
        st.error(f"❌ Boto3 client failed: {e}")
    
    # Test 2: ChatBedrock
    st.subheader("Test 2: ChatBedrock")
    try:
        chat_llm = ChatBedrock(
            model_id="us.amazon.nova-pro-v1:0",
            region_name="us-west-2"
        )
        st.success("✅ ChatBedrock initialized!")
        
        # Test invocation
        from langchain.schema import HumanMessage
        response = chat_llm.invoke([HumanMessage(content="Hello, say 'TEST OK'")])
        st.success(f"✅ ChatBedrock invocation works: {response.content}")
    except Exception as e:
        st.error(f"❌ ChatBedrock failed: {e}")
    
    # Test 3: BedrockLLM
    st.subheader("Test 3: BedrockLLM")
    try:
        llm = BedrockLLM(
            model_id="us.amazon.nova-pro-v1:0",
            region_name="us-west-2"
        )
        st.success("✅ BedrockLLM initialized!")
        
        response = llm.invoke("Say 'TEST OK'")
        st.success(f"✅ BedrockLLM invocation works: {response}")
    except Exception as e:
        st.error(f"❌ BedrockLLM failed: {e}")

if __name__ == "__main__":
    test_bedrock_access()