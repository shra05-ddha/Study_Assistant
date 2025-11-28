import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama

@st.cache_resource
def get_llm():
    """Return a cached ChatOllama LLM instance."""
    model_name = os.getenv("LLM_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        temperature=0.2,
        max_tokens=800,
        base_url=base_url,
        streaming=False   
    )
    return llm
