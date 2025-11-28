import os
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------
# CREATE VECTOR STORE FROM TEXT
# ---------------------------
def create_vector_store_from_text(text: str):
    """
    Takes plain text extracted from PDF and builds a Chroma vector DB.
    Returns the Chroma vectorstore instance (already persisted to directory).
    """
    os.makedirs(CHROMA_DIR, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)
    docs: List[Document] = [Document(page_content=c) for c in chunks]

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Create Chroma vectorstore from Documents. persist_directory ensures persistence.
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory=CHROMA_DIR,
    )

    # NOTE: some Chroma variants don't implement a .persist() method;
    # using persist_directory on creation is the most compatible approach.
    return vectordb

# ---------------------------
# LOAD EXISTING VECTOR STORE
# ---------------------------
def load_vector_store():
    """
    Loads the Chroma vector DB if it exists. Returns a Chroma instance.
    """
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedder,
    )
    return vectordb

# ---------------------------
# RETRIEVE TOP-K CHUNKS
# ---------------------------
def retrieve_relevant_chunks(query: str, k: int = 3):
    vectordb = load_vector_store()
    # similarity_search returns a list of langchain Document objects
    docs = vectordb.similarity_search(query, k=k)
    return docs
