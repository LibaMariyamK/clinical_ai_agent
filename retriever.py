#retriever.py

import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
DB_PATH = "./chroma_db_data" # Directory to save the vector database
PDF_PATH = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition.pdf"


@st.cache_resource
def load_embedding_model():
    """Loads the embedding model for vectorization."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_retriever():
    """
    Initializes or loads the Vector Database (ChromaDB).
    Returns a retriever object used to search the PDF.
    """
    embedding_model = load_embedding_model()
    
    # 1. Check if DB exists
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        # Load existing chroma db
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        print("Loaded existing Vector Store.")
    else:
        # 2. Create DB if it doesn't exist
        if not os.path.exists(PDF_PATH):
            st.error(f"PDF file not found : {PDF_PATH}")
            return None
        with st.spinner("Indexing PDF document (this may take a while)..."):
            # Load the document
            loader = PyMuPDFLoader(PDF_PATH)
            doc = loader.load()[30:-439] # We excluded the pages at the beginning and end that are useless
            # Split the text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(doc)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                persist_directory=DB_PATH
            )
        print("Created new Vector Store.")
    return vectorstore.as_retriever()


