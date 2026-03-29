# retriever.py

import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
DB_PATH  = "./chroma_db_data"
PDF_PATH = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition.pdf"


@st.cache_resource
def load_embedding_model():
    """
    Loads PubMedBERT embedding model.
    Trained on PubMed biomedical literature — significantly better
    than general-purpose MiniLM for medical terminology retrieval.

    Previous model : sentence-transformers/all-MiniLM-L6-v2  (general)
    Current model  : NeuML/pubmedbert-base-embeddings          (medical)
    """
    return HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_kwargs={"device": "cpu"},       # cpu is fine for inference
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_retriever():
    """
    Initialises or loads the Vector Database (ChromaDB).
    Returns a retriever object used to search the PDF.

    NOTE: If you change the embedding model, delete chroma_db_data/
    and let it rebuild — old vectors are incompatible with new embeddings.
    """
    embedding_model = load_embedding_model()

    # 1. Load existing DB if present
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model,
        )
        print("Loaded existing Vector Store.")

    else:
        # 2. Build DB from PDF
        if not os.path.exists(PDF_PATH):
            st.error(f"PDF file not found: {PDF_PATH}")
            return None

        with st.spinner("📖 Indexing medical knowledge base — this runs once and may take a few minutes…"):
            loader = PyMuPDFLoader(PDF_PATH)
            # Exclude front matter and index pages (same slice as original)
            doc    = loader.load()[30:-439]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            splits = text_splitter.split_documents(doc)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                persist_directory=DB_PATH,
            )
        print("Created new Vector Store with PubMedBERT embeddings.")

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},   # return top 4 most relevant chunks
    )