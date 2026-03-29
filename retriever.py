# retriever.py

import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH  = "./chroma_db_data"
PDF_PATH = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition.pdf"

# ── Google Drive direct download URL ─────────────────────────────────────────
# Steps to get this URL:
# 1. Upload your PDF to Google Drive
# 2. Right-click → Share → Anyone with the link → Copy link
# 3. Your link looks like:
#    https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# 4. Replace FILE_ID below with your actual file ID
# 5. The download URL format is:
#    https://drive.google.com/uc?export=download&id=FILE_ID

GDRIVE_FILE_ID  = "1prjbub7OanD6ErwOPLaoHBfs8K99Y6u3"   # google drive - gale encyclopedia... pdf
GDRIVE_PDF_URL  = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"


def download_pdf_if_missing():
    """
    Downloads the PDF from Google Drive if it doesn't exist locally.
    This runs on Streamlit Cloud where the PDF is not in the repo.
    """
    if os.path.exists(PDF_PATH):
        return True

    st.info("📥 Downloading medical knowledge base — this runs once…")
    try:
        import requests
        with requests.get(GDRIVE_PDF_URL, stream=True) as r:
            r.raise_for_status()
            with open(PDF_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("✅ Knowledge base downloaded successfully.")
        return True
    except Exception as e:
        st.error(f"❌ Failed to download PDF: {e}")
        st.info("Please set GDRIVE_FILE_ID in retriever.py to your Google Drive file ID.")
        return False


@st.cache_resource
def load_embedding_model():
    """
    Loads PubMedBERT embedding model.
    Trained on PubMed biomedical literature — better than
    general-purpose MiniLM for medical terminology retrieval.
    """
    return HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_retriever():
    """
    Initialises or loads the Vector Database (ChromaDB).
    Downloads PDF from Google Drive if not found locally.
    Returns a retriever object used to search the PDF.
    """
    embedding_model = load_embedding_model()

    # 1. Load existing DB if present
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model,
        )
        print("Loaded existing Vector Store.")
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

    # 2. Download PDF if missing (cloud deployment)
    if not download_pdf_if_missing():
        return None

    # 3. Build DB from PDF
    with st.spinner("📖 Indexing medical knowledge base — this runs once and may take a few minutes…"):
        loader = PyMuPDFLoader(PDF_PATH)
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
        search_kwargs={"k": 4},
    )