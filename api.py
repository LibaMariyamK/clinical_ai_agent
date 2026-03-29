# api.py
# FastAPI server — exposes the ClinixAI agent as REST endpoints.
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from agent_core import build_agent, analyze_image

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClinixAI API",
    description="Agentic Clinical Decision Support — REST API",
    version="1.0.0",
)

# Allow Streamlit (localhost:8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Build agent once at startup ───────────────────────────────────────────────
print("Building agent on startup…")
agent = build_agent()
print("Agent ready.")


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:     str
    chat_history: Optional[List[str]] = []   # ["User: ...", "Assistant: ..."]

class QueryResponse(BaseModel):
    answer: str
    source: str          # "PDF" or "WEB"
    pages:  List[int]    # page numbers (empty for web)

class ImageQueryRequest(BaseModel):
    question:      str
    image_data_url: str              # base64 data URL
    chat_history:  Optional[List[str]] = []

class HealthResponse(BaseModel):
    status:  str
    version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def root():
    """Health check endpoint."""
    return {"status": "online", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return {"status": "online", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main query endpoint.
    Accepts a clinical question and optional chat history.
    Returns structured answer, source, and page numbers.
    """
    try:
        result = agent.invoke({
            "question":     request.question,
            "chat_history": request.chat_history or [],
            "pages":        [],
        })
        return QueryResponse(
            answer=result["answer"],
            source=result.get("source", "unknown").upper(),
            pages=result.get("pages", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-with-image", response_model=QueryResponse)
def query_with_image(request: ImageQueryRequest):
    """
    Vision + text query endpoint.
    Accepts a question + base64 image, runs vision analysis,
    then passes enriched query to the agent.
    """
    try:
        # Step 1 — analyse image
        image_description = analyze_image(request.image_data_url)

        # Step 2 — enrich question with vision output
        full_query = (
            f"User Question: {request.question}\n\n"
            f"Clinical Image Analysis: {image_description}"
        )

        # Step 3 — run agent
        result = agent.invoke({
            "question":     full_query,
            "chat_history": request.chat_history or [],
            "pages":        [],
        })
        return QueryResponse(
            answer=result["answer"],
            source=result.get("source", "unknown").upper(),
            pages=result.get("pages", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))