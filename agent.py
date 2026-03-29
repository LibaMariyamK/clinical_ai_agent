# agent.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, StateGraph, START
from typing import TypedDict, List
from langchain_core.messages import HumanMessage

from retriever import get_retriever


class AgentState(TypedDict):
    """The shared state of our medical agent workflow."""
    question:          str
    rewritten_question: str   # ← NEW: resolved/explicit version of the question
    chat_history:      List[str]
    context:           str
    answer:            str
    source:            str
    pages:             List[int]


@st.cache_resource
def load_llm():
    """Main reasoning model (Llama 3.3 70B) for the Agent."""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=2)


@st.cache_resource
def load_vision_model():
    """Vision model (Llama 4 Scout 17B) for Image Analysis."""
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3, max_retries=2)


def analyze_image(image_data_url):
    """Helper function to perform Vision Analysis (Image-to-Text)."""
    vision_llm = load_vision_model()
    vision_prompt = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this medical image strictly scientifically. Describe findings, symptoms, and anomalies without providing a definitive diagnosis."},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
    )
    return vision_llm.invoke([vision_prompt]).content


def build_medical_agent():
    """Constructs the Self-Correcting RAG Graph (The Agent)."""

    llm             = load_llm()
    retriever       = get_retriever()
    web_search_tool = DuckDuckGoSearchRun()

    # ── NODE FUNCTIONS ────────────────────────────────────────────────────────

    def rewrite_question_node(state: AgentState):
        # ── NEW: Resolve vague references before retrieval ────────────────────
        # Turns "What is the treatment for it?" →
        #       "What is the treatment for pneumonia?"
        print("--- REWRITE QUESTION ---")
        question     = state["question"]
        chat_history = state.get("chat_history", []) or []
        history_str  = "\n".join(chat_history[-6:]) if chat_history else ""

        # If no history, no need to rewrite
        if not history_str:
            print(f"--- NO HISTORY — keeping original: {question}")
            return {"rewritten_question": question}

        rewrite_prompt = ChatPromptTemplate.from_template(
            """You are a medical query resolver. \
Given the conversation history and the current question, \
rewrite the current question to be fully explicit and self-contained. \
Replace any vague references like "it", "this", "the condition", "the disease", \
"the treatment", "the symptoms" with the actual medical term from the history. \
If the question is already explicit, return it unchanged. \
Return ONLY the rewritten question, nothing else.

Conversation history:
{chat_history}

Current question: {question}

Rewritten question:"""
        )

        rewriter = rewrite_prompt | llm | StrOutputParser()
        rewritten = rewriter.invoke({
            "chat_history": history_str,
            "question":     question,
        }).strip()

        print(f"--- ORIGINAL:  {question}")
        print(f"--- REWRITTEN: {rewritten}")
        return {"rewritten_question": rewritten}

    def retrieve_node(state: AgentState):
        print("--- RETRIEVE (PDF) ---")
        # Use rewritten question for retrieval — explicit query = better chunks
        question = state.get("rewritten_question") or state["question"]
        pages    = []

        if retriever:
            documents = retriever.invoke(question)
            context   = "\n\n".join([doc.page_content for doc in documents])
            for doc in documents:
                page = doc.metadata.get("page", None)
                if page is not None:
                    pages.append(int(page) + 1)
            pages = sorted(set(pages))
        else:
            context = ""

        print(f"--- PAGES FOUND: {pages} ---")
        return {"context": context, "source": "pdf", "pages": pages}

    def web_search_node(state: AgentState):
        print("--- WEB SEARCH ---")
        # Also use rewritten question for web search
        question = state.get("rewritten_question") or state["question"]
        results  = web_search_tool.invoke(f"medical info: {question}")
        return {"context": results, "source": "web", "pages": []}

    def grade_documents_node(state: AgentState):
        print("--- GRADER ---")
        question = state.get("rewritten_question") or state["question"]
        context  = state["context"]

        if not context or len(context) < 20:
            return "web_search"

        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance. Return 'yes' if relevant, 'no' if not."),
            ("human", "Doc: {context}\n\nQuestion: {question}")
        ])
        grader = grader_prompt | llm | StrOutputParser()
        score  = grader.invoke({"question": question, "context": context})
        return "generate" if "yes" in score.lower() else "web_search"

    def generate_node(state: AgentState):
        print("--- GENERATE ---")
        question     = state.get("rewritten_question") or state["question"]
        context      = state["context"]
        chat_history = state.get("chat_history", []) or []
        history_str  = "\n".join(chat_history[-6:]) if chat_history else "No previous conversation."

        prompt = ChatPromptTemplate.from_template(
            """You are a clinical medical assistant. \
Answer strictly based on the provided context.

--- CONVERSATION HISTORY ---
{chat_history}
--- END OF HISTORY ---

--- MEDICAL CONTEXT ---
{context}
--- END OF CONTEXT ---

Current question: {question}

Respond ONLY using the following structured format. \
Skip any section that is genuinely not applicable to the question. \
Do not add any section not in this list. \
Use markdown exactly as shown:

**📋 Overview**
(2-3 sentence summary of the condition or answer)

**🔍 Key Symptoms**
- (bullet point list)

**🩺 Diagnosis Approach**
(brief description of how it is diagnosed)

**💊 Treatment Options**
(brief description of treatments)

**⚠ Warning Signs**
(red flags that require immediate medical attention — only include if relevant)

Answer:"""
        )

        chain  = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question":     question,
            "context":      context,
            "chat_history": history_str,
        })
        return {"answer": answer}

    # ── GRAPH DEFINITION ─────────────────────────────────────────────────────

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("rewrite",    rewrite_question_node)  # ← NEW first step
    workflow.add_node("retrieve",   retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate",   generate_node)

    # Flow: START → rewrite → retrieve → grade → generate/web_search → END
    workflow.add_edge(START,      "rewrite")
    workflow.add_edge("rewrite",  "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents_node,
        {"generate": "generate", "web_search": "web_search"}
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate",   END)

    return workflow.compile()