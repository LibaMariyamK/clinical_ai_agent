#agent.py

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, StateGraph, START
from typing import TypedDict
from langchain_core.messages import HumanMessage

# Import the retriever to connect it to the graph
from retriever import get_retriever

class AgentState(TypedDict):
    """The shared state of our medical agent workflow."""
    question: str
    context: str
    answer: str
    source: str  # "pdf" or "web"

@st.cache_resource
def load_llm():
    """Main reasoning model (Llama 3.3 70B) for the Agent."""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=2)

@st.cache_resource
def load_vision_model():
    """Vision model (Llama 4 Maverick 17B) for Image Analysis."""
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

@st.cache_resource
def build_medical_agent():
    """Constructs the Self-Correcting RAG Graph (The Agent)."""
    
    llm = load_llm()
    retriever = get_retriever()
    web_search_tool = DuckDuckGoSearchRun()

    # NODE FUNCTIONS

    def retrieve_node(state):
        # Step 1: Try to retrieve from PDF
        print("--- RETRIEVE (PDF) ---")
        question = state["question"]
        if retriever:
            documents = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in documents])
        else:
            context = ""
        return {"context": context, "source": "pdf"}

    def web_search_node(state):
        # Step 1b: Fallback to Web Search
        print("--- WEB SEARCH ---")
        question = state["question"]
        results = web_search_tool.invoke(f"medical info: {question}")
        return {"context": results, "source": "web"}

    def grade_documents_node(state):
        # Step 2: Grader (Self-Reflection)
        print("--- GRADER ---")
        question = state["question"]
        context = state["context"]
        
        # Fallback to web if context is too short or empty
        if not context or len(context) < 20:
            return "web_search"

        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grader assessing relevance. Return 'yes' if relevant, 'no' if not."),
            ("human", "Doc: {context}\n\nQuestion: {question}")
        ])
        grader = grader_prompt | llm | StrOutputParser()
        score = grader.invoke({"question": question, "context": context})
        return "generate" if "yes" in score.lower() else "web_search"

    def generate_node(state):
        # Step 3: Generate Final Answer
        print("--- GENERATE ---")
        question = state["question"]
        context = state["context"]
        
        prompt = ChatPromptTemplate.from_template(
            """You are a medical assistant. Answer strictly based on the context provided.
            Context: {context}
            Question: {question}"""
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})
        return {"answer": answer}

    # GRAPH DEFINITION
    
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    # Define Edges (Flow)
    workflow.add_edge(START, "retrieve")
    
    # Conditional logic
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents_node,
        {"generate": "generate", "web_search": "web_search"}
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()