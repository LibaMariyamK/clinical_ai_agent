# agent_core.py
# Pure agent logic — no Streamlit dependency.
# Used by both api.py (FastAPI) and agent.py (Streamlit wrapper).

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, StateGraph, START
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from functools import lru_cache

from retriever import get_retriever


class AgentState(TypedDict):
    """The shared state of our medical agent workflow."""
    question:           str
    rewritten_question: str
    chat_history:       List[str]
    context:            str
    answer:             str
    source:             str
    pages:              List[int]


@lru_cache(maxsize=1)
def load_llm():
    """Main reasoning model — cached with lru_cache (no Streamlit needed)."""
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_retries=2)


@lru_cache(maxsize=1)
def load_vision_model():
    """Vision model — cached with lru_cache (no Streamlit needed)."""
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        max_retries=2,
    )


# ── Type-specific analysis prompts ────────────────────────────────────────────

IMAGE_PROMPTS = {
    "chest_xray": """You are an experienced radiologist reviewing a chest X-Ray.
Analyse the image systematically and report findings for each area:

1. Lung fields — any consolidation, opacity, effusion, pneumothorax, 
   hyperinflation. Specify location (right/left, upper/middle/lower lobe).
2. Cardiac silhouette — size (normal <50% of thoracic width), borders, shape.
3. Mediastinum — width, tracheal deviation, hilar prominence.
4. Diaphragm — position, clarity, costophrenic angles.
5. Bones and soft tissue — rib fractures, bone density.

STRICT RULES:
- If a finding is NOT clearly visible, write 'No abnormality detected' for that section.
- Do NOT infer or assume findings that are not clearly visible.
- Do NOT report a finding unless you are confident it is present.
- Be specific about locations.
- Do not provide a definitive diagnosis.""",

    "brain_mri": """You are an experienced neuroradiologist reviewing a brain MRI.
Analyse the image systematically:

1. Brain parenchyma — any abnormal signal intensity, lesions, masses, oedema.
2. Ventricles — size, symmetry, hydrocephalus.
3. Cortical sulci and gyri — atrophy, effacement.
4. White matter — any hyperintensities, demyelination.
5. Midline structures — shift, displacement.
6. Posterior fossa — cerebellum, brainstem findings.

Specify lesion location, size estimate, and signal characteristics. Do not provide a definitive diagnosis.""",

    "ct_scan": """You are an experienced radiologist reviewing a CT scan.
Analyse the image systematically:

1. Identify the body region and scan plane (axial/coronal/sagittal).
2. Parenchymal findings — any masses, lesions, densities, calcifications.
3. Vascular structures — any abnormalities visible.
4. Surrounding structures — lymph nodes, fat stranding, fluid collections.
5. Bones — any fractures, erosions, sclerosis.

Report Hounsfield density characteristics where relevant. Do not provide a definitive diagnosis.""",

    "mri_general": """You are an experienced radiologist reviewing an MRI scan.
Analyse the image systematically:

1. Identify body region and sequence type if determinable (T1/T2/FLAIR/DWI).
2. Signal abnormalities — any hyperintense or hypointense lesions.
3. Mass effect — any displacement of surrounding structures.
4. Enhancement pattern if contrast visible.
5. Surrounding tissue — oedema, infiltration.

Specify location, size estimate, and signal characteristics. Do not provide a definitive diagnosis.""",

    "xray_bone": """You are an experienced musculoskeletal radiologist reviewing a bone X-Ray.
Analyse the image systematically:

1. Bone alignment — any fracture, dislocation, subluxation.
2. Bone density — osteoporosis, sclerosis, lytic lesions.
3. Joint spaces — narrowing, widening, erosions (suggest arthritis).
4. Soft tissue — swelling, calcification, foreign bodies.
5. Growth plates if visible — any abnormality.

Specify location precisely. Do not provide a definitive diagnosis.""",

    "skin": """You are an experienced dermatologist reviewing a clinical skin image.
Analyse the lesion systematically using ABCDE criteria:

1. Asymmetry — is the lesion asymmetric?
2. Border — regular or irregular, well-defined or diffuse.
3. Colour — uniform or multiple colours, distribution.
4. Diameter — estimate size relative to surrounding structures.
5. Evolution clues — any secondary changes (ulceration, scaling, crusting).
6. Surrounding skin — erythema, satellite lesions.

Do not provide a definitive diagnosis.""",

    "ultrasound": """You are an experienced sonographer reviewing an ultrasound image.
Analyse the image systematically:

1. Identify organ or region being imaged.
2. Echogenicity — hyperechoic, hypoechoic, anechoic findings.
3. Any masses or lesions — size, shape, margins, internal characteristics.
4. Vascularity if Doppler visible.
5. Surrounding structures — free fluid, lymph nodes.

Do not provide a definitive diagnosis.""",

    "general": """You are an experienced medical imaging specialist.
Analyse this medical image strictly scientifically:

1. Identify the imaging modality and body region.
2. Describe all visible findings systematically.
3. Note any obvious abnormalities, asymmetries, or areas of concern.
4. Describe location, size, and characteristics of any findings.

Do not provide a definitive diagnosis.""",
}

IMAGE_TYPE_KEYWORDS = {
    "chest_xray":  ["chest x-ray", "chest xray", "chest radiograph", "cxr", "chest film"],
    "brain_mri":   ["brain mri", "head mri", "cerebral mri", "cranial mri", "brain magnetic"],
    "ct_scan":     ["ct scan", "computed tomography", "cat scan", "ct of"],
    "mri_general": ["mri", "magnetic resonance"],
    "xray_bone":   ["bone x-ray", "bone xray", "skeletal", "knee x", "hip x", "spine x",
                    "shoulder x", "wrist x", "ankle x", "foot x", "hand x", "elbow x"],
    "skin":        ["skin", "dermatology", "lesion", "rash", "wound"],
    "ultrasound":  ["ultrasound", "sonography", "sonogram", "echo"],
}


def detect_image_type(image_data_url: str, vision_llm) -> str:
    """Stage 1 — detect what type of medical image this is."""
    detect_prompt = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "What type of medical image is this? "
                    "Reply in ONE short phrase only, for example: "
                    "'chest x-ray', 'brain MRI', 'CT scan abdomen', "
                    "'knee x-ray', 'skin lesion', 'ultrasound abdomen'. "
                    "Do not explain. Just the image type."
                ),
            },
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    )
    response = vision_llm.invoke([detect_prompt]).content.lower().strip()
    print(f"--- IMAGE TYPE DETECTED: {response} ---")

    for image_type, keywords in IMAGE_TYPE_KEYWORDS.items():
        if any(kw in response for kw in keywords):
            print(f"--- MATCHED TYPE: {image_type} ---")
            return image_type

    print("--- MATCHED TYPE: general (fallback) ---")
    return "general"


def analyze_image(image_data_url: str) -> str:
    """
    Two-stage adaptive vision analysis:
    Stage 1 — detect image type
    Stage 2 — apply type-specific radiologist prompt
    """
    vision_llm = load_vision_model()

    # Stage 1 — detect type
    image_type = detect_image_type(image_data_url, vision_llm)

    # Stage 2 — analyse with appropriate prompt
    analysis_prompt = HumanMessage(
        content=[
            {"type": "text",      "text": IMAGE_PROMPTS[image_type]},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    )
    analysis = vision_llm.invoke([analysis_prompt]).content
    print(f"--- IMAGE ANALYSIS COMPLETE ({image_type}) ---")

    return f"[Image type: {image_type.replace('_', ' ').title()}]\n\n{analysis}"


def build_agent():
    """Constructs and compiles the LangGraph agent."""

    llm             = load_llm()
    retriever       = get_retriever()
    web_search_tool = DuckDuckGoSearchRun()

    # ── NODES ────────────────────────────────────────────────────────────────

    def rewrite_question_node(state: AgentState):
        print("--- REWRITE QUESTION ---")
        question     = state["question"]
        chat_history = state.get("chat_history", []) or []
        history_str  = "\n".join(chat_history[-6:]) if chat_history else ""

        if not history_str:
            return {"rewritten_question": question}

        rewrite_prompt = ChatPromptTemplate.from_template(
            """You are a medical query resolver. \
Rewrite the current question to be fully explicit and self-contained \
using the conversation history. Replace vague references like "it", \
"this", "the condition" with the actual medical term from the history. \
If already explicit, return unchanged. Return ONLY the rewritten question.

Conversation history:
{chat_history}

Current question: {question}

Rewritten question:"""
        )
        rewriter  = rewrite_prompt | llm | StrOutputParser()
        rewritten = rewriter.invoke({
            "chat_history": history_str,
            "question":     question,
        }).strip()

        print(f"--- ORIGINAL:  {question}")
        print(f"--- REWRITTEN: {rewritten}")
        return {"rewritten_question": rewritten}

    def retrieve_node(state: AgentState):
        print("--- RETRIEVE (PDF) ---")
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
            ("human", "Doc: {context}\n\nQuestion: {question}"),
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
Skip any section not applicable to the question. \
Use markdown exactly as shown:

**📋 Overview**
(2-3 sentence summary)

**🔍 Key Symptoms**
- (bullet point list)

**🩺 Diagnosis Approach**
(brief description)

**💊 Treatment Options**
(brief description)

**⚠ Warning Signs**
(red flags — only if relevant)

Answer:"""
        )
        chain  = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question":     question,
            "context":      context,
            "chat_history": history_str,
        })
        return {"answer": answer}

    # ── GRAPH ────────────────────────────────────────────────────────────────

    workflow = StateGraph(AgentState)

    workflow.add_node("rewrite",    rewrite_question_node)
    workflow.add_node("retrieve",   retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate",   generate_node)

    workflow.add_edge(START,     "rewrite")
    workflow.add_edge("rewrite", "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents_node,
        {"generate": "generate", "web_search": "web_search"},
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate",   END)

    return workflow.compile()