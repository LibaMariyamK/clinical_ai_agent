import streamlit as st
import base64
from PIL import Image
from dotenv import load_dotenv

from agent import build_medical_agent, analyze_image

load_dotenv()

APP_NAME = "ClinixAI"
APP_ICON = "🩺"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Lora:wght@400;600&display=swap');

:root {
    --bg-primary:      #0a0f1e;
    --bg-secondary:    #0f172a;
    --bg-card:         #111827;
    --bg-hover:        #1e293b;
    --accent-teal:     #0ea5e9;
    --accent-cyan:     #06b6d4;
    --accent-green:    #10b981;
    --accent-amber:    #f59e0b;
    --accent-red:      #ef4444;
    --border:          rgba(14,165,233,0.18);
    --text-primary:    #f1f5f9;
    --text-secondary:  #94a3b8;
    --text-muted:      #475569;
    --font-main:       'Space Grotesk', sans-serif;
    --font-mono:       'JetBrains Mono', monospace;
    --font-serif:      'Lora', serif;
    --radius:          12px;
    --radius-sm:       8px;
    --shadow:          0 4px 24px rgba(0,0,0,0.4);
}

html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-primary) !important;
}
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stBottom"],
[data-testid="stDecoration"],
header[data-testid="stHeader"] {
    background: var(--bg-primary) !important;
    background-color: var(--bg-primary) !important;
}
.main .block-container { padding: 1.5rem 2rem 3rem !important; max-width: 900px; }
.ms-header {
    display: flex; align-items: center; gap: 16px; padding: 18px 24px;
    background: linear-gradient(135deg, #0f2044 0%, #0a1628 50%, #071324 100%);
    border: 1px solid var(--border); border-radius: var(--radius);
    margin-bottom: 1.5rem;
    box-shadow: 0 0 40px rgba(14,165,233,0.08), var(--shadow);
    position: relative; overflow: hidden;
}
.ms-header::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(14,165,233,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.ms-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
    border-radius: 14px; display: flex; align-items: center;
    justify-content: center; font-size: 26px; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(14,165,233,0.35);
}
.ms-title {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #e0f2fe 30%, #7dd3fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.02em; line-height: 1.1;
}
.ms-subtitle { font-size: 0.75rem; color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase; font-weight: 500; margin-top: 3px; }
.ms-status {
    margin-left: auto; display: flex; align-items: center; gap: 8px;
    font-size: 0.72rem; color: var(--accent-green); font-family: var(--font-mono);
    background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);
    border-radius: 20px; padding: 5px 12px;
}
.status-dot { width: 7px; height: 7px; background: var(--accent-green); border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.6; transform: scale(0.85); } }
[data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
.sb-logo { text-align: center; padding: 0.2rem 0 1.2rem; border-bottom: 1px solid var(--border); margin-bottom: 4px; }
.sb-logo-icon { font-size: 2.4rem; line-height: 1; }
.sb-logo-name { font-family: var(--font-serif) !important; font-weight: 600; font-size: 1.05rem; color: #e0f2fe; margin-top: 5px; }
.sb-logo-ver  { font-size: 0.62rem; font-weight: 500; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px; }
.sb-section   { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--text-muted); padding: 1rem 0 0.4rem; border-bottom: 1px solid var(--border); margin-bottom: 0.6rem; }
.sb-footer    { margin-top: 1.6rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.65rem; color: var(--text-muted); text-align: center; line-height: 1.8; }
.sb-footer span { color: var(--accent-teal); font-weight: 600; }
[data-testid="stFileUploader"] { background: var(--bg-card) !important; border: 1.5px dashed var(--border) !important; border-radius: var(--radius-sm) !important; transition: border-color 0.2s, background 0.2s !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent-teal) !important; background: var(--bg-hover) !important; }
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important; border: none !important; border-radius: var(--radius-sm) !important;
    font-family: var(--font-main) !important; font-weight: 600 !important; font-size: 0.8rem !important;
    letter-spacing: 0.03em; padding: 8px 16px !important; width: 100%; transition: all 0.2s ease !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #0284c7, #0ea5e9) !important; box-shadow: 0 0 16px rgba(14,165,233,0.35) !important; transform: translateY(-1px); }
[data-testid="stChatMessage"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; margin-bottom: 0.75rem !important; padding: 14px 18px !important; box-shadow: var(--shadow); transition: border-color 0.2s; }
[data-testid="stChatMessage"]:hover { border-color: rgba(14,165,233,0.3) !important; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * { color: #f1f5f9 !important; }
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] b { color: #e2e8f0 !important; font-weight: 700 !important; }
[data-testid="stChatMessage"] code { background: rgba(14,165,233,0.12) !important; color: #7dd3fc !important; padding: 1px 5px; border-radius: 4px; font-family: var(--font-mono) !important; font-size: 0.88em; }
[data-testid="stChatInput"] { background: var(--bg-card) !important; border: 1.5px solid var(--border) !important; border-radius: var(--radius) !important; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent-teal) !important; box-shadow: 0 0 0 3px rgba(14,165,233,0.12) !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: var(--text-primary) !important; font-family: var(--font-main) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--font-mono) !important; font-size: 0.7rem !important; color: var(--text-muted) !important; background: var(--bg-hover) !important; border-radius: 20px !important; padding: 2px 10px !important; display: inline-block !important; border: 1px solid var(--border) !important; margin-top: 4px !important; }
[data-testid="stSpinner"] p { color: var(--text-secondary) !important; font-family: var(--font-main) !important; font-size: 0.85rem !important; font-weight: 600 !important; }
.empty-state { text-align: center; padding: 3.5rem 1rem 2rem; }
.empty-state-icon  { font-size: 3rem; margin-bottom: 0.8rem; opacity: 0.6; }
.empty-state-title { font-size: 1.1rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.4rem; }
.empty-state-body  { font-size: 0.82rem; max-width: 380px; margin: 0 auto; line-height: 1.65; color: var(--text-muted); }
.hint-row  { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 1.2rem; }
.hint-chip { background: var(--bg-card); border: 1px solid var(--border); border-radius: 20px; padding: 5px 14px; font-size: 0.75rem; font-weight: 600; color: var(--text-secondary); }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-teal); }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ms-header">
    <div class="ms-logo">{APP_ICON}</div>
    <div>
        <div class="ms-title">{APP_NAME}</div>
        <div class="ms-subtitle">Agentic Clinical Decision Support System</div>
    </div>
    <div class="ms-status">
        <div class="status-dot"></div>
        SYSTEM ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("🔧 Initialising agent…"):
        st.session_state.agent = build_medical_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo">
        <div class="sb-logo-icon">🏥</div>
        <div class="sb-logo-name">{APP_NAME}</div>
        <div class="sb-logo-ver">Clinical Assistant · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">🩻 Medical Imaging</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload scan",
        type=["jpg", "png", "jpeg"],
        key=str(st.session_state["uploader_key"]),
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded scan", use_container_width=True)

    st.markdown('<div class="sb-section">⚙ Session</div>', unsafe_allow_html=True)
    if st.button("🗑  Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div class="sb-footer">
        Reasoning · <span>Llama 3.3 70B</span><br>
        Vision · <span>Llama 4 Scout 17B</span><br>
        Knowledge · <span>Gale Encyclopedia of Medicine</span>
    </div>
    """, unsafe_allow_html=True)

# ── HELPER — format source badge ──────────────────────────────────────────────
def format_source_badge(src: str, pages: list) -> str:
    """
    Returns a readable source string for st.caption().
    PDF: shows page numbers if available.
    Web: shows external source label.
    """
    if src == "WEB":
        return "🌐  Web Search — external source"
    else:
        if pages:
            page_str = ", ".join(str(p) for p in pages)
            return f"📄  Knowledge Base — Pages {page_str}"
        else:
            return "📄  Knowledge Base — Gale Encyclopedia of Medicine"

# ── CHAT HISTORY DISPLAY ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else APP_ICON
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], width=180)
        if "source" in message:
            st.caption(format_source_badge(
                message["source"],
                message.get("pages", [])
            ))

# ── EMPTY STATE ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🔬</div>
        <div class="empty-state-title">How can I assist you today?</div>
        <div class="empty-state-body">
            Ask about symptoms, conditions, drug interactions, or upload a medical scan
            for AI-assisted visual analysis.
        </div>
        <div class="hint-row">
            <span class="hint-chip">💊 Drug interactions</span>
            <span class="hint-chip">🫁 Radiology findings</span>
            <span class="hint-chip">🧬 Differential diagnosis</span>
            <span class="hint-chip">📋 Treatment protocols</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── HELPER — build chat history ───────────────────────────────────────────────
def build_chat_history(messages: list, max_turns: int = 3) -> list:
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            history.append(f"Assistant: {msg['content']}")
    return history[-(max_turns * 2):]

# ── MAIN CHAT LOOP ────────────────────────────────────────────────────────────
if prompt := st.chat_input("Enter clinical query…"):

    user_msg_data = {"role": "user", "content": prompt}

    if uploaded_file:
        image = Image.open(uploaded_file)
        user_msg_data["image"] = image

        uploaded_file.seek(0)
        b64_image      = base64.b64encode(uploaded_file.read()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{b64_image}"

        with st.spinner("🩻 Analysing imaging features…"):
            image_description = analyze_image(image_data_url)

        full_query = f"User Question: {prompt}\n\nClinical Image Analysis: {image_description}"
    else:
        full_query = prompt

    chat_history = build_chat_history(st.session_state.messages)

    st.session_state.messages.append(user_msg_data)

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        if uploaded_file:
            st.image(image, width=180)

    with st.chat_message("assistant", avatar=APP_ICON):
        with st.spinner("🧠 Thinking — Retrieving → Grading → Generating…"):
            try:
                result = st.session_state.agent.invoke({
                    "question":     full_query,
                    "chat_history": chat_history,
                    "pages":        [],   # initialise empty — retrieve_node fills it
                })
                answer = result["answer"]
                source = result.get("source", "unknown").upper()
                pages  = result.get("pages", [])   # ← NEW

                st.markdown(answer)
                st.caption(format_source_badge(source, pages))   # ← NEW

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "source":  source,
                    "pages":   pages,    # ← NEW: saved so history display works too
                })

            except Exception as e:
                st.error(f"⚠ Something went wrong: {e}")

    if uploaded_file:
        st.session_state["uploader_key"] += 1
        st.rerun()