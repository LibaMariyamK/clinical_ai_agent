# evaluate.py
# RAGAS evaluation for ClinixAI RAG pipeline.
#
# Run with:   python evaluate.py
# Output:     evaluation_results.json + printed scores
#
# Install:    pip install ragas datasets

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")   # suppress deprecation noise

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ── Updated imports — ragas v0.2+ ────────────────────────────────────────────
try:
    # New location (ragas >= 0.2)
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
    )
except ImportError:
    # Fallback for older ragas
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
    )

from ragas import evaluate
from datasets import Dataset
from retriever import get_retriever

# ── Config ────────────────────────────────────────────────────────────────────
# Reduced to 5 questions to stay within Groq free tier token limit
# Each question uses ~2000-3000 tokens for evaluation
TEST_QUESTIONS = [
    "What are the symptoms of appendicitis?",
    "How is Type 2 diabetes diagnosed?",
    "What is the treatment for pneumonia?",
    "What are the causes of hypertension?",
    "What are the early signs of Alzheimer's disease?",
]

# Delay between API calls to avoid rate limiting (seconds)
API_DELAY = 3

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models…")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_retries=2,
    max_tokens=1024,    # limit tokens per call to preserve daily quota
)
retriever  = get_retriever()
embeddings = HuggingFaceEmbeddings(
    model_name="NeuML/pubmedbert-base-embeddings",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Models loaded.\n")


# ── Generate answers and retrieve contexts ────────────────────────────────────

def get_answer_and_context(question: str) -> dict:
    """Run retrieval + generation for one question."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Retrieve
    documents = retriever.invoke(question)
    contexts  = [doc.page_content for doc in documents]
    context   = "\n\n".join(contexts)

    # Generate
    prompt = ChatPromptTemplate.from_template(
        """You are a clinical medical assistant.
Answer strictly based on the context provided.

Context: {context}
Question: {question}

Answer:"""
    )
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    return {
        "question":     question,
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": "",
    }


print("Running retrieval and generation for all test questions…")
print(f"Testing {len(TEST_QUESTIONS)} questions with {API_DELAY}s delay between calls.\n")

results = []
for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")
    try:
        result = get_answer_and_context(question)
        results.append(result)
        print(f"         ✓ Done")
        if i < len(TEST_QUESTIONS):
            time.sleep(API_DELAY)  # avoid rate limit
    except Exception as e:
        print(f"         ✗ Error: {e}")

print(f"\n{len(results)}/{len(TEST_QUESTIONS)} questions completed successfully.\n")

if len(results) == 0:
    print("No results to evaluate. Check your GROQ_API_KEY and try again.")
    exit(1)

# ── Build RAGAS dataset ───────────────────────────────────────────────────────
dataset = Dataset.from_dict({
    "question":     [r["question"]     for r in results],
    "answer":       [r["answer"]       for r in results],
    "contexts":     [r["contexts"]     for r in results],
    "ground_truth": [r["ground_truth"] for r in results],
})

# ── Run RAGAS evaluation ──────────────────────────────────────────────────────
print("Running RAGAS evaluation…")
print("Note: RAGAS makes additional LLM calls — this may take 3-5 minutes.\n")

try:
    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=llm,
        embeddings=embeddings,
    )
except Exception as e:
    print(f"RAGAS evaluation error: {e}")
    print("This may be due to rate limits. Wait 15 minutes and try again.")
    exit(1)

# ── Safe score extraction — handles NaN and mixed types ──────────────────────
df = score.to_pandas()

# Only compute mean on known numeric metric columns
metric_cols = ["faithfulness", "answer_relevancy", "context_precision"]
available   = [c for c in metric_cols if c in df.columns]

score_dict = {}
for col in available:
    try:
        # coerce to numeric, drop NaN, then mean
        numeric_vals = df[col].apply(
            lambda x: float(x) if x is not None and str(x) not in ["nan", ""] else None
        ).dropna()
        score_dict[col] = float(numeric_vals.mean()) if len(numeric_vals) > 0 else 0.0
    except Exception:
        score_dict[col] = 0.0

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  CLINIXAI — RAGAS EVALUATION RESULTS")
print("=" * 55)

metric_labels = {
    "faithfulness":      "Answer Faithfulness   (hallucination check)",
    "answer_relevancy":  "Answer Relevancy      (answers the question)",
    "context_precision": "Context Precision     (retrieval quality)",
}

overall = []
for metric, label in metric_labels.items():
    value = score_dict.get(metric, None)
    if value is None:
        print(f"\n  {label}")
        print(f"  [{'?' * 20}] N/A  (not computed — rate limit hit)")
        continue
    overall.append(value)
    bar   = "█" * int(value * 20) + "░" * (20 - int(value * 20))
    grade = "✅ Good" if value >= 0.7 else ("⚠ Fair" if value >= 0.5 else "❌ Needs work")
    print(f"\n  {label}")
    print(f"  [{bar}] {value:.3f}  {grade}")

if overall:
    avg = sum(overall) / len(overall)
    print(f"\n{'='*55}")
    print(f"  Overall Average Score: {avg:.3f}")
    print(f"{'='*55}\n")
else:
    avg = 0.0
    print("\n  Could not compute average — too many rate limit errors.")
    print("  Wait 15 minutes and re-run.\n")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "summary": {
        "faithfulness":      score_dict.get("faithfulness",      None),
        "answer_relevancy":  score_dict.get("answer_relevancy",  None),
        "context_precision": score_dict.get("context_precision", None),
        "overall_average":   round(avg, 3) if overall else None,
        "questions_tested":  len(results),
    },
    "per_question": df[["question", "answer"] + available].to_dict(orient="records"),
}

with open("evaluation_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Detailed results saved to: evaluation_results.json")
print("Add these scores to your README to show evaluation rigor.")