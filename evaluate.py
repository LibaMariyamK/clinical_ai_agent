# evaluate.py
# Custom RAG evaluation for ClinixAI — no RAGAS needed.
# Uses only packages already installed in the project.
#
# Measures:
#   - Answer Faithfulness  : does the answer stay grounded in retrieved context?
#   - Answer Relevancy     : does the answer actually address the question?
#   - Context Precision    : is the retrieved context relevant to the question?
#
# Run with: python evaluate.py

import os
import json
import time
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import get_retriever

# ── Config ────────────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    "What are the symptoms of appendicitis?",
    "How is Type 2 diabetes diagnosed?",
    "What is the treatment for pneumonia?",
    "What are the causes of hypertension?",
    "What are the early signs of Alzheimer's disease?",
]

API_DELAY = 4  # seconds between calls to avoid rate limiting

# ── Load models ───────────────────────────────────────────────────────────────
print("\nLoading models…")
llm       = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512)
retriever = get_retriever()
print("Models loaded.\n")


# ── Step 1 — Retrieve context and generate answer ─────────────────────────────
def retrieve_and_answer(question: str) -> dict:
    documents = retriever.invoke(question)
    contexts  = [doc.page_content for doc in documents]
    context   = "\n\n".join(contexts)

    prompt = ChatPromptTemplate.from_template(
        """You are a clinical medical assistant.
Answer strictly based on the context provided.

Context: {context}
Question: {question}
Answer:"""
    )
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    return {"question": question, "answer": answer, "context": context, "contexts": contexts}


# ── Step 2 — Score each metric using LLM as judge ────────────────────────────

def score_faithfulness(question: str, answer: str, context: str) -> float:
    """
    Faithfulness: does the answer contain only information from the context?
    LLM returns a score 1-5. We normalise to 0-1.
    """
    prompt = ChatPromptTemplate.from_template(
        """You are an expert evaluator. Score the faithfulness of this answer.

FAITHFULNESS measures whether the answer contains ONLY information present in the context.
A faithful answer does not introduce facts not found in the context.

Context:
{context}

Question: {question}
Answer: {answer}

Score the faithfulness from 1 to 5:
1 = answer contains many facts not in context (very unfaithful)
2 = answer has several unsupported claims
3 = answer is mostly faithful with minor unsupported claims
4 = answer is largely faithful with very minor issues
5 = answer is completely faithful to the context

Reply with ONLY a single integer (1, 2, 3, 4, or 5). Nothing else."""
    )
    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer, "context": context}).strip()
    try:
        score = int(result[0])
        return round((score - 1) / 4, 3)   # normalise 1-5 → 0.0-1.0
    except Exception:
        return 0.5


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: does the answer actually address the question asked?
    LLM returns a score 1-5. We normalise to 0-1.
    """
    prompt = ChatPromptTemplate.from_template(
        """You are an expert evaluator. Score the relevancy of this answer.

ANSWER RELEVANCY measures whether the answer directly addresses the question asked.
A relevant answer is on-topic and answers what was actually asked.

Question: {question}
Answer: {answer}

Score the relevancy from 1 to 5:
1 = answer is completely off-topic
2 = answer partially addresses the question
3 = answer mostly addresses the question
4 = answer addresses the question well
5 = answer directly and completely addresses the question

Reply with ONLY a single integer (1, 2, 3, 4, or 5). Nothing else."""
    )
    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer}).strip()
    try:
        score = int(result[0])
        return round((score - 1) / 4, 3)
    except Exception:
        return 0.5


def score_context_precision(question: str, context: str) -> float:
    """
    Context Precision: is the retrieved context relevant to the question?
    LLM returns a score 1-5. We normalise to 0-1.
    """
    prompt = ChatPromptTemplate.from_template(
        """You are an expert evaluator. Score the precision of this retrieved context.

CONTEXT PRECISION measures whether the retrieved context is relevant and useful
for answering the question asked.

Question: {question}
Retrieved Context:
{context}

Score the context precision from 1 to 5:
1 = context is completely irrelevant to the question
2 = context is mostly irrelevant with minor relevant parts
3 = context is partially relevant
4 = context is mostly relevant and useful
5 = context is highly relevant and directly useful for answering

Reply with ONLY a single integer (1, 2, 3, 4, or 5). Nothing else."""
    )
    chain  = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context}).strip()
    try:
        score = int(result[0])
        return round((score - 1) / 4, 3)
    except Exception:
        return 0.5


# ── Run evaluation ────────────────────────────────────────────────────────────
print(f"Evaluating {len(TEST_QUESTIONS)} questions…\n")
print("Step 1 — Retrieving context and generating answers:")

qa_results = []
for i, question in enumerate(TEST_QUESTIONS, 1):
    print(f"  [{i}/{len(TEST_QUESTIONS)}] {question[:55]}…")
    try:
        result = retrieve_and_answer(question)
        qa_results.append(result)
        print(f"           ✓ Answer generated")
        time.sleep(API_DELAY)
    except Exception as e:
        print(f"           ✗ Error: {e}")

print(f"\n{len(qa_results)}/{len(TEST_QUESTIONS)} answers generated.\n")

if not qa_results:
    print("No results. Check GROQ_API_KEY.")
    exit(1)

print("Step 2 — Scoring each metric (LLM-as-judge):\n")

per_question = []
all_faithfulness    = []
all_relevancy       = []
all_precision       = []

for i, r in enumerate(qa_results, 1):
    print(f"  [{i}/{len(qa_results)}] Scoring: {r['question'][:50]}…")

    try:
        f_score = score_faithfulness(r["question"], r["answer"], r["context"])
        time.sleep(2)
        r_score = score_answer_relevancy(r["question"], r["answer"])
        time.sleep(2)
        p_score = score_context_precision(r["question"], r["context"])
        time.sleep(2)

        all_faithfulness.append(f_score)
        all_relevancy.append(r_score)
        all_precision.append(p_score)

        per_question.append({
            "question":          r["question"],
            "answer":            r["answer"],
            "faithfulness":      f_score,
            "answer_relevancy":  r_score,
            "context_precision": p_score,
        })
        print(f"           ✓  F:{f_score:.2f}  R:{r_score:.2f}  P:{p_score:.2f}")

    except Exception as e:
        print(f"           ✗ Scoring error: {e}")
        time.sleep(5)

# ── Compute averages ──────────────────────────────────────────────────────────
def safe_avg(lst):
    return round(sum(lst) / len(lst), 3) if lst else None

avg_faithfulness = safe_avg(all_faithfulness)
avg_relevancy    = safe_avg(all_relevancy)
avg_precision    = safe_avg(all_precision)
all_scores       = [s for s in [avg_faithfulness, avg_relevancy, avg_precision] if s is not None]
overall_avg      = safe_avg(all_scores)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  CLINIXAI — EVALUATION RESULTS")
print("=" * 55)

def print_metric(label, value):
    if value is None:
        print(f"\n  {label}")
        print(f"  [{'?' * 20}]  N/A")
        return
    bar   = "█" * int(value * 20) + "░" * (20 - int(value * 20))
    grade = "✅ Good" if value >= 0.7 else ("⚠ Fair" if value >= 0.5 else "❌ Needs work")
    print(f"\n  {label}")
    print(f"  [{bar}] {value:.3f}  {grade}")

print_metric("Answer Faithfulness   (hallucination check)", avg_faithfulness)
print_metric("Answer Relevancy      (answers the question)", avg_relevancy)
print_metric("Context Precision     (retrieval quality)",    avg_precision)

print(f"\n{'=' * 55}")
if overall_avg:
    print(f"  Overall Average Score : {overall_avg}")
print(f"  Questions Evaluated   : {len(per_question)}")
print(f"  Evaluation Method     : LLM-as-judge (Llama 3.3 70B)")
print(f"{'=' * 55}\n")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "summary": {
        "faithfulness":      avg_faithfulness,
        "answer_relevancy":  avg_relevancy,
        "context_precision": avg_precision,
        "overall_average":   overall_avg,
        "questions_tested":  len(per_question),
        "evaluation_method": "LLM-as-judge (Llama 3.3 70B)",
        "embeddings":        "NeuML/pubmedbert-base-embeddings",
    },
    "per_question": per_question,
}

with open("evaluation_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("Results saved to evaluation_results.json")
print("Paste your scores here and resume content will be written.\n")