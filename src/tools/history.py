"""Patient-history retrieval tool. Combines FAISS semantic search over the H&P
notes with an LLM summarization step (the 'RAG' step)."""
from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..config import HP_TOP_K, LLM_MODEL
from ..db import get_conn
from ..embeddings import load_hp_index
from ..prompts import HISTORY_SUMMARY_PROMPT


def _fetch_patient(patient_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT patient_id, name, age, gender, summary FROM patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
    return dict(row) if row else None


def _get_patient_history_impl(patient_id: int, question: str, k: int = HP_TOP_K) -> dict:
    patient = _fetch_patient(patient_id)
    if not patient:
        return {"error": f"No patient with id={patient_id}"}

    store = load_hp_index()

    # Semantic retrieval scoped to this patient via metadata filter.
    # FAISS doesn't filter natively; over-fetch then filter.
    candidates = store.similarity_search(question, k=k * 4)
    scoped = [d for d in candidates if d.metadata.get("patient_id") == patient_id][:k]

    # If the scoped filter returns nothing (e.g. synthetic patient with no PDF),
    # fall back to the SQL `summary` column.
    if not scoped:
        excerpts = [patient["summary"]] if patient.get("summary") else []
        source_chunks: list[dict] = []
    else:
        excerpts = [d.page_content for d in scoped]
        source_chunks = [
            {"source": d.metadata.get("source"), "chunk": d.metadata.get("chunk")}
            for d in scoped
        ]

    if not excerpts:
        return {
            "patient": patient,
            "summary": "No recorded history for this patient.",
            "sources": [],
        }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = ChatPromptTemplate.from_template(HISTORY_SUMMARY_PROMPT) | llm
    result = chain.invoke({
        "patient_name": patient["name"],
        "question": question,
        "excerpts": "\n\n---\n\n".join(excerpts),
    })
    return {
        "patient": patient,
        "summary": result.content.strip(),
        "sources": source_chunks,
    }


@tool
def get_patient_history(patient_id: int, question: str) -> dict:
    """Retrieve and summarize a patient's medical history relevant to `question`.

    Pulls semantically relevant chunks from the patient's H&P notes (FAISS) and
    synthesizes a focused summary via the LLM. Returns `{patient, summary, sources}`.
    """
    return _get_patient_history_impl(patient_id, question)
