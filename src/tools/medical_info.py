"""Medical info search tool (RAG over the seeded Medline/WHO corpus)."""
from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..config import LLM_MODEL, MEDICAL_TOP_K
from ..embeddings import load_medical_index
from ..prompts import MEDICAL_INFO_SYNTHESIS_PROMPT


def _dedup_sources(docs) -> list[dict]:
    seen: set = set()
    sources: list[dict] = []
    for d in docs:
        key = d.metadata.get("file")
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": d.metadata.get("title"),
            "source": d.metadata.get("source"),
            "source_url": d.metadata.get("source_url"),
            "fetched_at": d.metadata.get("fetched_at"),
        })
    return sources


def _medical_info_search_impl(condition: str, subtopic: Optional[str] = None,
                              k: int = MEDICAL_TOP_K) -> dict:
    query = condition if not subtopic else f"{condition} — {subtopic}"
    store = load_medical_index()
    docs = store.similarity_search(query, k=k)
    if not docs:
        return {"query": query, "answer": "No relevant information found.", "sources": []}

    passages_numbered = "\n\n".join(
        f"[{i+1}] ({d.metadata.get('title')}, fetched {d.metadata.get('fetched_at')})\n"
        f"{d.page_content}"
        for i, d in enumerate(docs)
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = ChatPromptTemplate.from_template(MEDICAL_INFO_SYNTHESIS_PROMPT) | llm
    result = chain.invoke({
        "condition": condition,
        "subtopic": subtopic or "general overview",
        "passages": passages_numbered,
    })

    return {
        "query": query,
        "answer": result.content.strip(),
        "sources": _dedup_sources(docs),
    }


@tool
def medical_info_search(condition: str, subtopic: Optional[str] = None) -> dict:
    """Search the trusted medical corpus (Medline / WHO) for information about a
    condition, with optional `subtopic` (e.g. 'latest treatment', 'symptoms',
    'prevention'). Uses RAG: retrieves top passages from FAISS, then synthesizes
    a cited answer with the LLM.

    Returns `{query, answer, sources}`. The `answer` contains inline citation
    markers like `[1]` that map to entries in `sources`.
    """
    return _medical_info_search_impl(condition, subtopic)
