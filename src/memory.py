"""Long-term memory: per-patient rolling summary + recent agent-run recap.

Kept lightweight on purpose. A heavier FAISS-per-patient store is reasonable future
work, but the `patients.summary` column already captures what we need for demo scale.
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import LLM_MODEL
from .db import get_conn


_SUMMARY_UPDATE_PROMPT = """You maintain a rolling clinical summary for a patient.

Current summary:
{current}

New information from this interaction:
{new_info}

Write an updated summary, under 120 words, integrating the new information.
Keep it factual, clinical, and in third person. Do not invent details."""


def build_context_for_planner(patient_id: Optional[int], max_runs: int = 3) -> str:
    """Return a short block of context to splice into the planner prompt."""
    if not patient_id:
        return "(no patient context yet)"

    with get_conn() as conn:
        p = conn.execute(
            "SELECT patient_id, name, age, gender, summary FROM patients WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()
        if not p:
            return "(patient not found)"

        # Include this patient's recent agent runs if we logged any.
        runs = conn.execute(
            "SELECT user_query, final_answer, created_at FROM agent_runs "
            "ORDER BY created_at DESC LIMIT ?",
            (max_runs,),
        ).fetchall()

    lines = [
        f"Patient #{p['patient_id']}: {p['name']}, "
        f"{p['age']}y {p['gender']}.",
        f"Rolling summary: {p['summary'] or '(none)'}",
    ]
    if runs:
        lines.append("Recent interactions:")
        for r in runs:
            ans = (r["final_answer"] or "").split("\n")[0][:100]
            lines.append(f"  - [{r['created_at']}] {r['user_query'][:60]!r} -> {ans!r}")
    return "\n".join(lines)


def update_patient_summary(patient_id: int, new_info: str) -> str:
    """LLM-summarise old + new info and write back to patients.summary."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT summary FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
    current = (row["summary"] if row else "") or "(empty)"

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = ChatPromptTemplate.from_template(_SUMMARY_UPDATE_PROMPT) | llm
    out = chain.invoke({"current": current, "new_info": new_info}).content.strip()

    with get_conn() as conn:
        conn.execute(
            "UPDATE patients SET summary = ?, updated_at = CURRENT_TIMESTAMP WHERE patient_id = ?",
            (out, patient_id),
        )
    return out


def record_interaction_note(patient_id: int, user_query: str, agent_final: str) -> None:
    """Shorthand: append this interaction into the patient's rolling summary."""
    note = f"User asked: {user_query}\nAssistant action: {agent_final[:300]}"
    update_patient_summary(patient_id, note)


def dump_state(run_id: int) -> dict:
    """Return a serialisable snapshot of an agent run (for the Trace UI tab)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    if not row:
        return {}
    return {
        **dict(row),
        "plan": json.loads(row["plan_json"] or "[]"),
        "trace": json.loads(row["trace_json"] or "[]"),
    }
