"""LangGraph planner-executor agent.

Graph:
    START -> plan -> execute -> compose -> END

State fields live in AgentState below. Each agent run is persisted to the
`agent_runs` table and returned as a structured dict for the UI.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import LLM_MODEL
from .db import log_agent_run
from .memory import build_context_for_planner
from .prompts import (
    FINAL_COMPOSER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
)
from .tools import (
    add_or_update_record,
    book_appointment,
    cancel_appointment,
    get_patient_history,
    list_doctor_slots,
    lookup_patient,
    medical_info_search,
    verify_caller,
)


TOOL_REGISTRY: dict[str, Any] = {
    "lookup_patient": lookup_patient,
    "verify_caller": verify_caller,
    "get_patient_history": get_patient_history,
    "add_or_update_record": add_or_update_record,
    "list_doctor_slots": list_doctor_slots,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "medical_info_search": medical_info_search,
}

# Tools that read or modify a specific patient's PHI. Calls to these are
# blocked unless verify_caller has been called and returned authorized:true
# for the same patient_id earlier in the same run.
PHI_TOOLS = {
    "get_patient_history",
    "add_or_update_record",  # only blocked when patient_id is set (i.e., update mode)
    "book_appointment",
    "cancel_appointment",
}

OFFICE_PHONE = "+91-123-456-7890"


# ---- State ------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    user_query: str
    patient_id: Optional[int]
    caller_name: Optional[str]
    verified_patient_ids: list[int]
    plan: list[dict]
    step_results: list[dict]
    trace: list[dict]
    final_answer: str
    latency_ms: int
    success: bool


# ---- Utility: $step{N}.path substitution ------------------------------------

# Match $step{N} optionally followed by a path that mixes .foo and [N] segments,
# e.g. $step{3}[0].slot_id, $step{1}.patient_id, $step{2}.0.name, $step{4}[0]
_STEP_REF_RE = re.compile(r"\$step\{(\d+)\}((?:\.[A-Za-z0-9_]+|\[\d+\])*)")

_PATH_SEG_RE = re.compile(r"\.([A-Za-z0-9_]+)|\[(\d+)\]")


def _resolve_path(value: Any, path: Optional[str]) -> Any:
    """Walk a path expression like '.slot_id', '[0].slot_id', '.0.name'."""
    if not path:
        return value
    cur = value
    for m in _PATH_SEG_RE.finditer(path):
        key, idx = m.group(1), m.group(2)
        if idx is not None:
            i = int(idx)
            if isinstance(cur, list) and 0 <= i < len(cur):
                cur = cur[i]
            else:
                return None
        else:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            elif isinstance(cur, list) and key.isdigit() and 0 <= int(key) < len(cur):
                cur = cur[int(key)]
            elif isinstance(cur, list) and cur and isinstance(cur[0], dict) and key in cur[0]:
                # List-of-dicts shortcut: default to first element's field.
                cur = cur[0][key]
            else:
                return None
    return cur


# Args we know should be integers if numeric — coerce after substitution.
_INT_ARGS = {"patient_id", "slot_id", "doctor_id", "appt_id"}


def _maybe_int(key: str, val: Any) -> Any:
    if key in _INT_ARGS and isinstance(val, str) and val.strip().lstrip("-").isdigit():
        return int(val.strip())
    return val


def _substitute_refs(args: dict, step_results: list[dict]) -> dict:
    """Replace $step{N}<path> tokens in arg values with real values, then
    coerce known-integer args (slot_id, patient_id, ...) when they're numeric."""
    out: dict = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            stripped = v.strip()
            m = _STEP_REF_RE.fullmatch(stripped)
            if m:
                idx = int(m.group(1)) - 1
                path = m.group(2) or ""
                if 0 <= idx < len(step_results):
                    out[k] = _resolve_path(step_results[idx]["output"], path)
                else:
                    out[k] = None
            else:
                # Allow embedded references (rare but possible).
                def _sub(mm):
                    i = int(mm.group(1)) - 1
                    p = mm.group(2) or ""
                    if 0 <= i < len(step_results):
                        return str(_resolve_path(step_results[i]["output"], p))
                    return ""
                out[k] = _STEP_REF_RE.sub(_sub, v)
        else:
            out[k] = v
        out[k] = _maybe_int(k, out[k])
    return out


# ---- Planner node -----------------------------------------------------------

def _planner_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    active_pid = state.get("patient_id")
    context = build_context_for_planner(active_pid)
    caller = (state.get("caller_name") or "").strip()
    pre_verified = list(state.get("verified_patient_ids") or [])

    header_lines = []
    if caller:
        header_lines.append(f"Caller name (for verify_caller): {caller}")
    else:
        header_lines.append(
            "Caller name: (not supplied — PHI access will be blocked; "
            "you may still answer general medical_info_search questions)"
        )
    if pre_verified:
        header_lines.append(
            f"Already-verified patient_ids this session: {pre_verified}. "
            "You may SKIP lookup_patient + verify_caller for these patients "
            "and reference them directly by patient_id in subsequent steps."
        )
    if active_pid:
        header_lines.append(
            f"Active patient context: patient_id={active_pid}. "
            "If the user's request refers to 'him', 'her', 'this appointment', "
            "'the slot', or otherwise implies the previously-discussed patient, "
            "use this patient_id directly without calling lookup_patient again."
        )
    context = "\n".join(header_lines) + "\n" + context
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=PLANNER_USER_PROMPT.format(
            user_query=state["user_query"],
            context=context,
        )),
    ]
    resp = llm.invoke(messages)
    raw = resp.content.strip()

    # Strip code fences if the model wrapped the JSON.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        plan = json.loads(raw)
        if not isinstance(plan, list):
            raise ValueError("plan is not a list")
    except Exception as e:
        plan = [{"step": 1, "tool": "_planning_failed", "args": {}, "why": f"parse error: {e}",
                 "raw": raw}]
    state["plan"] = plan
    state.setdefault("trace", []).append({
        "phase": "plan",
        "num_steps": len(plan),
        "plan": plan,
    })
    return state


# ---- Executor node ----------------------------------------------------------

# ---- Inference-leak gate for medical_info_search --------------------------
# Common abbreviation -> full-name aliases so a user query with "CKD" still
# authorises a medical_info_search(condition="chronic kidney disease").
_CONDITION_ALIASES: dict[str, list[str]] = {
    "ckd":  ["chronic kidney disease", "kidney disease", "renal disease"],
    "t2dm": ["type 2 diabetes", "diabetes mellitus", "diabetes"],
    "t1dm": ["type 1 diabetes", "diabetes mellitus", "diabetes"],
    "htn":  ["hypertension", "high blood pressure"],
    "uri":  ["upper respiratory infection", "common cold"],
    "copd": ["chronic obstructive pulmonary disease"],
    "chf":  ["heart failure", "congestive heart failure"],
    "mi":   ["myocardial infarction", "heart attack"],
    "uti":  ["urinary tract infection"],
    "gerd": ["gastroesophageal reflux disease", "acid reflux"],
}


def _condition_derivable_from_query(condition: str, user_query: str) -> bool:
    """Is the requested medical_info_search.condition something the USER named,
    versus something the agent inferred from a patient record?

    Returns True if any of these holds:
      - The condition appears as a substring of the query (case-insensitive).
      - The query contains a known abbreviation that maps to the condition.
      - A meaningful (>3 char) token from the condition appears in the query.

    The bar is intentionally loose so legitimate phrasings ("CKD", "kidney
    problem", "diabetes") still authorise relevant searches; the goal is to
    block topic-inference (user says nothing medical, agent searches "asthma"
    based on Anjali's record).
    """
    if not condition or not user_query:
        return False
    q = user_query.lower()
    cond = condition.lower()

    if cond in q or q in cond:
        return True

    # Expand abbreviations the user might have written.
    for abbr, fulls in _CONDITION_ALIASES.items():
        if re.search(rf"\b{re.escape(abbr)}\b", q):
            if any(f in cond or cond in f for f in fulls):
                return True

    # Reverse direction: condition is an abbreviation, query has the full name.
    if cond in _CONDITION_ALIASES:
        if any(f in q for f in _CONDITION_ALIASES[cond]):
            return True

    # Token overlap fallback — at least one significant (>3 char) token shared.
    cond_tokens = {t for t in re.findall(r"[a-z]+", cond) if len(t) > 3}
    q_tokens = set(re.findall(r"[a-z]+", q))
    if cond_tokens & q_tokens:
        return True
    return False


def _is_phi_call(tool_name: str, resolved_args: dict) -> tuple[bool, Optional[int]]:
    """Return (is_phi, patient_id_being_accessed). For add_or_update_record,
    only counts as PHI if a specific patient_id is being updated (creating a
    brand-new patient record is not access to existing PHI). For
    cancel_appointment, the patient_id is derived from the appt_id."""
    if tool_name == "get_patient_history":
        pid = resolved_args.get("patient_id")
        return True, int(pid) if pid is not None else None
    if tool_name == "book_appointment":
        pid = resolved_args.get("patient_id")
        return True, int(pid) if pid is not None else None
    if tool_name == "add_or_update_record":
        pid = resolved_args.get("patient_id")
        if pid is not None:
            return True, int(pid)
        return False, None
    if tool_name == "cancel_appointment":
        # Look up the appointment's patient_id so the gate can verify against it.
        from .db import get_conn
        appt_id = resolved_args.get("appt_id")
        if appt_id is None:
            return True, None
        try:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT patient_id FROM appointments WHERE appt_id = ?",
                    (int(appt_id),),
                ).fetchone()
            return True, int(row["patient_id"]) if row else None
        except Exception:
            return True, None
    return False, None


def _execute_node(state: AgentState) -> AgentState:
    step_results: list[dict] = []
    trace: list[dict] = state.get("trace", [])
    verified: set[int] = set(state.get("verified_patient_ids", []) or [])

    for step in state.get("plan", []):
        tool_name = step.get("tool")
        raw_args = step.get("args", {}) or {}
        t0 = time.time()
        resolved_args = _substitute_refs(raw_args, step_results)

        # ---- HARD AUTHORIZATION GATE -----------------------------------
        # (1) Block PHI tools unless the corresponding patient_id has been
        # verified earlier in this run. (2) Block medical_info_search whose
        # `condition` argument was not mentioned by the user — without this,
        # the agent can leak PHI by inferring a patient's condition from
        # their record and then "helpfully" researching it.
        # Both checks are enforced in code, not just in the planner prompt.
        block_out: Optional[dict] = None

        is_phi, target_pid = _is_phi_call(tool_name, resolved_args)
        if is_phi and (target_pid is None or target_pid not in verified):
            block_out = {
                "blocked": True,
                "reason": "not_authorized",
                "patient_id": target_pid,
                "office_phone": OFFICE_PHONE,
                "message": (
                    "PHI access blocked: caller has not been verified as an "
                    "authorized contact for this patient."
                ),
            }
        elif tool_name == "medical_info_search":
            cond = resolved_args.get("condition") or ""
            user_q = state.get("user_query", "") or ""
            if not _condition_derivable_from_query(cond, user_q):
                block_out = {
                    "blocked": True,
                    "reason": "condition_not_in_query",
                    "condition": cond,
                    "message": (
                        f"Refused to research '{cond}' because the user did "
                        "not name that condition. Researching a condition "
                        "inferred from a patient's record would leak PHI."
                    ),
                }

        if block_out is not None:
            latency_ms = int((time.time() - t0) * 1000)
            step_results.append({
                "step": step.get("step", len(step_results) + 1),
                "tool": tool_name,
                "args": resolved_args,
                "output": block_out,
                "success": False,
                "latency_ms": latency_ms,
                "blocked_by_gate": True,
            })
            trace.append({
                "phase": "execute",
                "step": step.get("step"),
                "tool": tool_name,
                "args": resolved_args,
                "latency_ms": latency_ms,
                "success": False,
                "blocked_by_gate": True,
                "output_preview": _preview(block_out),
            })
            continue
        # ---- end gate --------------------------------------------------

        if tool_name not in TOOL_REGISTRY:
            out = {"error": f"unknown tool: {tool_name}"}
            success = False
        else:
            tool = TOOL_REGISTRY[tool_name]
            try:
                # Inject caller_name for verify_caller if planner forgot it.
                if tool_name == "verify_caller" and "caller_name" not in resolved_args:
                    resolved_args["caller_name"] = state.get("caller_name") or ""
                out = tool.invoke(resolved_args)   # LangChain tool invoke
                success = "error" not in (out if isinstance(out, dict) else {})
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
                success = False

        latency_ms = int((time.time() - t0) * 1000)
        step_results.append({
            "step": step.get("step", len(step_results) + 1),
            "tool": tool_name,
            "args": resolved_args,
            "output": out,
            "success": success,
            "latency_ms": latency_ms,
        })
        trace.append({
            "phase": "execute",
            "step": step.get("step"),
            "tool": tool_name,
            "args": resolved_args,
            "latency_ms": latency_ms,
            "success": success,
            "output_preview": _preview(out),
        })

        # Side-effect: after a successful lookup_patient, cache patient_id in state.
        if tool_name == "lookup_patient" and isinstance(out, list) and out:
            state["patient_id"] = out[0].get("patient_id")

        # Side-effect: when verify_caller succeeds, mark the patient as verified
        # for the rest of this run.
        if tool_name == "verify_caller" and isinstance(out, dict) and out.get("authorized"):
            pid = out.get("patient_id")
            if pid is not None:
                verified.add(int(pid))

    state["verified_patient_ids"] = sorted(verified)
    state["step_results"] = step_results
    state["trace"] = trace
    return state


def _preview(value: Any, max_chars: int = 300) -> str:
    try:
        s = json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s if len(s) <= max_chars else s[:max_chars] + "..."


# ---- Composer node ----------------------------------------------------------

def _compose_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = FINAL_COMPOSER_PROMPT.format(
        user_query=state["user_query"],
        plan=json.dumps(state.get("plan", []), default=str, indent=2),
        step_results=json.dumps(state.get("step_results", []), default=str, indent=2),
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    state["final_answer"] = resp.content.strip()
    state["success"] = all(s.get("success", True) for s in state.get("step_results", []))
    state.setdefault("trace", []).append({"phase": "compose", "chars": len(state["final_answer"])})
    return state


# ---- Graph ------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", _planner_node)
    g.add_node("executor", _execute_node)
    g.add_node("composer", _compose_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "composer")
    g.add_edge("composer", END)
    return g.compile()


# ---- Public entry point -----------------------------------------------------

def run_agent(user_query: str, patient_id: Optional[int] = None,
              caller_name: Optional[str] = None,
              pre_verified_patient_ids: Optional[list[int]] = None,
              persist: bool = True) -> dict:
    """Run one agent turn end-to-end. Returns the full state dict.

    Args:
        user_query: the natural-language request from the user.
        patient_id: optional pre-resolved patient context (e.g. when the user
            is already viewing a specific patient page, or when this is a
            follow-up turn and we already know who the active patient is).
        caller_name: name of the person making the request. Required for any
            PHI-touching operation; medical-info-only queries can omit it.
        pre_verified_patient_ids: patient_ids that were already verified in
            an earlier turn of this same session. The hard gate honors these
            without re-running verify_caller, so follow-up questions like
            "what about the 11:30 slot?" don't get rejected.
        persist: write the run to the agent_runs table.
    """
    graph = build_graph()
    t0 = time.time()
    init: AgentState = {
        "user_query": user_query,
        "patient_id": patient_id,
        "caller_name": caller_name,
        "step_results": [],
        "trace": [],
        "verified_patient_ids": list(pre_verified_patient_ids or []),
    }
    final = graph.invoke(init)
    final["latency_ms"] = int((time.time() - t0) * 1000)

    if persist:
        final["run_id"] = log_agent_run(
            user_query=user_query,
            plan=final.get("plan", []),
            trace=final.get("trace", []),
            final_answer=final.get("final_answer", ""),
            latency_ms=final["latency_ms"],
            success=bool(final.get("success", True)),
        )
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="User query")
    parser.add_argument("--patient-id", type=int, default=None)
    args = parser.parse_args()

    state = run_agent(args.query, patient_id=args.patient_id)
    print("\n=== FINAL ANSWER ===")
    print(state["final_answer"])
    print("\n=== PLAN ===")
    print(json.dumps(state.get("plan", []), indent=2, default=str))
