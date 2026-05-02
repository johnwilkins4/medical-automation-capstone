"""Evaluation harness.

Runs the labeled test set through the agent, grades each response with
LangChain's QAEvalChain, and aggregates per-capability / per-tool metrics.

Run from project root:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI

from src.agent import run_agent
from src.config import LLM_MODEL


THIS_DIR = Path(__file__).resolve().parent
TEST_SET = THIS_DIR / "test_set.jsonl"
DEFAULT_OUTPUT = THIS_DIR / "last_results.json"


def _load_test_set() -> list[dict]:
    return [json.loads(line) for line in TEST_SET.read_text().splitlines() if line.strip()]


def _tool_overlap(expected: list[str], actual: list[str]) -> float:
    """Simple set-overlap score: |expected ∩ actual| / |expected|."""
    if not expected:
        return 1.0
    exp = set(expected)
    act = set(actual)
    return len(exp & act) / len(exp)


def _grade_responses(items: list[dict]) -> list[dict]:
    """Run QAEvalChain over (query, reference, prediction) triples."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    evaluator = QAEvalChain.from_llm(llm)

    examples = [{"query": it["query"], "answer": it["reference"]} for it in items]
    predictions = [{"result": it["prediction"]} for it in items]
    graded = evaluator.evaluate(examples, predictions,
                                question_key="query",
                                answer_key="answer",
                                prediction_key="result")
    for it, g in zip(items, graded):
        verdict = (g.get("results") or g.get("text") or "").strip().upper()
        it["correct"] = "CORRECT" in verdict
        it["grader_raw"] = g
    return items


def run_evaluation(write_to: Optional[Path] = None) -> dict:
    items = _load_test_set()
    out_items: list[dict] = []

    for it in items:
        t0 = time.time()
        try:
            state = run_agent(
                it["query"],
                caller_name=it.get("caller_name"),
                persist=False,
            )
            pred = state.get("final_answer", "")
            tools_used = [s.get("tool") for s in state.get("step_results", [])]
        except Exception as e:
            pred = f"(agent error: {type(e).__name__}: {e})"
            tools_used = []
        latency_ms = int((time.time() - t0) * 1000)

        tool_overlap = _tool_overlap(it.get("expected_tools", []), tools_used)
        out_items.append({
            **it,
            "prediction": pred,
            "tools_used": tools_used,
            "tool_overlap": round(tool_overlap, 3),
            "latency_ms": latency_ms,
        })

    # Grade with QAEvalChain (may cost a few API calls; skip quietly on failure).
    try:
        out_items = _grade_responses(out_items)
    except Exception as e:
        print(f"[warn] QAEvalChain failed: {e}. Marking all items as ungraded.")
        for it in out_items:
            it["correct"] = None
            it["grader_raw"] = {"error": str(e)}

    # Aggregate.
    total = len(out_items)
    correct = sum(1 for it in out_items if it.get("correct"))

    by_cap: dict[str, list[bool]] = defaultdict(list)
    for it in out_items:
        by_cap[it["capability"]].append(bool(it.get("correct")))
    per_cap_acc = {k: round(sum(v) / len(v), 3) if v else 0 for k, v in by_cap.items()}

    tool_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "total": 0})
    for it in out_items:
        for t in it["tools_used"]:
            tool_counts[t]["total"] += 1
            tool_counts[t]["ok"] += 1  # if it made it into step_results, it didn't raise
    tool_success = {t: round(v["ok"] / v["total"], 3) if v["total"] else 0
                    for t, v in tool_counts.items()}

    latencies = [it["latency_ms"] for it in out_items]
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0,
        "by_capability": per_cap_acc,
        "tool_success": tool_success,
        "median_latency_ms": int(statistics.median(latencies)) if latencies else 0,
        "p95_latency_ms": int(sorted(latencies)[int(0.95 * len(latencies)) - 1]) if latencies else 0,
    }

    result = {
        "ran_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": LLM_MODEL,
        "summary": summary,
        "items": out_items,
    }

    out_path = write_to or DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote results to {out_path}")
    print(f"Accuracy: {summary['accuracy']*100:.1f}%  "
          f"({summary['correct']}/{summary['total']}), "
          f"median latency {summary['median_latency_ms']} ms")
    return result


if __name__ == "__main__":
    run_evaluation()
