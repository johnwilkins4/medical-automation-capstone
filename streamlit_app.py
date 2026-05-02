"""Streamlit dashboard for the Agentic Healthcare Assistant.

Run from the project root:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import run_agent
from src.config import APP_PASSCODE
from src.db import get_conn, recent_runs
from src.tools.patients import _add_or_update_record_impl, _lookup_patient_impl
from src.tools.scheduling import _book_appointment_impl, _list_doctor_slots_impl


st.set_page_config(page_title="Agentic Healthcare Assistant", layout="wide")


# ---------------------------------------------------------------------------
# Passcode gate (optional). If APP_PASSCODE is set in Streamlit secrets, the
# user must enter it once per session before the app loads. Locally the
# secret is unset, so this is a no-op and the app runs as before.
# ---------------------------------------------------------------------------
if APP_PASSCODE:
    if not st.session_state.get("passcode_ok"):
        st.markdown("# Agentic Healthcare Assistant")
        st.caption(
            "This is a capstone demo. To prevent random use of the deployer's "
            "OpenAI key, please enter the passcode shared with you."
        )
        entered = st.text_input("Passcode", type="password")
        if st.button("Enter"):
            if entered == APP_PASSCODE:
                st.session_state["passcode_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect passcode.")
        st.stop()


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

# Note: deliberately NOT cached. The patients table is small (single-digit
# rows in this demo) and is mutated frequently from both the form and the
# agent. Caching introduced cross-widget staleness (e.g. a newly-created
# patient appearing in the table but missing from the update dropdown until
# the cache TTL expired). Reading on every render is effectively free.
def load_patients() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM patients ORDER BY patient_id", conn)
    return df


@st.cache_data(ttl=5)
def load_doctors() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM doctors ORDER BY specialty, name", conn)
    return df


@st.cache_data(ttl=5)
def load_appointments() -> pd.DataFrame:
    q = """
    SELECT a.appt_id, a.slot_time, a.status,
           p.name AS patient_name, p.patient_id,
           d.name AS doctor_name, d.specialty, d.location
    FROM appointments a
    JOIN patients p ON p.patient_id = a.patient_id
    JOIN doctors  d ON d.doctor_id  = a.doctor_id
    ORDER BY a.slot_time DESC
    """
    with get_conn() as conn:
        return pd.read_sql(q, conn)


@st.cache_data(ttl=5)
def load_run_history(limit: int = 50) -> list[dict]:
    return recent_runs(limit=limit)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Agentic Healthcare Assistant")
st.caption(
    "Capstone for Applied Generative AI Specialisation (Purdue / Simplilearn). "
    "All patient data is synthetic. Not medical advice."
)

tabs = st.tabs([
    "Chat", "Patients", "Doctors & Appointments", "Agent Trace", "Evaluation"
])


# ---------------------------------------------------------------------------
# Tab 1 — Chat
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Chat with the assistant")

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("**Try one of these:**")
        samples = [
            "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
            "Book a follow-up with an endocrinologist for David Thompson next week.",
            "What does Anjali Mehra's record say about her last visit?",
            "What are the latest treatments for hypertension?",
            "Update Ramesh Kulkarni's phone to +91-98220-99999.",
        ]
        for s in samples:
            if st.button(s, key=f"sample_{hash(s)}"):
                st.session_state["pending_query"] = s

    with col1:
        caller = st.text_input(
            "Your name (required to access a patient's records)",
            value=st.session_state.get("caller_name", ""),
            key="caller_input",
            placeholder="e.g. Priya Kumar",
            help=(
                "We'll check this against the patient's authorized-contacts "
                "list. Public medical-info questions don't require a name. "
                "Verification carries forward across follow-up turns until "
                "you change names or click 'New caller / clear session'."
            ),
        )

        # If the caller name changed since last turn, wipe verification +
        # active-patient state — different person, different trust.
        if caller.strip() != (st.session_state.get("caller_name") or "").strip():
            st.session_state["verified_patient_ids"] = []
            st.session_state["last_patient_id"] = None
        st.session_state["caller_name"] = caller

        q = st.text_area(
            "Your request",
            value=st.session_state.get("pending_query", ""),
            height=100,
            key="chat_input",
        )
        c_send, c_reset = st.columns([1, 1])
        with c_send:
            submit = st.button("Send", type="primary")
        with c_reset:
            if st.button("New caller / clear session"):
                for k in ("verified_patient_ids", "last_patient_id",
                          "caller_name", "pending_query"):
                    st.session_state.pop(k, None)
                st.rerun()

        # Surface the active session state so the user can see what's carrying.
        verified = st.session_state.get("verified_patient_ids") or []
        active_pid = st.session_state.get("last_patient_id")
        if verified or active_pid:
            with st.expander("Session context (carries across turns)", expanded=False):
                st.write({
                    "caller": caller,
                    "verified_patient_ids": verified,
                    "active_patient_id": active_pid,
                })

    if submit and q.strip():
        with st.spinner("Agent thinking..."):
            state = run_agent(
                q.strip(),
                caller_name=caller.strip() or None,
                patient_id=st.session_state.get("last_patient_id"),
                pre_verified_patient_ids=st.session_state.get("verified_patient_ids") or [],
            )
        # Persist the resulting verification + active patient back into session.
        st.session_state["verified_patient_ids"] = sorted(set(
            (st.session_state.get("verified_patient_ids") or [])
            + (state.get("verified_patient_ids") or [])
        ))
        if state.get("patient_id"):
            st.session_state["last_patient_id"] = state["patient_id"]

        load_run_history.clear()
        load_appointments.clear()

        st.markdown("### Response")
        st.markdown(state.get("final_answer", "(no answer)"))

        with st.expander("Plan", expanded=False):
            st.json(state.get("plan", []))
        with st.expander("Step results", expanded=False):
            st.json(state.get("step_results", []))
        st.caption(
            f"Run #{state.get('run_id')} · "
            f"{state.get('latency_ms', 0)} ms · "
            f"success={state.get('success')}"
        )
        st.session_state.pop("pending_query", None)


# ---------------------------------------------------------------------------
# Tab 2 — Patients
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Patient records")
    df = load_patients()
    st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Add or update a patient"):
        mode = st.radio("Mode", ["Create new", "Update existing"], horizontal=True)
        if mode == "Create new":
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Name *")
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
            with c2:
                gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
                phone = st.text_input("Phone")
            with c3:
                email = st.text_input("Email")
                address = st.text_input("Address")
            summary = st.text_area("Summary (clinical)", height=80)
            if st.button("Create patient", type="primary"):
                if not name.strip():
                    st.error("Name is required.")
                else:
                    row = _add_or_update_record_impl(None, None, None, {
                        "name": name, "age": int(age), "gender": gender,
                        "phone": phone, "email": email, "address": address,
                        "summary": summary,
                    })
                    st.success(f"Created patient #{row['patient_id']}")
                    st.rerun()
        else:
            if df.empty:
                st.info("No patients to update.")
            else:
                pid = st.selectbox(
                    "Patient",
                    df["patient_id"].tolist(),
                    format_func=lambda p: f"#{p} — {df.loc[df.patient_id==p,'name'].iloc[0]}",
                )
                field = st.selectbox("Field",
                    ["age", "gender", "phone", "email", "address", "summary"])
                value = st.text_input("New value")
                if st.button("Update", type="primary"):
                    row = _add_or_update_record_impl(int(pid), field, value)
                    st.success(f"Updated {field} for patient #{row['patient_id']}")
                    st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 — Doctors & Appointments
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Doctors")
    st.dataframe(load_doctors(), width="stretch", hide_index=True)

    st.subheader("Upcoming appointments")
    st.dataframe(load_appointments(), width="stretch", hide_index=True)

    with st.expander("Manually book a slot"):
        specialty = st.selectbox(
            "Specialty",
            ["Nephrologist", "Cardiologist", "Endocrinologist",
             "General Physician", "Pulmonologist", "Dermatologist"],
        )
        window = st.slider("Window (days)", 1, 14, 7)
        if st.button("Find slots"):
            slots = _list_doctor_slots_impl(specialty, window_days=window)
            st.session_state["slots"] = slots

        slots = st.session_state.get("slots", [])
        if slots:
            sdf = pd.DataFrame(slots)
            st.dataframe(sdf[["slot_id", "slot_time", "doctor_name", "location"]],
                         width="stretch", hide_index=True)
            patients_df = load_patients()
            pid = st.selectbox(
                "Patient",
                patients_df["patient_id"].tolist(),
                format_func=lambda p: f"#{p} — {patients_df.loc[patients_df.patient_id==p,'name'].iloc[0]}",
            )
            slot_id = st.selectbox("Slot", [s["slot_id"] for s in slots])
            if st.button("Confirm booking", type="primary"):
                out = _book_appointment_impl(int(pid), int(slot_id))
                if "error" in out:
                    st.error(out["error"])
                else:
                    st.success(
                        f"Booked #{out['appt_id']}: {out['patient_name']} with "
                        f"{out['doctor_name']} on {out['slot_time']}"
                    )
                    load_appointments.clear()
                    st.session_state.pop("slots", None)
                    st.rerun()


# ---------------------------------------------------------------------------
# Tab 4 — Agent Trace
# ---------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Agent run history")
    runs = load_run_history(limit=100)
    if not runs:
        st.info("No agent runs yet. Send a query from the Chat tab first.")
    else:
        rdf = pd.DataFrame([{
            "run_id": r["run_id"],
            "created_at": r["created_at"],
            "query": (r["user_query"] or "")[:80],
            "latency_ms": r["latency_ms"],
            "success": bool(r["success"]),
        } for r in runs])
        st.dataframe(rdf, width="stretch", hide_index=True)

        run_id = st.selectbox("Inspect run", [r["run_id"] for r in runs])
        run = next(r for r in runs if r["run_id"] == run_id)
        st.markdown(f"**Query:** {run['user_query']}")
        st.markdown(f"**Final answer:**")
        st.write(run["final_answer"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Plan**")
            st.json(json.loads(run["plan_json"] or "[]"))
        with col2:
            st.markdown("**Trace (step-by-step)**")
            st.json(json.loads(run["trace_json"] or "[]"))


# ---------------------------------------------------------------------------
# Tab 5 — Evaluation
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Model evaluation")
    st.caption(
        "Run the labeled test set (15 queries) through the agent. "
        "Uses LangChain's QAEvalChain for correctness and tracks per-tool success."
    )

    eval_file = Path(__file__).parent / "eval" / "last_results.json"
    if eval_file.exists():
        data = json.loads(eval_file.read_text())
        st.markdown(f"**Last run:** {data.get('ran_at', 'unknown')}")
        summary = data.get("summary", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", summary.get("total", 0))
        c2.metric("Correct", summary.get("correct", 0))
        c3.metric("Accuracy", f"{summary.get('accuracy', 0)*100:.1f}%")
        c4.metric("Median latency (ms)", summary.get("median_latency_ms", 0))

        st.markdown("**Per-capability accuracy**")
        st.bar_chart(pd.DataFrame(summary.get("by_capability", {}), index=["accuracy"]).T)

        st.markdown("**Per-tool success rate**")
        st.bar_chart(pd.DataFrame(summary.get("tool_success", {}), index=["success_rate"]).T)

        with st.expander("Per-query breakdown"):
            st.dataframe(pd.DataFrame(data.get("items", [])),
                         width="stretch", hide_index=True)
    else:
        st.info("No evaluation run yet. From the terminal: `python -m eval.run_eval`")

    if st.button("Run evaluation now", type="primary"):
        st.info("Kicking off evaluation in-process (may take ~30-90 seconds)...")
        with st.spinner("Evaluating..."):
            from eval.run_eval import run_evaluation  # lazy import
            run_evaluation(write_to=eval_file)
        st.rerun()
