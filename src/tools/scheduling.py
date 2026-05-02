"""Doctor-scheduling tools: list open slots, book an appointment.

The doctor calendar lives in the `doctors` + `schedule` tables. Each booking
flips `schedule.is_booked = 1` and inserts a row into `appointments`.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool

from ..config import DEFAULT_SLOT_WINDOW_DAYS
from ..db import get_conn, rows_to_dicts


# Map user-facing specialty synonyms to the canonical names in the doctors table.
# Keys are partial matches — we look for any key as a substring of the input.
SPECIALTY_ALIASES = {
    "nephro":      "Nephrologist",       # nephrology, nephrologist
    "kidney":      "Nephrologist",
    "renal":       "Nephrologist",
    "cardio":      "Cardiologist",       # cardiology, cardiologist, cardio
    "heart":       "Cardiologist",
    "endocrin":    "Endocrinologist",    # endocrinology, endocrinologist
    "diabet":      "Endocrinologist",    # diabetes, diabetic
    "thyroid":     "Endocrinologist",
    "hormone":     "Endocrinologist",
    "general physician": "General Physician",
    "general":     "General Physician",
    "gp":          "General Physician",
    "family":      "General Physician",
    "pulmon":      "Pulmonologist",      # pulmonology, pulmonologist
    "lung":        "Pulmonologist",
    "respirator":  "Pulmonologist",
    "dermat":      "Dermatologist",      # dermatology, dermatologist
    "skin":        "Dermatologist",
}

# Canonical specialties that exist in the doctors table (used as fallback).
_CANONICAL = {"Nephrologist", "Cardiologist", "Endocrinologist",
              "General Physician", "Pulmonologist", "Dermatologist"}


def _normalize_specialty(s: str) -> str:
    """Resolve the input to one of the canonical specialty names.

    1. If `s` already equals a canonical name, use it.
    2. Otherwise scan SPECIALTY_ALIASES for any key that appears as a
       substring (case-insensitive) — first hit wins. This catches
       "Nephrology" -> nephro -> Nephrologist, "kidney doctor" -> kidney
       -> Nephrologist, "dr. for diabetes" -> diabet -> Endocrinologist.
    3. If no match, return the input title-cased so the SQL filter just
       returns no rows rather than raising.
    """
    if not s:
        return ""
    raw = s.strip()
    if raw in _CANONICAL:
        return raw
    s_l = raw.lower()
    for key, canon in SPECIALTY_ALIASES.items():
        if key in s_l:
            return canon
    return raw.title()


# ---------------------------------------------------------------------------
# list_doctor_slots
# ---------------------------------------------------------------------------

def _list_doctor_slots_impl(
    specialty: str,
    window_days: int = DEFAULT_SLOT_WINDOW_DAYS,
    location: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """List open slots for a specialty.

    If `location` is passed, returns only slots in that city when any exist.
    If no slots match the location, falls back to slots in any city and marks
    each result with `out_of_city: true` so the agent can mention it. This
    keeps the patient near home when possible without leaving them stuck.
    """
    canonical = _normalize_specialty(specialty)
    now = datetime.now()
    end = now + timedelta(days=window_days)

    base_sql = """
        SELECT s.slot_id, s.slot_time, d.doctor_id, d.name AS doctor_name,
               d.specialty, d.location
        FROM schedule s
        JOIN doctors d ON d.doctor_id = s.doctor_id
        WHERE s.is_booked = 0
          AND d.specialty = ?
          AND s.slot_time BETWEEN ? AND ?
    """
    base_params = [canonical, now.isoformat(sep=" "), end.isoformat(sep=" ")]
    tail = " ORDER BY s.slot_time ASC LIMIT ?"

    with get_conn() as conn:
        if location:
            sql = base_sql + " AND d.location = ?" + tail
            params = base_params + [location, limit]
            rows = conn.execute(sql, params).fetchall()
            local = rows_to_dicts(rows)
            if local:
                # Add an explicit flag so the composer can confirm "in your city".
                return [{**r, "out_of_city": False} for r in local]
            # Fallback: any city, but mark the results so the composer can warn.
            sql = base_sql + tail
            rows = conn.execute(sql, base_params + [limit]).fetchall()
            return [{**r, "out_of_city": True, "preferred_location": location}
                    for r in rows_to_dicts(rows)]

        # No location preference — just return what's available, marked neutral.
        sql = base_sql + tail
        rows = conn.execute(sql, base_params + [limit]).fetchall()
        return [{**r, "out_of_city": False} for r in rows_to_dicts(rows)]


@tool
def list_doctor_slots(
    specialty: str,
    window_days: int = DEFAULT_SLOT_WINDOW_DAYS,
    location: Optional[str] = None,
) -> list[dict]:
    """List available appointment slots for a specialty within the next `window_days`.

    `specialty` accepts canonical names ("Nephrologist") and informal ones
    ("kidney doctor", "nephrology"). When `location` (city) is supplied the
    function prefers same-city slots and only falls back to other cities if
    none are available — fallback rows are marked `out_of_city: true`.

    Returns a list of slot dicts: `{slot_id, slot_time, doctor_id, doctor_name,
    specialty, location, out_of_city}`.
    """
    return _list_doctor_slots_impl(specialty, window_days, location)


# ---------------------------------------------------------------------------
# book_appointment
# ---------------------------------------------------------------------------

def _book_appointment_impl(patient_id: int, slot_id: int) -> dict:
    with get_conn() as conn:
        slot = conn.execute(
            """SELECT s.slot_id, s.doctor_id, s.slot_time, s.is_booked,
                      d.name AS doctor_name, d.specialty, d.location
                 FROM schedule s JOIN doctors d ON d.doctor_id = s.doctor_id
                WHERE s.slot_id = ?""",
            (slot_id,),
        ).fetchone()
        if not slot:
            return {"error": f"slot_id {slot_id} not found"}
        if slot["is_booked"]:
            return {"error": f"slot_id {slot_id} already booked"}

        patient = conn.execute(
            "SELECT patient_id, name FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        if not patient:
            return {"error": f"patient_id {patient_id} not found"}

        cur = conn.execute(
            """INSERT INTO appointments (patient_id, doctor_id, slot_time, status)
               VALUES (?, ?, ?, 'booked')""",
            (patient_id, slot["doctor_id"], slot["slot_time"]),
        )
        appt_id = int(cur.lastrowid)
        conn.execute("UPDATE schedule SET is_booked = 1 WHERE slot_id = ?", (slot_id,))

    return {
        "appt_id": appt_id,
        "patient_id": patient_id,
        "patient_name": patient["name"],
        "doctor_id": slot["doctor_id"],
        "doctor_name": slot["doctor_name"],
        "specialty": slot["specialty"],
        "location": slot["location"],
        "slot_time": slot["slot_time"],
        "status": "booked",
    }


@tool
def book_appointment(patient_id: int, slot_id: int) -> dict:
    """Book an available slot for a patient.

    Pass the `slot_id` returned by `list_doctor_slots` plus the patient_id
    resolved by `lookup_patient`. Returns a confirmation dict with appointment
    details, or `{"error": ...}` if the slot is already taken.
    """
    return _book_appointment_impl(patient_id, slot_id)


# ---------------------------------------------------------------------------
# cancel_appointment
# ---------------------------------------------------------------------------

def _cancel_appointment_impl(appt_id: int) -> dict:
    """Cancel an existing appointment and free its slot.

    Marks `appointments.status = 'cancelled'` and flips
    `schedule.is_booked = 0` for the underlying slot so it can be re-booked.
    """
    with get_conn() as conn:
        appt = conn.execute(
            """SELECT a.appt_id, a.patient_id, a.doctor_id, a.slot_time, a.status,
                      p.name AS patient_name, d.name AS doctor_name,
                      d.specialty, d.location
                 FROM appointments a
                 JOIN patients p ON p.patient_id = a.patient_id
                 JOIN doctors  d ON d.doctor_id  = a.doctor_id
                WHERE a.appt_id = ?""",
            (appt_id,),
        ).fetchone()
        if not appt:
            return {"error": f"appt_id {appt_id} not found"}
        if appt["status"] == "cancelled":
            return {"error": f"appt_id {appt_id} is already cancelled"}

        # Free the underlying slot.
        conn.execute(
            """UPDATE schedule
                  SET is_booked = 0
                WHERE doctor_id = ? AND slot_time = ?""",
            (appt["doctor_id"], appt["slot_time"]),
        )
        conn.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE appt_id = ?",
            (appt_id,),
        )

    return {
        "appt_id": appt_id,
        "patient_id": appt["patient_id"],
        "patient_name": appt["patient_name"],
        "doctor_name": appt["doctor_name"],
        "specialty": appt["specialty"],
        "location": appt["location"],
        "slot_time": appt["slot_time"],
        "status": "cancelled",
    }


@tool
def cancel_appointment(appt_id: int) -> dict:
    """Cancel a booked appointment and release its slot.

    Pass the `appt_id` returned by `book_appointment` (or visible in the
    Appointments tab). The previously-booked slot becomes available again
    so the same caregiver can rebook to a different time. PHI tool —
    requires the patient on the appointment to have been verify_caller'd
    earlier in the run.
    """
    return _cancel_appointment_impl(appt_id)
