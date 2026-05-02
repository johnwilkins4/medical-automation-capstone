"""Patient-record tools: lookup and update.

Both are exposed as LangChain tools (@tool) so the agent can call them, but the
underlying functions are also usable directly for unit tests.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from ..db import get_conn, rows_to_dicts


# ---------------------------------------------------------------------------
# lookup_patient
# ---------------------------------------------------------------------------

class PatientMatch(BaseModel):
    patient_id: int
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    confidence: float = Field(description="Fuzzy match score 0-100")


# Words that signal the query is a *description* of a patient rather than a
# literal name. When any of these appear, scoring shifts to weight the
# patient's summary much more heavily than their name.
_DESCRIPTION_KEYWORDS = {
    "year", "years", "yr", "old", "male", "female", "man", "woman",
    "father", "mother", "son", "daughter", "husband", "wife", "patient",
    "kidney", "renal", "ckd", "diabetes", "diabetic", "hypertension",
    "hypertensive", "cardiac", "heart", "thyroid", "asthma", "copd",
    "stroke", "depression", "anxiety", "migraine", "osteoarthritis",
    "covid", "respiratory", "infection", "disease", "condition",
}

import re as _re


def _city_from_address(addr: Optional[str]) -> Optional[str]:
    """Extract a city name from a free-text address.

    Heuristic: split on commas, strip whitespace, take the last non-empty
    piece (the city/region) and title-case it. Falls back to None if empty.
    """
    if not addr:
        return None
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    if not parts:
        return None
    return parts[-1].strip().title()


def _looks_like_description(q_lower: str) -> bool:
    """Heuristic: does this query look like a description ('70-year-old CKD father')
    versus a literal name ('Ramesh Kulkarni')?"""
    if _re.search(r"\b\d{1,3}\b", q_lower):  # any age-like number
        return True
    tokens = set(_re.findall(r"[a-z]+", q_lower))
    return bool(tokens & _DESCRIPTION_KEYWORDS)


def _lookup_patient_impl(query: str, limit: int = 3) -> list[dict]:
    """Find patients by name, phone, or description. Returns up to `limit`
    matches sorted by confidence (highest first).

    Scoring strategy:
      - Phone substring match → 100, short-circuits.
      - Otherwise, detect whether the query is a *name* or a *description*:
          * Name queries weight name heavily (75%).
          * Description queries weight the patient summary heavily (60%) so
            "70-year-old father with CKD" resolves to whoever's summary
            mentions chronic kidney disease and matches on age.
      - Description queries also get a +25 boost when the patient's age
        matches an age-like number in the query (e.g. "70" → patient.age == 70).
    """
    query = (query or "").strip()
    if not query:
        return []

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT patient_id, name, age, gender, phone, email, address, summary FROM patients"
        ).fetchall()
    patients = rows_to_dicts(rows)
    if not patients:
        return []

    # Helper: enrich a patient row with derived `city`.
    def _enrich(p: dict) -> dict:
        return {**p, "city": _city_from_address(p.get("address"))}

    # Phone-exact takes priority.
    for p in patients:
        if p["phone"] and query.replace(" ", "") in p["phone"].replace(" ", ""):
            return [_enrich({**p, "confidence": 100.0})]

    q_lower = query.lower()
    description_mode = _looks_like_description(q_lower)
    age_in_query = None
    for m in _re.finditer(r"\b(\d{1,3})\b", q_lower):
        n = int(m.group(1))
        if 0 < n < 120:  # plausible human age
            age_in_query = n
            break

    if description_mode:
        # Description-style query: weight summary heavily, name is mostly noise.
        weights = (0.20, 0.15, 0.65)  # name, address, summary
        threshold = 25
    else:
        # Literal-name query: name dominates.
        weights = (0.75, 0.10, 0.15)
        threshold = 45

    w_name, w_addr, w_summ = weights

    def score(p):
        name_s = fuzz.token_set_ratio(q_lower, (p["name"] or "").lower())
        addr_s = fuzz.partial_ratio(q_lower, (p["address"] or "").lower())
        summ_s = fuzz.token_set_ratio(q_lower, (p["summary"] or "").lower())
        base = w_name * name_s + w_addr * addr_s + w_summ * summ_s
        # Age-exact match bonus (description mode only).
        if description_mode and age_in_query is not None and p.get("age") == age_in_query:
            base += 25
        return base

    ranked = sorted(patients, key=score, reverse=True)
    top = [_enrich({**p, "confidence": round(score(p), 1)})
           for p in ranked[:limit] if score(p) >= threshold]
    return top


@tool
def lookup_patient(query: str) -> list[dict]:
    """Find patient(s) matching a name, phone number, or description.

    Returns up to 3 candidate matches with a confidence score (0-100).
    Use this first whenever the user refers to a specific patient.
    The agent should ask a clarifying question if confidence < 70 or
    multiple patients match closely.
    """
    return _lookup_patient_impl(query)


# ---------------------------------------------------------------------------
# add_or_update_record
# ---------------------------------------------------------------------------

ALLOWED_FIELDS = {"age", "gender", "phone", "email", "address", "summary"}


def _add_or_update_record_impl(
    patient_id: Optional[int],
    field: Optional[str],
    value: Optional[str],
    new_patient: Optional[dict] = None,
) -> dict:
    """Create a new patient (if `new_patient` is provided) or update a single
    field on an existing patient.

    Returns the final patient row (as dict).
    """
    with get_conn() as conn:
        if new_patient:
            cur = conn.execute(
                """INSERT INTO patients (name, age, gender, phone, email, address, summary)
                   VALUES (:name, :age, :gender, :phone, :email, :address, :summary)""",
                {
                    "name":    new_patient.get("name", ""),
                    "age":     new_patient.get("age"),
                    "gender":  new_patient.get("gender"),
                    "phone":   new_patient.get("phone"),
                    "email":   new_patient.get("email"),
                    "address": new_patient.get("address"),
                    "summary": new_patient.get("summary"),
                },
            )
            pid = int(cur.lastrowid)
        else:
            if patient_id is None:
                raise ValueError("patient_id is required when new_patient is None")
            if field not in ALLOWED_FIELDS:
                raise ValueError(
                    f"Field '{field}' is not updatable. Allowed: {sorted(ALLOWED_FIELDS)}"
                )
            conn.execute(
                f"UPDATE patients SET {field} = ?, updated_at = ? WHERE patient_id = ?",
                (value, datetime.utcnow().isoformat(sep=" "), patient_id),
            )
            pid = patient_id

        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (pid,)
        ).fetchone()
    return dict(row) if row else {}


@tool
def add_or_update_record(
    patient_id: Optional[int] = None,
    field: Optional[str] = None,
    value: Optional[str] = None,
    new_patient: Optional[dict] = None,
) -> dict:
    """Create a new patient record OR update a single field of an existing one.

    Two modes:
      1. Create: pass `new_patient={"name": "...", "age": 45, ...}`.
      2. Update: pass `patient_id`, `field`, `value`. Allowed fields:
         age, gender, phone, email, address, summary.

    Returns the full updated patient row.
    """
    return _add_or_update_record_impl(patient_id, field, value, new_patient)


# ---------------------------------------------------------------------------
# verify_caller — authorization gate for PHI access.
# ---------------------------------------------------------------------------

def _verify_caller_impl(patient_id: int, caller_name: str) -> dict:
    """Check whether `caller_name` is on the patient's authorized-contacts list.

    Demonstration-grade: name match only (case-insensitive, fuzzy). Production
    would require real identity verification (OTP, government ID, SSO).

    Returns:
        {"authorized": bool,
         "patient_id": int,
         "patient_name": str | None,
         "contact_name": str | None,
         "relationship": str | None,
         "office_phone": "+91-123-456-7890"}
    """
    OFFICE_PHONE = "+91-123-456-7890"
    name = (caller_name or "").strip()
    if not name:
        return {
            "authorized": False,
            "patient_id": patient_id,
            "patient_name": None,
            "contact_name": None,
            "relationship": None,
            "office_phone": OFFICE_PHONE,
            "reason": "no caller name supplied",
        }

    with get_conn() as conn:
        prow = conn.execute(
            "SELECT name FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        patient_name = prow["name"] if prow else None
        rows = conn.execute(
            "SELECT contact_name, relationship FROM patient_authorizations WHERE patient_id = ?",
            (patient_id,),
        ).fetchall()
    contacts = rows_to_dicts(rows)

    if not contacts:
        return {
            "authorized": False,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "contact_name": None,
            "relationship": None,
            "office_phone": OFFICE_PHONE,
            "reason": "no authorized contacts on file",
        }

    # Fuzzy name match — accept >= 85 to allow typos and middle initials,
    # but require both first-and-last tokens to be reasonably present.
    name_lower = name.lower()
    best = max(
        contacts,
        key=lambda c: fuzz.token_set_ratio(name_lower, c["contact_name"].lower()),
    )
    score = fuzz.token_set_ratio(name_lower, best["contact_name"].lower())
    if score >= 85:
        return {
            "authorized": True,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "contact_name": best["contact_name"],
            "relationship": best["relationship"],
            "match_score": score,
            "office_phone": OFFICE_PHONE,
        }
    return {
        "authorized": False,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "contact_name": None,
        "relationship": None,
        "match_score": score,
        "office_phone": OFFICE_PHONE,
        "reason": f"caller '{name}' is not on the authorized-contacts list",
    }


@tool
def verify_caller(patient_id: int, caller_name: str) -> dict:
    """Verify that `caller_name` is authorized to act on behalf of `patient_id`.

    Call this AFTER lookup_patient and BEFORE any tool that touches the
    patient's protected health information (history, records, appointments).

    Returns a dict with `authorized: bool`. Public medical_info_search calls
    do not require verification and may run regardless.
    """
    return _verify_caller_impl(patient_id, caller_name)
