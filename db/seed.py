"""Build healthcare.db from seed data.

Run from the project root:
    python -m db.seed

What this does:
  1. Drops and recreates all tables from schema.sql.
  2. Loads records.xlsx into the patients table (dedup + cleanup).
  3. Creates a synthetic doctor roster covering the specialties we'll demo.
  4. Generates 14 days of half-hour slots (weekdays, 9am-5pm) per doctor.
"""
from __future__ import annotations

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

# Allow running as `python -m db.seed` OR `python db/seed.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RECORDS_XLSX, DB_PATH  # noqa: E402
from src.db import get_conn, init_schema  # noqa: E402


# ---- Doctors ----------------------------------------------------------------

DOCTORS = [
    # (name, specialty, location)
    ("Dr. Priya Sharma",      "Nephrologist",     "Pune"),
    ("Dr. Arjun Kapoor",      "Nephrologist",     "Bangalore"),
    ("Dr. Emily Watson",      "Cardiologist",     "Chennai"),
    ("Dr. Rajesh Nair",       "Cardiologist",     "Pune"),
    ("Dr. Sunita Rao",        "Endocrinologist",  "Bangalore"),
    ("Dr. Michael Chen",      "Endocrinologist",  "Chennai"),
    ("Dr. Neha Gupta",        "General Physician", "Pune"),
    ("Dr. Rohan Desai",       "General Physician", "Chennai"),
    ("Dr. Lisa O'Connor",     "Pulmonologist",    "Bangalore"),
    ("Dr. Vikram Patel",      "Dermatologist",    "Pune"),
]

SPECIALTY_ALIASES = {
    "kidney":      "Nephrologist",
    "renal":       "Nephrologist",
    "heart":       "Cardiologist",
    "cardiac":     "Cardiologist",
    "diabetes":    "Endocrinologist",
    "thyroid":     "Endocrinologist",
    "hormone":     "Endocrinologist",
    "gp":          "General Physician",
    "family":      "General Physician",
    "lung":        "Pulmonologist",
    "respiratory": "Pulmonologist",
    "skin":        "Dermatologist",
}


# ---- Patients ---------------------------------------------------------------

def clean_phone(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    # 10-digit Indian numbers without + prefix — add it.
    if s.isdigit() and len(s) == 10:
        return f"+91-{s[:5]}-{s[5:]}"
    return s


def load_patients() -> list[tuple]:
    df = pd.read_excel(RECORDS_XLSX)
    df = df.drop_duplicates(subset=["Name", "Phone_number"]).reset_index(drop=True)

    patients: list[tuple] = []
    for _, row in df.iterrows():
        patients.append((
            str(row["Name"]).strip(),
            int(row["Age"]) if not pd.isna(row["Age"]) else None,
            str(row["Gender"]).strip() if not pd.isna(row["Gender"]) else None,
            clean_phone(row["Phone_number"]),
            str(row["Email"]).strip() if not pd.isna(row["Email"]) else None,
            str(row["Address"]).strip() if not pd.isna(row["Address"]) else None,
            str(row["Summary"]).strip() if not pd.isna(row["Summary"]) else None,
        ))

    # Add a synthetic "father" patient for the flagship CKD demo scenario.
    patients.append((
        "Mr. Harish Kumar",
        70,
        "Male",
        "+91-98765-00001",
        None,
        "12 MG Road, Bangalore",
        "70-year-old male with a 4-year history of chronic kidney disease (CKD stage 3b). "
        "Comorbid hypertension managed with amlodipine. Last eGFR 38. No dialysis yet. "
        "Demo record seeded for the CKD flagship scenario.",
    ))
    return patients


# ---- Authorized contacts ----------------------------------------------------

# (patient_name_substring, contact_name, relationship, phone)
# Matched against the patient name with `LIKE %X%` so partial matches work
# regardless of titles ("Mr.") or middle names.
AUTHORIZATIONS: list[tuple[str, str, str, str]] = [
    # Flagship CKD demo patient
    ("Harish Kumar",     "Priya Kumar",         "daughter", "+91-98765-10001"),
    ("Harish Kumar",     "Anil Kumar",          "son",      "+91-98765-10002"),
    # Other H&P patients
    ("David Thompson",   "Sarah Thompson",      "wife",     "+1-415-555-0142"),
    ("David Thompson",   "Michael Thompson",    "son",      "+1-415-555-0177"),
    ("Anjali Mehra",     "Vikram Mehra",        "husband",  "+91-98220-77711"),
    ("Ramesh Kulkarni",  "Sunita Kulkarni",     "wife",     "+91-98220-88822"),
    ("Ramesh Kulkarni",  "Anish Kulkarni",      "son",      "+91-98220-88833"),
]


# ---- Schedule ---------------------------------------------------------------

def generate_slots(start_date: datetime, days: int = 14) -> list[datetime]:
    """Weekdays only, 09:00–17:00, every 30 minutes."""
    slots: list[datetime] = []
    for d in range(days):
        day = start_date + timedelta(days=d)
        if day.weekday() >= 5:  # 5 = Sat, 6 = Sun
            continue
        base = datetime.combine(day.date(), time(9, 0))
        for half_hours in range(16):  # 16 half-hours = 9am..5pm
            slots.append(base + timedelta(minutes=30 * half_hours))
    return slots


# ---- Main -------------------------------------------------------------------

def main() -> None:
    print(f"Initializing schema at {DB_PATH}...")
    init_schema()

    with get_conn() as conn:
        # Patients
        patients = load_patients()
        conn.executemany(
            """INSERT INTO patients
                   (name, age, gender, phone, email, address, summary)
               VALUES (?,?,?,?,?,?,?)""",
            patients,
        )
        print(f"  inserted {len(patients)} patients")

        # Doctors
        conn.executemany(
            "INSERT INTO doctors (name, specialty, location) VALUES (?,?,?)",
            DOCTORS,
        )
        print(f"  inserted {len(DOCTORS)} doctors")

        # Schedule
        start = datetime.combine(datetime.now().date(), time(0, 0))
        slots = generate_slots(start, days=14)
        doctor_ids = [r["doctor_id"] for r in conn.execute("SELECT doctor_id FROM doctors").fetchall()]
        rows = [(did, slot.isoformat(sep=" ")) for did in doctor_ids for slot in slots]
        conn.executemany(
            "INSERT OR IGNORE INTO schedule (doctor_id, slot_time) VALUES (?, ?)",
            rows,
        )
        print(f"  inserted {len(rows)} schedule slots "
              f"({len(slots)} per doctor across {len(doctor_ids)} doctors)")

        # Authorized contacts — resolve patient_ids by name match, then insert.
        auth_rows: list[tuple] = []
        for name_substr, contact, rel, phone in AUTHORIZATIONS:
            row = conn.execute(
                "SELECT patient_id FROM patients WHERE name LIKE ? LIMIT 1",
                (f"%{name_substr}%",),
            ).fetchone()
            if row is None:
                print(f"  [warn] no patient matched '{name_substr}', skipping auth for {contact}")
                continue
            auth_rows.append((row["patient_id"], contact, rel, phone))
        conn.executemany(
            """INSERT OR IGNORE INTO patient_authorizations
                   (patient_id, contact_name, relationship, phone)
               VALUES (?,?,?,?)""",
            auth_rows,
        )
        print(f"  inserted {len(auth_rows)} authorized contacts")

    print(f"Done. Database ready at {DB_PATH}")


if __name__ == "__main__":
    main()
