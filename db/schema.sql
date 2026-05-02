-- Healthcare Assistant schema.
-- One-shot DDL. Drop tables first so seed.py can re-run idempotently.

DROP TABLE IF EXISTS patient_authorizations;
DROP TABLE IF EXISTS appointments;
DROP TABLE IF EXISTS schedule;
DROP TABLE IF EXISTS doctors;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS agent_runs;

CREATE TABLE patients (
    patient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    age          INTEGER,
    gender       TEXT,
    phone        TEXT,
    email        TEXT,
    address      TEXT,
    summary      TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patients_name ON patients(name);
CREATE INDEX idx_patients_phone ON patients(phone);

CREATE TABLE doctors (
    doctor_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    specialty    TEXT NOT NULL,
    location     TEXT
);

CREATE INDEX idx_doctors_specialty ON doctors(specialty);

CREATE TABLE schedule (
    slot_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_id    INTEGER NOT NULL REFERENCES doctors(doctor_id),
    slot_time    TIMESTAMP NOT NULL,
    is_booked    INTEGER DEFAULT 0,
    UNIQUE (doctor_id, slot_time)
);

CREATE INDEX idx_schedule_doctor_time ON schedule(doctor_id, slot_time);

CREATE TABLE appointments (
    appt_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER NOT NULL REFERENCES patients(patient_id),
    doctor_id    INTEGER NOT NULL REFERENCES doctors(doctor_id),
    slot_time    TIMESTAMP NOT NULL,
    status       TEXT DEFAULT 'booked',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (patient_id, doctor_id, slot_time)
);

CREATE INDEX idx_appt_patient ON appointments(patient_id);

CREATE TABLE agent_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_query   TEXT NOT NULL,
    plan_json    TEXT,
    trace_json   TEXT,
    final_answer TEXT,
    latency_ms   INTEGER,
    success      INTEGER DEFAULT 1,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agent_runs_created ON agent_runs(created_at DESC);

-- Authorized contacts who may act on a patient's behalf (PHI access gate).
-- Demonstration-grade: name match only. Production would require real identity
-- verification (OTP, government ID, SSO).
CREATE TABLE patient_authorizations (
    auth_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER NOT NULL REFERENCES patients(patient_id),
    contact_name TEXT NOT NULL,
    relationship TEXT,
    phone        TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (patient_id, contact_name)
);

CREATE INDEX idx_patient_auth_patient ON patient_authorizations(patient_id);
CREATE INDEX idx_patient_auth_name ON patient_authorizations(contact_name);
