"""Central configuration. Reads from env with sensible defaults.

All paths are resolved relative to the project root (parent of src/).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_secret(name: str, default: str = "") -> str:
    """Read a secret from (in order): the OS env, .env file (already loaded
    via load_dotenv above), or Streamlit Cloud's st.secrets if running there.

    Streamlit Cloud doesn't ship a .env file — it stores secrets in a
    dashboard-managed secrets.toml. This fallback lets the same code work
    locally (with .env) and in deployment (with st.secrets) without changes.
    """
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st  # only available in Streamlit runtime
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


# --- LLM / embeddings ---
OPENAI_API_KEY = _resolve_secret("OPENAI_API_KEY", "")
LLM_MODEL = _resolve_secret("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = _resolve_secret("EMBEDDING_MODEL", "text-embedding-3-small")
# Streamlit-Cloud-only: an optional shared passcode that gates the chat tab.
# Leave unset for an open URL.
APP_PASSCODE = _resolve_secret("APP_PASSCODE", "")

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
HP_NOTES_DIR = DATA_DIR / "hp_notes"
MEDICAL_CORPUS_DIR = DATA_DIR / "medical_corpus"
RECORDS_XLSX = DATA_DIR / "records.xlsx"

DB_PATH = PROJECT_ROOT / os.getenv("DB_PATH", "healthcare.db")
SCHEMA_SQL = PROJECT_ROOT / "db" / "schema.sql"

INDICES_DIR = PROJECT_ROOT / "indices"
FAISS_HP_PATH = PROJECT_ROOT / os.getenv("FAISS_HP_PATH", "indices/hp_index")
FAISS_MEDICAL_PATH = PROJECT_ROOT / os.getenv("FAISS_MEDICAL_PATH", "indices/medical_index")

# --- Agent tuning ---
HP_TOP_K = 4
MEDICAL_TOP_K = 5
DEFAULT_SLOT_WINDOW_DAYS = 7


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OPENAI_API_KEY
