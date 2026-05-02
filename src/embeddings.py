"""FAISS index build + load helpers.

Two indices are maintained:
  1. HP index — chunks of the patient History & Physical PDFs, metadata
     includes `patient_id` and `patient_name` so queries can be scoped per patient.
  2. Medical index — chunks of the seeded medical info corpus (Medline/WHO).

Run the build from the project root:
    python -m src.embeddings --build
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import (
    EMBEDDING_MODEL,
    FAISS_HP_PATH,
    FAISS_MEDICAL_PATH,
    HP_NOTES_DIR,
    INDICES_DIR,
    MEDICAL_CORPUS_DIR,
    require_openai_key,
)
from .db import get_conn


# ---- Helpers ----------------------------------------------------------------

def get_embedder() -> OpenAIEmbeddings:
    require_openai_key()
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Lightweight YAML-ish frontmatter parser (key: value only, no nesting)."""
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip()
    return meta, m.group(2)


# ---- H&P index --------------------------------------------------------------

def _load_hp_documents() -> list[Document]:
    """Each PDF filename (e.g. `anjali_mehra.pdf`) is matched to a patient by
    case-insensitive name containment. Chunks carry `patient_id` + `patient_name`.
    """
    # Build a name -> id map once.
    with get_conn() as conn:
        pmap = {r["name"].lower(): r["patient_id"]
                for r in conn.execute("SELECT patient_id, name FROM patients").fetchall()}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs: list[Document] = []
    for pdf_path in sorted(HP_NOTES_DIR.glob("*.pdf")):
        raw = _pdf_to_text(pdf_path)
        if not raw.strip():
            continue

        # Filename-based patient match — "anjali_mehra.pdf" → "anjali mehra".
        stem = pdf_path.stem.replace("_", " ").lower()
        patient_id = next((pid for name, pid in pmap.items() if stem in name or name in stem),
                          None)
        patient_name = stem.title()

        for i, chunk in enumerate(splitter.split_text(raw)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path.name,
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "chunk": i,
                },
            ))
    return docs


def build_hp_index() -> FAISS:
    docs = _load_hp_documents()
    if not docs:
        raise RuntimeError(f"No H&P PDFs found under {HP_NOTES_DIR}")
    emb = get_embedder()
    store = FAISS.from_documents(docs, emb)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(FAISS_HP_PATH))
    print(f"  HP index: {len(docs)} chunks -> {FAISS_HP_PATH}")
    return store


def load_hp_index() -> FAISS:
    return FAISS.load_local(str(FAISS_HP_PATH), get_embedder(),
                            allow_dangerous_deserialization=True)


# ---- Medical info index -----------------------------------------------------

def _load_medical_documents() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs: list[Document] = []
    for md_path in sorted(MEDICAL_CORPUS_DIR.glob("*.md")):
        raw = md_path.read_text()
        meta, body = _parse_frontmatter(raw)
        title = meta.get("title", md_path.stem.replace("_", " ").title())
        source_url = meta.get("source_url", "")
        for i, chunk in enumerate(splitter.split_text(body)):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title": title,
                    "source": meta.get("source", "medical_corpus"),
                    "source_url": source_url,
                    "fetched_at": meta.get("fetched_at", ""),
                    "file": md_path.name,
                    "chunk": i,
                },
            ))
    return docs


def build_medical_index() -> FAISS:
    docs = _load_medical_documents()
    if not docs:
        raise RuntimeError(f"No markdown files found under {MEDICAL_CORPUS_DIR}")
    emb = get_embedder()
    store = FAISS.from_documents(docs, emb)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(FAISS_MEDICAL_PATH))
    print(f"  Medical index: {len(docs)} chunks -> {FAISS_MEDICAL_PATH}")
    return store


def load_medical_index() -> FAISS:
    return FAISS.load_local(str(FAISS_MEDICAL_PATH), get_embedder(),
                            allow_dangerous_deserialization=True)


# ---- CLI --------------------------------------------------------------------

def build_all() -> None:
    print("Building FAISS indices...")
    build_hp_index()
    build_medical_index()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build both FAISS indices")
    args = parser.parse_args()
    if args.build:
        build_all()
    else:
        parser.print_help()
