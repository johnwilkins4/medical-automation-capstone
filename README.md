# Agentic Healthcare Assistant

Capstone for the Applied Generative AI Specialisation (Purdue / Simplilearn).
Builds an LLM agent that books appointments, manages patient records, retrieves patient histories, and searches trusted medical sources — all accessible through a Streamlit UI.

Architecture details are in [`01_architecture.md`](../01_architecture.md) (in the parent outputs folder).

## Requirements

- Python 3.10+
- An OpenAI API key (`OPENAI_API_KEY`). Default models: `gpt-4o-mini` + `text-embedding-3-small`.

## Setup

```bash
cd medical-automation-capstone
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then edit .env to add your OPENAI_API_KEY

# 1. Build the SQLite DB (patients, doctors, schedule)
python -m db.seed

# 2. Build the two FAISS indices (H&P notes + medical corpus)
python -m src.embeddings --build

# 3. Run the app
streamlit run streamlit_app.py
```

## Deploying to Streamlit Community Cloud

The app reads `OPENAI_API_KEY` from `.env` locally and from `st.secrets` in
the cloud, so no code changes are needed for deployment.

1. Push this repo to GitHub.
2. Sign into https://share.streamlit.io with the same GitHub account.
3. Click **New app** and point it at this repo, branch `main`, file
   `streamlit_app.py`.
4. Open the app's **Settings → Secrets** and paste:

   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   APP_PASSCODE   = "your-shared-passcode"   # optional gate
   ```

5. Save. The app boots in a few minutes and is reachable at the URL
   Streamlit assigns. If `APP_PASSCODE` is set, anyone visiting the URL is
   prompted for it before the chat loads.

## Project layout

```
medical-automation-capstone/
├── data/
│   ├── records.xlsx                 patient seed file
│   ├── hp_notes/                    History & Physical PDFs (indexed into FAISS)
│   └── medical_corpus/              Medline/WHO-derived disease pages (indexed into FAISS)
├── db/
│   ├── schema.sql
│   └── seed.py                      builds healthcare.db + schedules
├── src/
│   ├── config.py                    env + paths
│   ├── db.py                        SQLite helpers + agent-run logger
│   ├── embeddings.py                FAISS index build/load
│   ├── prompts.py                   planner / summariser / composer prompts
│   ├── memory.py                    long-term per-patient summary memory
│   ├── agent.py                     LangGraph planner-executor
│   └── tools/
│       ├── patients.py              lookup_patient, add_or_update_record
│       ├── history.py               get_patient_history
│       ├── scheduling.py            list_doctor_slots, book_appointment
│       └── medical_info.py          medical_info_search
├── eval/
│   ├── test_set.jsonl               hand-labeled queries
│   └── run_eval.py                  QAEvalChain + tool metrics
├── notebooks/
│   └── demo.ipynb                   flagship CKD scenario walkthrough
├── streamlit_app.py                 Streamlit entry point
├── healthcare.db                    created by db/seed.py
├── indices/                         created by src/embeddings.py --build
├── requirements.txt
└── .env.example
```

## Demo scenario

The flagship scenario from the problem statement:

> *"My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?"*

A synthetic patient "Mr. Harish Kumar" (70M, CKD stage 3b) is seeded into the DB specifically so this scenario runs end-to-end.

## Data notes

- **Patients:** 5 unique rows from `records.xlsx` (duplicate "Rebeca Nagle" rows merged) + 1 synthetic demo patient.
- **Doctors:** 10 synthetic doctors across 6 specialties (Nephrologist, Cardiologist, Endocrinologist, General Physician, Pulmonologist, Dermatologist).
- **Schedule:** next 14 weekdays × 16 half-hour slots per doctor (9am–5pm).
- **Medical corpus:** 12 markdown files covering CKD, T2DM, hypertension, URI, asthma, hypothyroidism, heart failure, migraine, COVID-19, depression/anxiety, stroke, osteoarthritis. Each carries `source_url` + `fetched_at` frontmatter for citation.

## Evaluation

`python -m eval.run_eval` runs a 15-query labeled test set through the agent and writes per-query correctness (QAEvalChain), per-tool success rate, and latency metrics to the `agent_runs` table. The Streamlit Eval tab surfaces the same numbers.

## Disclaimer

All patient data is synthetic. Medical content is demo material, not clinical guidance.
