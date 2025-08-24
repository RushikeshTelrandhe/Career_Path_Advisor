HEAD
# Career_Path_Advisor

# 🧭 AI-Driven Career Path Advisor (Starter)

Takes a résumé, extracts skills using **BERT embeddings** (Sentence-Transformers),
matches to role profiles with **scikit-learn** nearest neighbors + coverage re-rank,
and serves a **Streamlit** UI. Ready to deploy on **Hugging Face Spaces**.

## Quickstart (local)

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

Then open the URL printed by Streamlit (usually http://localhost:8501).

## How it works

- Upload a résumé (PDF/DOCX/TXT).
- The app extracts skills via hybrid matching:
  - Substring match vs a skills catalog
  - Semantic match: sentence embeddings vs skill embeddings (cosine sim)
- Roles are represented as the mean of required skill embeddings.
- Recommendations are a weighted blend of cosine similarity (resume→role) and skill coverage.
- Missing skills map to curated learning resources.

## Hugging Face Spaces

1. Create a new Space → **Streamlit** template.
2. Upload these files: `app.py`, `requirements.txt`, `roles.csv`, `skills.csv`, `resources.csv`, `sample_resume.txt`.
3. Commit and **Build**. First load will download the embedding model (cache persists).

## Customize

- Expand `roles.csv`/`skills.csv`/`resources.csv` with your domain data.
- Swap the model in the Select box (e.g., `all-mpnet-base-v2`). Larger models = better quality but slower.
- Add more fields (salary, location, seniority) and re-rank accordingly.
- Add feedback storage to iteratively improve recommendations.

## Notes

- For production, consider a robust parser, PII scrubbing, and skill taxonomies (e.g., ESCO/O*NET — check licenses).
- To speed cold-starts in Spaces, you can pre-download models in a setup script or pin smaller models.
 ca80c08 (Initial commit - Career Path Advisor project)

## Check Depolyed App - https://careerpathadvisorgit-8tcabudtqsdqrbia7wjtks.streamlit.app/
