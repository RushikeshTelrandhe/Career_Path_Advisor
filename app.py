
import os
import io
import re
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Dict, Tuple, Set

# ML
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util

# File parsing
from pypdf import PdfReader
import docx

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="AI Career Path Advisor", page_icon="🧭", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads and caches a sentence-transformers model.
    """
    model = SentenceTransformer(model_name)
    return model

def safe_lower(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def read_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    content = ""
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text() or ""
        except Exception as e:
            st.warning(f"PDF parse failed: {e}")
    elif name.endswith(".docx"):
        try:
            d = docx.Document(uploaded_file)
            content = "\n".join([p.text for p in d.paragraphs])
        except Exception as e:
            st.warning(f"DOCX parse failed: {e}")
    else:
        try:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            content = uploaded_file.read().decode("latin-1", errors="ignore")
    return content

def split_sentences(text: str) -> List[str]:
    # naive sentence splitter
    parts = re.split(r"[.\n?!;]+", text)
    parts = [p.strip() for p in parts if len(p.strip().split()) >= 2]
    return parts

def normalize_skill(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def compute_skill_embeddings(skills: List[str], model) -> Dict[str, np.ndarray]:
    embs = model.encode(skills, normalize_embeddings=True, convert_to_numpy=True)
    return {skills[i]: embs[i] for i in range(len(skills))}

def extract_skills_from_resume(
    resume_text: str,
    skills_catalog: List[str],
    model,
    embed_threshold: float = 0.55,
) -> Set[str]:
    """
    Hybrid extractor:
      1) Exact/substring match against catalog (case-insensitive)
      2) Semantic match: if any sentence embedding cos-sim with a skill > threshold
    """
    text_lower = " " + resume_text.lower() + " "
    sentences = split_sentences(resume_text)
    sent_embs = model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True) if sentences else np.zeros((0, 384))
    skill_embs = model.encode(skills_catalog, normalize_embeddings=True, convert_to_numpy=True)

    found = set()
    for i, skill in enumerate(skills_catalog):
        s_norm = skill.lower()
        # Substring/exact boundary-aware
        if f" {s_norm} " in text_lower or s_norm in text_lower:
            found.add(skill)
            continue
        # Semantic: max similarity with any sentence
        if len(sent_embs) > 0:
            sims = np.dot(sent_embs, skill_embs[i])
            if np.max(sims) >= embed_threshold:
                found.add(skill)
    return found

def build_role_embeddings(roles_df: pd.DataFrame, skill2emb: Dict[str, np.ndarray]) -> np.ndarray:
    role_vecs = []
    for _, row in roles_df.iterrows():
        req = [normalize_skill(s) for s in row["required_skills"].split(",")]
        req = [s for s in req if s in skill2emb]
        if not req:
            role_vecs.append(np.zeros_like(next(iter(skill2emb.values()))))
        else:
            role_vecs.append(np.mean([skill2emb[s] for s in req], axis=0))
    return np.vstack(role_vecs)

def role_coverage(user_skills: Set[str], req_skills: List[str]) -> float:
    req = set([normalize_skill(s) for s in req_skills])
    inter = len(req.intersection(user_skills))
    if len(req) == 0: return 0.0
    return inter / len(req)

def recommend_roles(
    user_skills: Set[str],
    roles_df: pd.DataFrame,
    skill2emb: Dict[str, np.ndarray],
    role_vecs: np.ndarray,
    top_k: int = 5,
    weight_sim: float = 0.6,
    weight_cov: float = 0.4,
):
    # Build user vector
    have_vecs = [skill2emb[s] for s in user_skills if s in skill2emb]
    if not have_vecs:
        user_vec = np.zeros(role_vecs.shape[1])
    else:
        user_vec = np.mean(have_vecs, axis=0)

    # Nearest neighbors (cosine) for initial candidates
    nbrs = NearestNeighbors(n_neighbors=min(top_k*3, len(roles_df)), metric="cosine")
    nbrs.fit(role_vecs)
    dists, idxs = nbrs.kneighbors([user_vec])
    idxs = idxs[0].tolist()
    dists = dists[0].tolist()

    candidates = []
    for i, dist in zip(idxs, dists):
        row = roles_df.iloc[i]
        req = [normalize_skill(s) for s in row["required_skills"].split(",")]
        cov = role_coverage(user_skills, req)
        # cosine distance -> similarity
        sim = 1.0 - float(dist)
        score = weight_sim * sim + weight_cov * cov
        candidates.append((i, sim, cov, score))

    # Re-rank and take top_k
    candidates.sort(key=lambda x: x[3], reverse=True)
    results = []
    for i, sim, cov, score in candidates[:top_k]:
        row = roles_df.iloc[i]
        req = [normalize_skill(s) for s in row["required_skills"].split(",")]
        missing = [s for s in req if s not in user_skills]
        extra = [s for s in user_skills if s not in req]
        results.append({
            "role_id": row["role_id"],
            "role_name": row["role_name"],
            "similarity": round(sim, 3),
            "coverage": round(cov, 3),
            "score": round(score, 3),
            "required_skills": req,
            "missing_skills": missing,
            "extra_skills": extra
        })
    return results

def plan_learning(missing_skills: List[str], resources_df: pd.DataFrame, max_per_skill: int = 2):
    plan = []
    for s in missing_skills:
        recs = resources_df[resources_df["skill"].str.lower() == s.lower()].head(max_per_skill)
        for _, r in recs.iterrows():
            plan.append({
                "skill": s,
                "title": r["title"],
                "provider": r["provider"],
                "level": r["level"],
                "url": r["url"],
            })
    return plan

# -----------------------------
# App
# -----------------------------
st.title("🧭 AI-Driven Career Path Advisor")
st.caption("Upload your résumé, get role recommendations, identify skill gaps, and follow a learning plan.")

colA, colB = st.columns([2, 1])

with colA:
    uploaded = st.file_uploader("Upload résumé (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
    sample = st.checkbox("Use sample résumé", value=False)
with colB:
    model_name = st.selectbox("Embedding model", [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2"
    ])
    topk = st.slider("Top-K roles", 1, 10, 5)

# Load datasets
@st.cache_resource(show_spinner=False)
def load_data():
    roles = pd.read_csv("roles.csv")
    skills = pd.read_csv("skills.csv")
    resources = pd.read_csv("resources.csv")
    skills["skill"] = skills["skill"].apply(normalize_skill)
    # catalog list
    catalog = skills["skill"].tolist()
    return roles, skills, resources, catalog

roles_df, skills_df, resources_df, skills_catalog = load_data()

if uploaded or sample:
    if sample:
        resume_text = open("sample_resume.txt", "r", encoding="utf-8").read()
    else:
        resume_text = read_text_from_file(uploaded)

    st.subheader("Résumé Preview")
    st.text_area("Extracted Text", resume_text[:5000], height=200)

    with st.spinner("Loading embedder & extracting skills..."):
        model = load_embedder(model_name)
        found_skills = extract_skills_from_resume(resume_text, skills_catalog, model)

    st.subheader("Extracted Skills")
    if found_skills:
        st.success(f"Detected {len(found_skills)} skills")
        st.write(", ".join(sorted(found_skills)))
    else:
        st.warning("No skills detected from the provided résumé.")

    # Role embeddings
    with st.spinner("Building role index..."):
        skill2emb = compute_skill_embeddings(skills_catalog, model)
        role_vecs = build_role_embeddings(roles_df, skill2emb)

    # Recommend
    with st.spinner("Recommending roles..."):
        recs = recommend_roles(found_skills, roles_df, skill2emb, role_vecs, top_k=topk)

    st.subheader("Recommended Roles")
    if recs:
        for r in recs:
            with st.expander(f"{r['role_name']} — score {r['score']} (sim {r['similarity']}, cov {r['coverage']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Required Skills:** " + ", ".join(r["required_skills"]))
                    st.markdown("**You Have:** " + ", ".join(sorted(set(found_skills).intersection(r["required_skills"]))))
                with col2:
                    st.markdown("**Missing Skills:** " + (", ".join(r["missing_skills"]) if r["missing_skills"] else "None 🎉"))
                    if r["missing_skills"]:
                        plan = plan_learning(r["missing_skills"], resources_df, max_per_skill=2)
                        if plan:
                            st.markdown("**Learning Plan (quick start):**")
                            for p in plan:
                                st.write(f"- {p['skill']}: [{p['title']}]({p['url']}) — {p['provider']} · {p['level']}")
    else:
        st.info("No recommendations could be generated.")

    # Download JSON
    result_blob = {
        "extracted_skills": sorted(list(found_skills)),
        "recommendations": recs
    }
    st.download_button(
        label="⬇️ Download Recommendations (JSON)",
        data=json.dumps(result_blob, indent=2),
        file_name="career_recommendations.json",
        mime="application/json"
    )

else:
    st.info("Upload a résumé or use the sample to see recommendations.")
