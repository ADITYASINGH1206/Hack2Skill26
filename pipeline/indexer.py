import json
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from rank import (
    SEMANTIC_JD_QUERY,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
)

ARTIFACTS_DIR = "artifacts"
DENSE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss_index_v2.bin")
BM25_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "bm25_model_v2.pkl")
CANDIDATES_META_PATH = os.path.join(ARTIFACTS_DIR, "candidates_meta.pkl")
JD_VECTOR_PATH = os.path.join(ARTIFACTS_DIR, "jd_vector.npy")

def extract_text(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]

    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        if name:
            parts.append(f"{name} ({prof})" if prof else name)

    for role in candidate.get("career_history", [])[:3]:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")
        if title:
            parts.append(f"{title} at {company}" if company else title)
        if desc:
            parts.append(desc)

    return " ".join(filter(None, parts))



def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def extract_lean_meta(candidate: dict) -> dict:

    profile = candidate.get("profile", {})
    return {
        "candidate_id": candidate["candidate_id"],
        "career_history": candidate.get("career_history", []),
        "skills": candidate.get("skills", []),
        "profile": {
            "current_title": profile.get("current_title", ""),
            "years_of_experience": profile.get("years_of_experience", 0),
            "location": profile.get("location", ""),
            "country": profile.get("country", ""),
            "headline": profile.get("headline", ""),
            "summary": profile.get("summary", "")[:200],
            "current_company": profile.get("current_company", ""),
            "current_industry": profile.get("current_industry", ""),
        },
        "redrob_signals": candidate.get("redrob_signals", {}),
    }



def main(input_file: str = "clean_pool.jsonl", output_dir: str = "."):
    candidates = []
    texts = []

    if not os.path.exists(input_file):
        fallback = os.path.join(
            "hackathon_rules_conditions",
            "[PUB] India_runs_data_and_ai_challenge",
            "[PUB] India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "sample_candidates.json",
        )
        print(f"[WARN] {input_file} not found -- falling back to {fallback}")
        input_file = fallback

        with open(input_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for c in raw:
                candidates.append(c)
                texts.append(c.get("text", extract_text(c)))
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    candidates.append(c)
                    texts.append(c.get("text", extract_text(c)))

    print(f"[OK] Loaded {len(candidates)} candidates")

    # Generate embeddings
    print(f"Loading model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding candidate texts ...")
    candidate_vectors = model.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    print("Encoding JD semantic query ...")
    jd_vector = model.encode(
        [SEMANTIC_JD_QUERY], normalize_embeddings=True
    )

    candidate_vectors = np.ascontiguousarray(candidate_vectors, dtype=np.float32)
    jd_vector = np.ascontiguousarray(jd_vector, dtype=np.float32)

    # Build FAISS dense index
    print("Building FAISS IndexFlatIP ...")
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(candidate_vectors)

    dense_path = os.path.join(output_dir, DENSE_INDEX_PATH)
    faiss.write_index(index, dense_path)

    jd_vec_path = os.path.join(output_dir, JD_VECTOR_PATH)
    np.save(jd_vec_path, jd_vector)

    # Build BM25 sparse index
    print("Building BM25 index ...")
    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path = os.path.join(output_dir, BM25_INDEX_PATH)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save lean metadata
    print("Saving lean candidate metadata ...")
    lean_meta = [extract_lean_meta(c) for c in candidates]

    meta_path = os.path.join(output_dir, CANDIDATES_META_PATH)
    with open(meta_path, "wb") as f:
        pickle.dump(lean_meta, f)

    # Report artifact sizes
    print("\n-- Artifact Sizes --")
    total = 0
    for name in [DENSE_INDEX_PATH, BM25_INDEX_PATH, CANDIDATES_META_PATH, JD_VECTOR_PATH]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            total += size_mb
            print(f"  {name:30s}  {size_mb:8.1f} MB")
    print(f"  {'TOTAL':30s}  {total:8.1f} MB")
    if total > 5000:
        print("  [WARN] Total exceeds 5 GB disk limit!")
    else:
        print(f"  [OK] Within 5 GB limit ({total / 5000 * 100:.1f}% used)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(input_file=sys.argv[1])
    else:
        main()
