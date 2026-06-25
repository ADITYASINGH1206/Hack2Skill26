# Redrob Hackathon: AI Candidate Ranking Engine

This repository contains the complete ML Ranking Pipeline (Role 2) and the Deterministic Reasoning Engine (Role 3) for the Redrob AI "Intelligent Candidate Discovery & Ranking Challenge."

Our engine is engineered to strictly adhere to the Hackathon Stage 3 sandbox constraints: **16GB RAM limit, CPU-only, under 5 minutes inference, and no external LLM network calls.**

---

## 🚀 How to Run the Submission (Stage 3 Reproduction)

The rules mandate that the final ranking step must complete within 5 minutes on a CPU. To achieve this, we decoupled the heavy ML embeddings from the lightning-fast heuristic ranking.

### 1. Ensure Artifacts are Present
The heavy machine learning computations are performed completely offline. The resulting embeddings and indexes are stored in the `artifacts/` folder. 
*(If the folder is empty, see the "Regenerating Artifacts" section below).*

### 2. Run the Ranking Engine
Simply execute `rank.py`. This script instantly loads the pre-computed artifacts, applies our 5-layer heuristic scoring algorithm across all 100,000 candidates, generates the deterministic reasoning, and produces the final CSV.

```bash
python rank.py
```
**Performance:** This takes ~25 seconds on a standard CPU, using less than 10% of the 5-minute limit!

### 3. Check the Output
The script will output `submission.csv` to the root folder. It includes an automatic internal validation pass that ensures it perfectly matches the hackathon's Stage 1 auto-checker constraints (100 rows, unique non-increasing scores, unique reasoning).

---

## 🛠️ Architecture

Our pipeline features two distinct stages to maximize accuracy while beating the clock constraint:

### Phase 1: Offline Pre-computation (Indexer)
- **`src/ml/indexer.py`**: Reads `candidates.jsonl`, runs the `BAAI/bge-base-en-v1.5` neural embedding model, and builds a FAISS inner-product index and a BM25Okapi sparse index.

### Phase 2: Online Inference (Scorer & Ranker)
- **`src/ml/scorer.py`**: The blazing-fast heuristic engine. Scores candidates using a fusion of Semantic similarity (65%) and Keyword BM25 (35%), heavily modulated by 5 defensive multiplier layers (Honeypot detection, Title traps, Career Trajectory, Experience, and Location).
- **`src/ml/config.py`**: The single source of truth for all query weights, JD requirements, and honeypot thresholds.
- **`rank.py`**: The final executable. Wraps the scorer, dynamically generates highly-specific deterministic reasoning text for each Top 100 candidate (with no hallucinations), and exports the CSV.

---

## 📦 Regenerating Artifacts (Optional)

If the judges choose to verify the codebase against a new hidden dataset (or if you need to regenerate the artifacts), you must first generate the new embeddings before running the 5-minute `rank.py` test. 

As per Section 10.3 of the rules: *"pre-computation may exceed the 5-minute window"*.

**Option A: Google Colab (Recommended Sandbox - ~5 mins)**
1. Open Google Colab and upload `notebooks/role2_pipeline.ipynb`
2. Change Runtime to **T4 GPU**
3. Upload `candidates.jsonl` to the Colab environment when prompted
4. Run all cells to process 100,000 records in ~5 minutes.
5. Download the 4 generated files and drop them into the `artifacts/` folder of this repo.

**Option B: Local CPU (~45-60 mins)**
Ensure `candidates.jsonl` is in the project root, then run:
```bash
python -m src.ml.indexer candidates.jsonl
```
