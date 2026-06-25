# Redrob Hackathon: AI Candidate Ranking Engine

This repository contains the complete ML Ranking Pipeline (Role 2) and the Deterministic Reasoning Engine (Role 3) for the Redrob AI "Intelligent Candidate Discovery & Ranking Challenge."

Our engine is engineered to strictly adhere to the Hackathon Stage 3 sandbox constraints: **16GB RAM limit, CPU-only, under 5 minutes inference, and no external LLM network calls.**

---

## 🚀 How to Run the Submission (Stage 3 Reproduction)

The rules mandate that the final ranking step must complete within 5 minutes on a CPU. To achieve this, we decoupled the heavy ML embeddings from the lightning-fast heuristic ranking.

### 1. Ensure Artifacts are Present
The heavy machine learning computations are performed completely offline. The resulting embeddings and indexes are stored in the `artifacts/` folder. 
*(If the folder is empty, see the "Regenerating Artifacts" section below).*

### 2. Install Dependencies
We recommend creating a virtual environment before installing the requirements to avoid conflicts:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Ranking Engine
Simply execute `rank.py`. This script instantly loads the pre-computed artifacts, applies our 8-layer heuristic scoring algorithm across the candidate pool, generates the deterministic reasoning, and produces the final CSV.

```bash
python rank.py --candidates clean_pool.jsonl --out submission.csv
```
**Performance:** This takes ~15 seconds on a standard CPU, using less than 5% of the 5-minute limit!

### 3. Check the Output
The script will output `submission.csv` to the root folder. It includes an automatic internal validation pass that ensures it perfectly matches the hackathon's Stage 1 auto-checker constraints (100 rows, unique non-increasing scores, unique reasoning).

---

## 🛠️ Architecture

Our hybrid pipeline features three distinct stages to maximize accuracy while beating the clock constraint:

### Phase 1: Data Ingestion & Hard Filter (`src/data/cleaner.py`)
- An $O(1)$ memory streaming pre-processor that reads `candidates.jsonl`.
- **Hard Drops:** Instantly filters out mathematical impossibilities (honeypots) and non-technical titles (keyword stuffer traps) *before* embedding generation. 
- Reduces dataset size by ~57%, saving massive compute time and guaranteeing a 0% honeypot rate.

### Phase 2: Offline Pre-computation (Indexer)
- **`src/ml/indexer.py`** (or Colab Notebook): Reads `clean_pool.jsonl`, runs the `BAAI/bge-base-en-v1.5` neural embedding model, and builds a FAISS inner-product index and a BM25Okapi sparse index.

### Phase 3: Online Inference (`rank.py` & `src/ml/scorer.py`)
- **`src/ml/scorer.py`**: The blazing-fast heuristic engine. Scores candidates using a fusion of Semantic similarity (65%) and Keyword BM25 (35%).
- **8-Layer Soft Penalties:** Applies dynamic runtime penalties and boosts for nuanced JD logic (Pure Researcher drop, Consulting penalty, Notice period tiering, Title chaser penalty, and high-demand Behavioral boosts).
- **`rank.py`**: The final executable. Wraps the scorer, dynamically generates highly-specific deterministic reasoning text for each Top 100 candidate (with no LLM hallucinations), and exports the CSV.

---

## 📦 Regenerating Artifacts (Optional)

If the judges choose to verify the codebase against a new hidden dataset (or if you need to regenerate the artifacts), you must first clean the data and generate the new embeddings before running the 5-minute `rank.py` test. 

As per Section 10.3 of the rules: *"pre-computation may exceed the 5-minute window"*.

**Step 1: Clean the Data (Local)**
```bash
python -m src.data.cleaner candidates.jsonl clean_pool.jsonl
```

**Step 2: Generate Embeddings**

*Option A: Google Colab (Recommended Sandbox - ~2 mins)*
1. Open Google Colab and upload `notebooks/role2_pipeline.ipynb`
2. Change Runtime to **T4 GPU**
3. Upload `clean_pool.jsonl` to the Colab environment when prompted
4. Run all cells to process the clean pool in ~2 minutes.
5. Download the 4 generated files and drop them into the `artifacts/` folder of this repo.

*Option B: Local CPU (~25 mins)*
```bash
python -m src.ml.indexer clean_pool.jsonl
```
