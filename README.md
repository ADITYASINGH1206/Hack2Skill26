# Redrob Hackathon: AI Candidate Ranking Engine

This repository contains our complete ML Ranking Pipeline for the Redrob AI "Intelligent Candidate Discovery & Ranking Challenge."

Our engine is engineered to strictly adhere to the Hackathon Stage 3 sandbox constraints: **16GB RAM limit, CPU-only, under 5 minutes inference, and no external LLM network calls.** 
It mathematically matches the JD requirements while aggressively penalizing honeypots and trap candidates.

---

## 🚀 How to Run the Submission (Stage 3 Reproduction)

The rules mandate that the final ranking step must complete within 5 minutes on a CPU. 

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pre-Computation (Generates Artifacts)
As per Section 10.3 of the rules (*"pre-computation may exceed the 5-minute window"*), we do not push heavy >100MB binary artifacts to GitHub. Instead, you can deterministically generate them using our provided pipeline scripts.

First, run the data cleaner (hard filters honeypots and non-technical domains):
```bash
python pipeline/ingest_and_filter.py
```
Next, run the offline embedder to generate the FAISS and BM25 indexes. *(This step downloads the BGE model and may take ~15-20 minutes on a CPU depending on your machine, but runs outside the 5-minute limit)*:
```bash
python pipeline/indexer.py clean_pool.jsonl
```
This will create the required indexes in the `artifacts/` folder.

### 3. Run the Ranking Engine (The 5-Minute Window)
Simply execute `rank.py`. This script instantly loads the pre-computed artifacts, applies our 8-layer heuristic scoring algorithm across the candidate pool, generates the deterministic reasoning, and produces the final CSV.

```bash
python rank.py --candidates clean_pool.jsonl --out submission_final.csv
```
**Performance:** This takes ~11 seconds on a standard CPU, using less than 4% of the 5-minute limit!

---

## 🛠️ Architecture

Our hybrid pipeline features three distinct stages to maximize accuracy while beating the clock constraint. (All scripts are available in the `pipeline/` directory).

### Phase 1: Data Ingestion & Hard Filter (`pipeline/ingest_and_filter.py`)
- An $O(1)$ memory streaming pre-processor that reads `candidates.jsonl`.
- **Hard Drops:** Instantly filters out mathematical impossibilities (honeypots) and non-technical titles (keyword stuffer traps) *before* embedding generation. 
- Guarantees a 0% honeypot rate.

### Phase 2: Offline Pre-computation (`pipeline/indexer.py`)
- Reads the cleaned dataset and runs the `BAAI/bge-base-en-v1.5` neural embedding model offline.
- Builds a FAISS inner-product dense index (`faiss_index_v2.bin`) and a BM25 sparse index (`bm25_model_v2.pkl`).

### Phase 3: Online Inference (`rank.py`)
- Our blazing-fast, CPU-optimized heuristic engine. 
- Scores candidates using a fusion of Semantic similarity (65%) and Keyword BM25 (35%).
- **5-Layer Soft Penalties:** Applies dynamic runtime penalties and boosts for nuanced JD logic (Pure Researcher drop, Consulting penalty, Notice period tiering, Job Hopper penalty, and high-demand Behavioral boosts).
- Dynamically generates highly-specific deterministic reasoning text for each Top 100 candidate (with no LLM hallucinations), and exports the exact submission CSV format.


