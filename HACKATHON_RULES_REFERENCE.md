# 🏆 Redrob Hackathon — Complete Rules & Conditions Reference

> **Challenge Name:** Intelligent Candidate Discovery & Ranking Challenge  
> **Organizer:** Redrob AI  
> **This document is a synthesized reference of ALL rules, conditions, data schemas, scoring, and submission requirements.**

---

## 📋 Table of Contents

1. [Challenge Overview](#1-challenge-overview)
2. [The Job Description (What You're Ranking For)](#2-the-job-description)
3. [Dataset & Schema](#3-dataset--schema)
4. [Redrob Behavioral Signals (23 Signals)](#4-redrob-behavioral-signals)
5. [Trap Candidates & Honeypots](#5-trap-candidates--honeypots)
6. [Compute Constraints](#6-compute-constraints)
7. [Submission CSV Format](#7-submission-csv-format)
8. [Scoring & Evaluation Metrics](#8-scoring--evaluation-metrics)
9. [5-Stage Evaluation Pipeline](#9-5-stage-evaluation-pipeline)
10. [Reasoning Column Requirements](#10-reasoning-column-requirements)
11. [Final Submission Package (Full Checklist)](#11-final-submission-package)
12. [Portal Metadata Fields](#12-portal-metadata-fields)
13. [Sandbox / Demo Link Requirements](#13-sandbox--demo-link-requirements)
14. [submission_metadata.yaml Template](#14-submission_metadatayaml-template)
15. [Common Rejection Reasons](#15-common-rejection-reasons)
16. [Key Strategic Insights](#16-key-strategic-insights)
17. [Phased Implementation Roadmap](#17-phased-implementation-roadmap)
18. [File Inventory](#18-file-inventory)

---

## 1. Challenge Overview

**Goal:** Given a pool of **100,000 candidates** (`candidates.jsonl`), produce a CSV ranking the **top 100 candidates** best-fit for a specific Job Description (Senior AI Engineer at Redrob AI).

- **Rank 1** = best fit; **Rank 100** = 100th best fit
- You do NOT rank candidates 101+
- **No live leaderboard** — scores revealed only after submissions close
- **Max 3 submissions** — your **last valid submission** counts
- **AI tools are allowed** — declare them honestly; AI-assisted work with real engineering succeeds, AI-only submissions fail at Stages 3-5

---

## 2. The Job Description

**Role:** Senior AI Engineer — Founding Team  
**Company:** Redrob AI (Series A AI-native talent intelligence platform)  
**Location:** Pune/Noida, India (Hybrid) | Open to relocation from Tier-1 Indian cities  
**Experience:** 5–9 years

### What the JD ACTUALLY Wants (Key Signals)

| Category | What Matters |
|---|---|
| **Core Mandate** | Own the intelligence layer — ranking, retrieval, and matching systems |
| **Must-Have Skills** | Production embeddings-based retrieval (sentence-transformers, BGE, E5), vector DBs/hybrid search (FAISS, Pinecone, Weaviate, etc.), strong Python, evaluation frameworks (NDCG, MRR, MAP, A/B testing) |
| **Nice-to-Have** | LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost/neural), HR-tech exposure, distributed systems, open-source contributions |
| **Experience Type** | Applied ML/AI at **product companies** (NOT pure services/consulting) |
| **Ideal Profile** | 6-8 yrs total, 4-5 in applied ML/AI, shipped end-to-end ranking/search/recommendation system |
| **Location Pref** | Pune/Noida preferred; Hyderabad, Mumbai, Delhi NCR acceptable |
| **Notice Period** | Sub-30 days preferred; can buy out up to 30 days; 30+ still in scope but higher bar |

### Explicit DISQUALIFIERS from the JD

- ❌ **Pure researchers** (academic labs only, no production deployments)
- ❌ **Recent LangChain-only AI experience** (<12 months, no pre-LLM ML production)
- ❌ **Senior engineers who haven't written production code in 18+ months**
- ❌ **Title-chasers** switching companies every 1.5 years
- ❌ **Framework enthusiasts** (LangChain tutorials, no systems thinking)
- ❌ **Entire career at consulting firms** (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) — unless prior product-company experience exists
- ❌ **Primary expertise in CV/speech/robotics** without significant NLP/IR exposure
- ❌ **5+ years on closed-source proprietary systems** without external validation

### 🚨 Critical Hackathon-Specific Note from JD

> *"The right answer is NOT finding candidates whose skills section contains the most AI keywords. That's a trap we've explicitly built into the dataset."*

- A "Tier 5" candidate may NOT use words like "RAG" or "Pinecone" but has career history showing they built recommendation systems at product companies → **THEY ARE A FIT**
- A candidate with all AI keywords but title "Marketing Manager" → **NOT A FIT**, keyword stuffer trap
- Behavioral signals matter — a perfect-on-paper candidate inactive for 6 months with 5% recruiter response rate is **not actually available**

---

## 3. Dataset & Schema

### Dataset Files

| File | Size | Description |
|---|---|---|
| `candidates.jsonl` | ~487 MB | Full 100,000 candidate pool (one JSON per line) |
| `sample_candidates.json` | ~300 KB | Small sample for development/testing |
| `candidate_schema.json` | ~9 KB | JSON Schema for candidate profiles |

### Candidate Profile Structure

```
candidate_id          (string: CAND_XXXXXXX, 7 digits)
├── profile
│   ├── anonymized_name
│   ├── headline
│   ├── summary
│   ├── location / country
│   ├── years_of_experience (0-50)
│   ├── current_title
│   ├── current_company
│   ├── current_company_size (enum: "1-10" to "10001+")
│   └── current_industry
├── career_history[]  (1-10 entries)
│   ├── company / title
│   ├── start_date / end_date / duration_months
│   ├── is_current (bool)
│   ├── industry / company_size
│   └── description
├── education[]  (0-5 entries)
│   ├── institution / degree / field_of_study
│   ├── start_year / end_year
│   ├── grade
│   └── tier (enum: tier_1 to tier_4, unknown)
├── skills[]
│   ├── name
│   ├── proficiency (enum: beginner/intermediate/advanced/expert)
│   ├── endorsements (integer)
│   └── duration_months (integer)
├── certifications[]
│   ├── name / issuer / year
├── languages[]
│   ├── language / proficiency (basic/conversational/professional/native)
└── redrob_signals  (23 behavioral signals — see section 4)
```

---

## 4. Redrob Behavioral Signals

**23 signals** per candidate — these are behavioral data points from the Redrob platform, used as a **multiplier/modifier** on top of skill-match scoring.

| # | Signal | Type/Range | What It Measures |
|---|---|---|---|
| 1 | `profile_completeness_score` | 0-100 | % of profile filled in |
| 2 | `signup_date` | date | When they signed up |
| 3 | `last_active_date` | date | When they last logged in |
| 4 | `open_to_work_flag` | bool | Marked themselves available |
| 5 | `profile_views_received_30d` | int ≥ 0 | Recruiter views in last 30 days |
| 6 | `applications_submitted_30d` | int ≥ 0 | Roles applied to recently |
| 7 | `recruiter_response_rate` | 0.0-1.0 | Fraction of recruiter messages replied to |
| 8 | `avg_response_time_hours` | number ≥ 0 | Median response time to recruiter message |
| 9 | `skill_assessment_scores` | dict[str, 0-100] | Per-skill platform assessment scores |
| 10 | `connection_count` | int ≥ 0 | Number of Redrob connections |
| 11 | `endorsements_received` | int ≥ 0 | Total skill endorsements |
| 12 | `notice_period_days` | 0-180 | Stated notice period |
| 13 | `expected_salary_range_inr_lpa` | {min, max} | Salary expectations in INR LPA |
| 14 | `preferred_work_mode` | enum | remote / hybrid / onsite / flexible |
| 15 | `willing_to_relocate` | bool | Will relocate if needed |
| 16 | `github_activity_score` | -1 to 100 | GitHub activity (-1 = no GitHub) |
| 17 | `search_appearance_30d` | int ≥ 0 | Recruiter search appearances |
| 18 | `saved_by_recruiters_30d` | int ≥ 0 | Recruiters who bookmarked them |
| 19 | `interview_completion_rate` | 0.0-1.0 | Fraction of interviews attended |
| 20 | `offer_acceptance_rate` | -1 to 1.0 | Offer acceptance rate (-1 = no history) |
| 21 | `verified_email` | bool | Email verified |
| 22 | `verified_phone` | bool | Phone verified |
| 23 | `linkedin_connected` | bool | LinkedIn connected |

### Key Behavioral Signals for Ranking (per JD & rules)

- **`recruiter_response_rate`** — Penalize low rates (candidate not responsive)
- **`last_active_date`** — Penalize if not recent (candidate inactive/unavailable)
- **`interview_completion_rate`** — Penalize low rates (unreliable)
- **`notice_period_days`** — Prefer sub-30 days; 30+ raises the bar
- **`willing_to_relocate`** — Important for location requirements
- **`open_to_work_flag`** — Signal of availability
- **`github_activity_score`** — Technical engagement signal

---

## 5. Trap Candidates & Honeypots

### 🚨 DISQUALIFICATION RISK: Honeypot Rate > 10% in Top 100

The dataset contains **~80 honeypot candidates** with **subtly impossible profiles**:

- 8 years of experience at a company founded 3 years ago
- "Expert" proficiency in 10 skills with 0 months duration
- Overlapping career timelines
- 0 months of duration on "expert" skills

**Honeypots are forced to relevance tier 0 in the ground truth.**

### Other Trap Types

| Trap Type | Description | How to Detect |
|---|---|---|
| **Keyword Stuffers** | All the right AI keywords but wrong title (e.g., "Marketing Manager" with RAG skills) | Check title vs. skills mismatch |
| **Plain-language Tier 5s** | Great candidates who DON'T use buzzwords but have relevant career history | Look at career history, not just keywords |
| **Behavioral Twins** | Two candidates with similar profiles but vastly different behavioral signals | Use behavioral signals as tiebreaker |
| **Honeypots** | Impossible profiles (overlapping timelines, impossible experience) | Validate career timeline logic |

---

## 6. Compute Constraints

| Constraint | Limit |
|---|---|
| **Total runtime** | ≤ **5 minutes** wall-clock |
| **Memory** | ≤ **16 GB** RAM |
| **Compute** | **CPU only** — no GPU during ranking |
| **Network** | **OFF** — no external API calls (no OpenAI, Anthropic, Cohere, Gemini, etc.) |
| **Disk** | ≤ **5 GB** intermediate state |

### What This Means

- ❌ Cannot call hosted LLM APIs during ranking
- ❌ Cannot use GPUs
- ❌ Cannot exceed runtime/memory limits
- ✅ Pre-computation (embeddings, indexes) is allowed BEFORE the 5-min ranking window
- ✅ Lightweight local models are OK if they fit within constraints
- At **Stage 3**, top-N submissions are reproduced inside a **sandboxed Docker container** matching these constraints exactly

---

## 7. Submission CSV Format

### Filename
`<participant_id>.csv` (e.g., `team_xxx.csv`)

### Encoding
UTF-8

### Required Columns (in this exact order)

| Column | Type | Required? | Description |
|---|---|---|---|
| `candidate_id` | string | ✅ Yes | `CAND_XXXXXXX` from candidates.jsonl |
| `rank` | int (1-100) | ✅ Yes | Each integer 1-100 used exactly once |
| `score` | float | ✅ Yes | Monotonically non-increasing as rank increases |
| `reasoning` | string | ⚠ Strongly recommended | 1-2 sentence justification per candidate |

### CSV Rules

- **Exactly 101 rows**: 1 header + 100 data rows
- Each rank 1-100 appears **exactly once**
- Each `candidate_id` appears **exactly once**
- Every `candidate_id` **must exist** in `candidates.jsonl`
- **Score is non-increasing**: score[rank 1] ≥ score[rank 2] ≥ ... ≥ score[rank 100]
- **Tie-breaking**: Same score → break deterministically using secondary signal or `candidate_id` ascending
- **Format**: `CAND_XXXXXXX` (7 digits, regex: `^CAND_[0-9]{7}$`)

### Example

```csv
candidate_id,rank,score,reasoning
CAND_0042871,1,0.987,"Senior AI Engineer with 7 years building RAG systems at product companies; strong recent engagement and Bangalore-based."
CAND_0019884,2,0.973,"6 years applied ML; previously shipped vector search at scale; matches the 'product over research' profile in the JD."
...
CAND_0007729,100,0.412,"Adjacent skills only — likely below cutoff but included as final filler given experience and engagement signals."
```

---

## 8. Scoring & Evaluation Metrics

### Composite Formula

```
Final Composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

| Metric | Weight | What It Measures |
|---|---|---|
| **NDCG@10** | **0.50** | Quality of your top-10 picks (MOST IMPORTANT) |
| **NDCG@50** | **0.30** | Quality of your top-50 picks |
| **MAP** | **0.15** | Precision across all relevance levels |
| **P@10** | **0.05** | Fraction of top-10 that are "relevant" (tier 3+) |

### Tiebreaking (between submissions with identical composites)

1. Higher **P@5** wins
2. Higher **P@10** wins
3. **Earlier submission timestamp** wins

### Key Insight

> **50% of your score comes from getting the top 10 right.** Invest heavily in ensuring your top 10 are genuinely the best candidates.

---

## 9. 5-Stage Evaluation Pipeline

| Stage | What Happens | What Gets You Eliminated |
|---|---|---|
| **1. Format Validation** | Auto-validator runs on every submission | Any spec violation (Section 3 rules) |
| **2. Scoring** | Composite computed on full hidden ground truth after submissions close | Score below cutoff for Stage 3 advancement |
| **3. Code Reproduction + Honeypot Check** | Top-N: full code repo requested, ranking step reproduced in sandbox (5min, 16GB, no GPU, no network). Honeypot rate computed. | Cannot reproduce within limits; **honeypot rate >10%**; missing/fabricated code repo |
| **4. Manual Review** | Reasoning quality (6 checks). Methodology coherence. Git history authenticity. Code quality. | Failed reasoning checks; flat git history (single dump, no iteration); codebase is all LLM API calls |
| **5. Defend-Your-Work Interview** | Top-X finalists: 30-min video call with Redrob engineering. Walk through architecture, defend design choices. | Cannot explain architecture; contradicts code; clearly didn't build it |

---

## 10. Reasoning Column Requirements

At **Stage 4**, 10 random rows are sampled and checked against **6 criteria**:

| Check | What They Look For |
|---|---|
| **Specific facts** | References specific profile facts (years of experience, title, named skills, signal values) |
| **JD connection** | Connects to specific JD requirements, not generic praise |
| **Honest concerns** | Acknowledges obvious gaps or concerns |
| **No hallucination** | Every claim corresponds to something in the candidate's profile |
| **Variation** | 10 sampled reasonings are substantively different (not templated) |
| **Rank consistency** | Reasoning tone matches rank (rank-5 shouldn't have critical tone; rank-95 shouldn't have glowing tone) |

### What Gets Penalized

- ❌ Empty reasoning
- ❌ All-identical reasoning strings
- ❌ Templated reasoning that just inserts candidate name
- ❌ Mentions skills NOT in the candidate's profile (hallucination)
- ❌ Reasoning that contradicts the rank

### What Scores Well

- ✅ Plain-language reasoning showing you understood the profile
- ✅ Specific and honest
- ✅ References actual data from the profile
- ✅ Connects to JD requirements

---

## 11. Final Submission Package

### Three Required Parts

#### 11.1 — The CSV File
- Top-100 ranking per Sections 2 and 3 (see above)

#### 11.2 — Portal Metadata
- Collected at upload time (see Section 12)

#### 11.3 — Code Repository (GitHub)

Your GitHub repo **must include**:

- [ ] `README.md` with setup instructions and **exact reproduction command**
- [ ] Full source code that produced the CSV (no hidden steps, no manual edits)
- [ ] Pre-computed artifacts (embeddings, indexes, model weights) OR a script that produces them
- [ ] `requirements.txt` / `pyproject.toml` with all dependencies and versions
- [ ] `submission_metadata.yaml` at repo root (mirror of portal metadata)

**Reproduction command example:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

> **Note:** Pre-computation (embedding generation) may exceed 5 minutes, but the **ranking step that produces the CSV must complete within 5 minutes**.

#### 11.4 — AI Tools Declaration
- Honest declaration — not penalized
- If interview answers contradict declaration → strong negative signal

#### 11.5 — Sandbox / Demo Link (see Section 13)

---

## 12. Portal Metadata Fields

| Field | Required? | Notes |
|---|---|---|
| Team name | ✅ Yes | Used in leaderboard and results |
| Primary contact name | ✅ Yes | Point of contact |
| Primary contact email | ✅ Yes | All organizer communication |
| Primary contact phone | ✅ Yes | Top-N / Top-X outreach |
| GitHub repository URL | ✅ Yes | Must be reachable. Private OK if you grant access at Stage 3 |
| Sandbox / demo link | ✅ Yes | Working hosted environment (see Section 13) |
| AI tools declared | ✅ Yes | Multi-select: Claude / ChatGPT / Copilot / Cursor / Gemini / Other / None |
| Compute environment summary | ✅ Yes | One line (e.g., "MacBook Pro M2, 16GB RAM, Python 3.11") |
| Team member list | ✅ Yes | Name + email for each member |
| Methodology summary | Optional | ≤200 words. Strongly recommended for Stage 4 review |

---

## 13. Sandbox / Demo Link Requirements

### Acceptable Platforms

- HuggingFace Spaces (free tier OK)
- Streamlit Cloud (free tier OK)
- Replit (public repl)
- Google Colab (end-to-end notebook)
- Docker (`docker pull` + `docker run` with public registry image)
- Binder (runnable Jupyter notebook)

### Sandbox Must:

- ✅ Accept a small candidate sample (≤100 candidates) as input
- ✅ Run ranking system end-to-end and produce ranked CSV
- ✅ Complete within compute budget (≤5 min on CPU)
- ✅ Does NOT need to handle full 100K pool

### Why It's Mandatory

The sandbox is a **fast, low-stakes sanity check** before full Stage 3 reproduction. Submissions without a working sandbox link are **flagged at Stage 1**.

---

## 14. submission_metadata.yaml Template

The template is provided as `submission_metadata_template.yaml`. Key sections:

```yaml
team_name: "your-team-name-here"
primary_contact:
  name / email / phone
team_members:
  - name / email / role
github_repo: "https://github.com/..."
sandbox_link: "https://huggingface.co/spaces/..."
reproduce_command: "python rank.py --candidates ./candidates.jsonl --out ./submission.csv"
compute:
  platform / cpu_cores / ram_gb / python_version / os
  uses_gpu_for_inference: false        # MUST be false
  has_network_during_ranking: false    # MUST be false
ai_tools_used: [...]
ai_usage_summary: "..."
methodology_summary: "..."
declarations:
  read_submission_spec: true
  code_is_original_work: true
  no_collusion: true
  honeypot_check_done: false           # Set true if you checked
  reproduction_tested: true
```

---

## 15. Common Rejection Reasons

These are auto-rejected by the server-side validator:

| ❌ Rejection | Why |
|---|---|
| 99 or 101 rows | Must be exactly 100 data rows |
| Ranks starting at 0 | Must start at 1 |
| Duplicate `candidate_id`s | Each ID must appear exactly once |
| `candidate_id` typos | Must exist in `candidates.jsonl` |
| All scores identical | Model isn't differentiating |
| Scores increasing with rank | Rank 1 should have highest score |
| File submitted as `.xlsx` or `.json` | Must be `.csv` |

**Always run `validate_submission.py` locally before uploading!**

```bash
python validate_submission.py <participant_id>.csv
```

---

## 16. Key Strategic Insights

### 🎯 What Wins This Hackathon

1. **Top 10 is everything** — 50% of score is NDCG@10. Get the top 10 right.
2. **Don't chase keywords** — The dataset has keyword-stuffer traps. Look at career history and role descriptions.
3. **Behavioral signals are a multiplier** — Penalize inactive, non-responsive, high-notice-period candidates.
4. **Honeypot detection is survival** — >10% honeypots = disqualification. Validate career timelines.
5. **Reasoning quality matters at Stage 4** — Be specific, honest, fact-based. Reference actual profile data.
6. **Git history matters** — Show real iteration, not a single-commit dump.
7. **Code must be reproducible** — Test your full pipeline on a 16GB CPU-only machine within 5 minutes.
8. **Understand the JD deeply** — The gap between what it says and what it means is the key.

### 🧠 The "Reading Between the Lines" Strategy

The ideal candidate profile (from the JD) is:
- 6-8 years total, 4-5 in applied ML/AI at **product companies**
- Shipped end-to-end ranking/search/recommendation systems
- Strong opinions on retrieval + evaluation + LLM integration
- Located in or willing to relocate to Noida/Pune
- Active on Redrob platform (behavioral signals positive)

---

## 17. Phased Implementation Roadmap

### Phase 1: Data Ingestion & Trap Filter (Offline)
- Stream `candidates.jsonl` line-by-line (O(1) memory)
- Purge honeypots (impossible timelines, 0-month "expert" skills)
- Filter role mismatches (keyword stuffers)
- Filter logistical mismatches (pure researchers, consultants without product experience, excessive notice periods)
- Output: `clean_pool.jsonl`

### Phase 2: Embedding & Indexing (Offline, no time limit)
- Generate embeddings using lightweight model (e.g., `all-MiniLM-L6-v2` or `bge-small-en-v1.5`)
- Build dense vector index (FAISS)
- Build sparse index (BM25) for keyword matching
- Save artifacts (must stay under 5 GB disk limit)

### Phase 3: Scoring & Reasoning Engine
- Hybrid semantic score (vector + BM25 vs. JD)
- Behavioral multiplier (response rate, activity, interview completion)
- Rule-based reasoning generator (deterministic, no LLM, no hallucination)

### Phase 4: rank.py — The 5-Minute Inference Script
- Load pre-built artifacts
- Execute scoring pipeline
- Sort by score descending, top 100
- Tie-break by `candidate_id` ascending
- Output CSV: `<participant_id>.csv` (101 rows: header + 100 data)

### Phase 5: Validation, Packaging & Submission
- Run `validate_submission.py`
- Deploy sandbox (HuggingFace Spaces / Streamlit)
- Prepare GitHub repo (README, requirements.txt, submission_metadata.yaml)
- Submit via portal

---

## 18. File Inventory

| File | Purpose |
|---|---|
| `README.docx` | Bundle overview and getting-started guide |
| `job_description.docx` | The JD you're ranking candidates for |
| `submission_spec.docx` | Rules, scoring, evaluation pipeline |
| `redrob_signals_doc.docx` | Explanation of 23 behavioral signals |
| `candidate_schema.json` | JSON Schema for candidate profiles |
| `candidates.jsonl` | Full 100K candidate pool (~487 MB) |
| `sample_candidates.json` | Small sample for development (~300 KB) |
| `sample_submission.csv` | Format reference (not quality reference) |
| `submission_metadata_template.yaml` | YAML template to fill and include in repo |
| `validate_submission.py` | Local validator — run before submitting |

---

> **Last updated:** 2026-06-24  
> **Source files:** All documents from `hackathon_rules_conditions/[PUB] India_runs_data_and_ai_challenge/` bundle
