import os

# JD QUERIES â€” Dual strategy: semantic (natural language) + BM25 (keywords)

SEMANTIC_JD_QUERY = """
Senior AI Engineer with 6 to 8 years of applied machine learning experience at product companies.
Built and shipped end-to-end ranking systems, search engines, and recommendation systems to real users at scale.
Production experience with embeddings-based retrieval using sentence-transformers, BGE, or E5 models.
Hands-on with vector databases and hybrid search infrastructure including FAISS, Pinecone, Weaviate, Qdrant, Milvus, or Elasticsearch.
Designed evaluation frameworks for ranking quality using NDCG, MRR, MAP, precision, and A/B testing.
Strong Python engineer who writes production-quality code, not just prototypes.
Experience with LLM integration, fine-tuning with LoRA and PEFT, and retrieval-augmented generation.
Worked at startups or product-focused technology companies, not IT consulting or services firms.
Understands the full lifecycle of building ML systems from offline training to online serving and monitoring.
"""

BM25_JD_QUERY = """
AI engineer machine learning ranking retrieval recommendation search
embeddings sentence-transformers FAISS Pinecone Weaviate Qdrant Milvus Elasticsearch
vector database hybrid search semantic search information retrieval
NDCG MRR MAP precision recall evaluation A/B testing
Python PyTorch TensorFlow scikit-learn production deployment
LLM fine-tuning LoRA QLoRA PEFT RAG retrieval-augmented generation
NLP natural language processing text classification named entity recognition
product company startup SaaS platform scale
data pipeline feature engineering model serving inference optimization
"""

# EMBEDDING MODEL

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768

# SCORING WEIGHTS â€” Hybrid fusion

SEMANTIC_WEIGHT = 0.65
BM25_WEIGHT = 0.35

# CONSULTING FIRMS â€” Career trajectory penalty

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'tech mahindra', 'mindtree',
    'mphasis', 'l&t infotech', 'lti', 'ltimindtree', 'hexaware',
    'cyient', 'persistent systems', 'zensar', 'niit technologies',
    'birlasoft', 'coforge', 'sonata software', 'mastek',
}

# TITLE RELEVANCE TIERS â€” Keyword stuffer detection

TITLE_TIER_PERFECT = {
    'ai engineer', 'ml engineer', 'machine learning engineer',
    'senior ai engineer', 'senior ml engineer', 'senior machine learning engineer',
    'lead ai engineer', 'lead ml engineer', 'staff ai engineer',
    'data scientist', 'senior data scientist', 'lead data scientist',
    'nlp engineer', 'senior nlp engineer',
    'recommendation systems engineer', 'search engineer',
    'ranking engineer', 'applied scientist', 'research engineer',
    'junior ai engineer', 'junior ml engineer',
    'senior machine learning engineer',
}

TITLE_TIER_STRONG_ADJACENT = {
    'software engineer', 'senior software engineer', 'staff software engineer',
    'backend engineer', 'senior backend engineer',
    'full stack developer', 'senior full stack developer',
    'data engineer', 'senior data engineer',
    'platform engineer', 'infrastructure engineer',
    'devops engineer', 'senior devops engineer',
    '.net developer', 'java developer', 'python developer',
}

TITLE_TIER_WEAK_ADJACENT = {
    'project manager', 'senior project manager',
    'business analyst', 'senior business analyst',
    'product manager', 'senior product manager',
    'qa engineer', 'senior qa engineer', 'test engineer',
    'cloud engineer', 'senior cloud engineer',
    'frontend engineer', 'senior frontend engineer',
    'mobile developer', 'senior mobile developer',
    'technical lead', 'engineering manager',
}

# Anything not in the above tiers is treated as NON_TECH_TRAP with 0.05 multiplier.
# This includes: HR Manager, Marketing Manager, Sales Executive, Accountant,
# Content Writer, Graphic Designer, Customer Support, Civil Engineer,
# Mechanical Engineer, Operations Manager, etc.

TITLE_SCORES = {
    'perfect': 1.0,
    'strong_adjacent': 0.85,
    'weak_adjacent': 0.5,
    'non_tech_trap': 0.05,
}

# EXPERIENCE RANGE â€” JD says 5-9 years

EXPERIENCE_SCORES = [
    # (min_years, max_years, multiplier)
    (5, 9, 1.0),      # Sweet spot
    (4, 5, 0.85),     # Close â€” slightly under
    (9, 12, 0.85),    # Close â€” slightly over
    (3, 4, 0.6),      # Stretch â€” junior
    (12, 15, 0.6),    # Stretch â€” too senior
    (0, 3, 0.3),      # Poor fit â€” too junior
    (15, 100, 0.3),   # Poor fit â€” too senior
]

# LOCATION SCORING â€” JD prefers Pune/Noida/India

PREFERRED_CITIES_TIER1 = {
    'pune', 'noida', 'delhi', 'new delhi', 'gurgaon', 'gurugram',
    'faridabad', 'ghaziabad', 'greater noida',
}

PREFERRED_CITIES_TIER2 = {
    'hyderabad', 'mumbai', 'bangalore', 'bengaluru', 'chennai',
}

LOCATION_SCORES = {
    'india_tier1': 1.3,
    'india_tier2': 1.15,
    'india_other_relocate': 1.0,
    'india_other_no_relocate': 0.7,
    'outside_india_relocate': 0.5,
    'outside_india_no_relocate': 0.2,
}

# HONEYPOT THRESHOLDS

HONEYPOT_DATE_MISMATCH_TOLERANCE = 0.5  # 50% discrepancy between date-calculated and stated duration
HONEYPOT_CAREER_OVERLAP_DAYS = 60       # Max acceptable days of overlap between roles
HONEYPOT_EXPERT_COUNT_THRESHOLD = 8     # Flag if 8+ expert skills
HONEYPOT_EXPERT_AVG_DURATION_MIN = 6    # Flag if avg duration of expert skills < 6 months
HONEYPOT_EXPERIENCE_RATIO = 1.5         # Flag if total career months > years_of_experience * 12 * this

# ARTIFACT FILE PATHS

ARTIFACTS_DIR = "artifacts"
DENSE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss_index_v2.bin")
BM25_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "bm25_model_v2.pkl")
CANDIDATES_META_PATH = os.path.join(ARTIFACTS_DIR, "candidates_meta.pkl")
JD_VECTOR_PATH = os.path.join(ARTIFACTS_DIR, "jd_vector.npy")


"""
scorer.py â€” Multi-Layer Scoring Engine (Steps 5-6)

Runs during the strict 5-minute inference window.
Loads pre-built artifacts, computes 5 scoring layers, produces ranked top-100.
"""

import re
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import faiss



# â”€â”€ BM25 tokenizer (must match indexer.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# STEP 5A: Honeypot Detection

def is_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with subtly impossible profiles.
    Returns True if any honeypot signal fires.
    """
    # â”€â”€ Check 1: Expert skills with 0 duration_months â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 1) == 0:
            return True

    # â”€â”€ Check 2: Mass expert claims with low durations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    expert_skills = [
        s for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    ]
    if len(expert_skills) >= HONEYPOT_EXPERT_COUNT_THRESHOLD:
        durations = [s.get("duration_months", 0) for s in expert_skills]
        avg_duration = sum(durations) / len(durations) if durations else 0
        if avg_duration < HONEYPOT_EXPERT_AVG_DURATION_MIN:
            return True

    # â”€â”€ Check 3: Career duration exceeds stated experience â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    history = candidate.get("career_history", [])
    total_career_months = sum(h.get("duration_months", 0) for h in history)
    stated_years = candidate.get("profile", {}).get("years_of_experience", 0)
    stated_months = stated_years * 12
    if stated_months > 0 and total_career_months > stated_months * HONEYPOT_EXPERIENCE_RATIO:
        return True

    # â”€â”€ Check 4: Date-math mismatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for role in history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        stated_duration = role.get("duration_months", 0)

        if start_str and end_str and stated_duration > 0:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d")
                end = datetime.strptime(end_str, "%Y-%m-%d")
                calculated_months = max(0, (end.year - start.year) * 12 + (end.month - start.month))
                if calculated_months > 0:
                    discrepancy = abs(calculated_months - stated_duration) / calculated_months
                    if discrepancy > HONEYPOT_DATE_MISMATCH_TOLERANCE:
                        return True
            except (ValueError, TypeError):
                pass

    # â”€â”€ Check 5: Overlapping career timelines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    dated_roles = []
    for role in history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        if start_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d")
                end = (
                    datetime.strptime(end_str, "%Y-%m-%d")
                    if end_str
                    else datetime.now()
                )
                dated_roles.append((start, end))
            except (ValueError, TypeError):
                pass

    dated_roles.sort(key=lambda x: x[0])
    for i in range(len(dated_roles) - 1):
        _, end_i = dated_roles[i]
        start_next, _ = dated_roles[i + 1]
        overlap_days = (end_i - start_next).days
        if overlap_days > HONEYPOT_CAREER_OVERLAP_DAYS:
            return True

    return False


# STEP 5B: Title Relevance Score

def title_relevance_score(candidate: dict) -> float:
    """
    Categorize the candidate's current title into relevance tiers.
    This is the primary defense against keyword-stuffer traps.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower().strip()

    if title in TITLE_TIER_PERFECT:
        return TITLE_SCORES["perfect"]
    if title in TITLE_TIER_STRONG_ADJACENT:
        return TITLE_SCORES["strong_adjacent"]
    if title in TITLE_TIER_WEAK_ADJACENT:
        return TITLE_SCORES["weak_adjacent"]

    # Everything else is a non-tech trap
    return TITLE_SCORES["non_tech_trap"]


# STEP 5C: Career Trajectory Score

def career_trajectory_score(candidate: dict) -> float:
    """
    Penalize consulting-only careers, reward product company experience.
    """
    score = 1.0
    history = candidate.get("career_history", [])
    if not history:
        return 0.5  # No career history - weak signal

    # Check if entire career is consulting
    consulting_months = 0
    total_months = 0
    for role in history:
        dur = role.get("duration_months", 0)
        total_months += dur
        company = role.get("company", "").lower()
        if any(firm in company for firm in CONSULTING_FIRMS):
            consulting_months += dur
            
    if total_months > 0 and (consulting_months / total_months) > 0.95:
        score *= 0.1  # Heavy penalty for pure consultants

    # Calculate product company experience
    product_months = 0
    for role in history:
        company = role.get("company", "").lower()
        if not any(firm in company for firm in CONSULTING_FIRMS):
            product_months += role.get("duration_months", 0)

    if product_months > 48:    # 4+ years at product companies
        score *= 1.3
    elif product_months > 24:  # 2-4 years
        score *= 1.1
    elif product_months < 12:  # < 1 year
        score *= 0.7

    return score


# STEP 5D: Experience Range Score

def experience_range_score(candidate: dict) -> float:
    """
    Score based on how well years_of_experience fits the JD range (5-9).
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    for min_y, max_y, multiplier in EXPERIENCE_SCORES:
        if min_y <= yoe < max_y:
            return multiplier

    # Exact boundary: 9 years = 1.0
    if yoe == 9:
        return 1.0

    return 0.3  # Fallback


# STEP 5E: Location Score

def location_score(candidate: dict) -> float:
    """
    Score based on location proximity to Pune/Noida and willingness to relocate.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    country = profile.get("country", "").strip().lower()
    location = profile.get("location", "").strip().lower()
    willing = signals.get("willing_to_relocate", False)

    is_india = country == "india"

    if is_india:
        # Check if in preferred city tier 1
        if any(city in location for city in PREFERRED_CITIES_TIER1):
            return LOCATION_SCORES["india_tier1"]
        # Check if in preferred city tier 2
        if any(city in location for city in PREFERRED_CITIES_TIER2):
            return LOCATION_SCORES["india_tier2"]
        # Other Indian city
        if willing:
            return LOCATION_SCORES["india_other_relocate"]
        return LOCATION_SCORES["india_other_no_relocate"]
    else:
        # Outside India
        if willing:
            return LOCATION_SCORES["outside_india_relocate"]
        return LOCATION_SCORES["outside_india_no_relocate"]


# STEP 6: Final Score Combination

def compute_final_score(
    semantic_sim: float,
    bm25_sim: float,
    honeypot: bool,
    title_sc: float,
    trajectory_sc: float,
    experience_sc: float,
    location_sc: float,
    behavioral_mult: float = 1.0,
) -> float:
    """
    Combine all scoring layers into a single final score.
    """
    # Short-circuit honeypots
    if honeypot:
        return 0.001

    # Hybrid base score
    hybrid = SEMANTIC_WEIGHT * semantic_sim + BM25_WEIGHT * bm25_sim

    # Apply all multipliers
    hybrid *= title_sc
    hybrid *= trajectory_sc
    hybrid *= experience_sc
    hybrid *= location_sc
    hybrid *= behavioral_mult

    return hybrid


# STEP 5F: Pure Researcher Penalty (Soft Filter)

def pure_researcher_penalty(candidate: dict) -> float:
    """
    Penalize candidates whose entire career is in academic/research roles
    with no production engineering experience.
    JD explicitly says: 'If you've spent your career in pure research
    environments without any production deployment â€” we will not move forward.'
    """
    history = candidate.get("career_history", [])
    if not history:
        return 1.0  # No data â€” don't penalize

    research_titles = {
        "professor", "researcher", "research assistant", "research associate",
        "research fellow", "postdoc", "postdoctoral", "phd", "doctoral",
        "lab assistant", "teaching assistant", "academic",
        "research scientist", "principal researcher",
    }

    research_count = 0
    for role in history:
        title = role.get("title", "").lower().strip()
        if any(rt in title for rt in research_titles):
            research_count += 1

    # If ALL roles are research/academic -> heavy penalty
    if research_count == len(history):
        return 0.2
    # If most roles are research -> moderate penalty
    elif research_count > len(history) * 0.7:
        return 0.5

    return 1.0


# STEP 5G: Notice Period Penalty (Soft Filter)

def notice_period_penalty(candidate: dict) -> float:
    """
    Penalize candidates with high notice periods.
    JD says: 'We'd love sub-30-day notice. 30+ day notice candidates
    are still in scope but the bar gets higher.'
    """
    signals = candidate.get("redrob_signals", {})
    notice_days = signals.get("notice_period_days", 0)

    if notice_days <= 30:
        return 1.0   # Ideal
    elif notice_days <= 60:
        return 0.85  # Acceptable
    elif notice_days <= 90:
        return 0.7   # High bar
    else:
        return 0.5   # Very long â€” significant penalty


# STEP 5H: Title Chaser Penalty (Soft Filter)

def title_chaser_penalty(candidate: dict) -> float:
    """
    Penalize candidates who switch companies too frequently.
    JD says: 'If your career trajectory shows you optimizing for titles
    by switching companies every 1.5 years, we're not a fit.'
    """
    history = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if not history or yoe < 3 or len(history) < 2:
        return 1.0  # Not enough data or too junior to judge

    # Count unique companies
    companies = set()
    for role in history:
        company = role.get("company", "").lower().strip()
        if company:
            companies.add(company)

    if not companies:
        return 1.0

    avg_tenure_years = yoe / len(companies)

    if avg_tenure_years < 1.0:
        return 0.4   # Extreme job-hopper
    elif avg_tenure_years < 1.5:
        return 0.6   # Frequent switcher
    elif avg_tenure_years < 2.0:
        return 0.85  # Slight concern

    return 1.0


# NEW JD-SPECIFIC SOFT FILTERS

def non_coding_architect_penalty(candidate: dict) -> float:
    """
    JD: If you are a senior engineer who hasn't written production code in the last
    18 months because you've moved into 'architecture' or 'tech lead' roles...
    """
    history = candidate.get("career_history", [])
    if not history:
        return 1.0
        
    # Check the most recent role
    current_role = sorted(history, key=lambda x: x.get("start_date", ""), reverse=True)[0]
    title = current_role.get("title", "").lower()
    dur = current_role.get("duration_months", 0)
    
    if ("architect" in title or "head of" in title or "vp" in title or "director" in title):
        if dur > 18:
            return 0.3  # Heavy penalty
    return 1.0

def langchain_wrapper_penalty(candidate: dict) -> float:
    """
    JD: If your 'AI experience' consists primarily of recent projects using 
    LangChain to call OpenAI without pre-LLM-era ML production experience...
    """
    skills = [s.get("name", "").lower() for s in candidate.get("skills", [])]
    
    has_wrapper = any(w in sk for w in ["langchain", "openai", "llama_index"] for sk in skills)
    has_fundamentals = any(f in sk for f in ["machine learning", "nlp", "information retrieval", "pytorch", "tensorflow"] for sk in skills)
    
    if has_wrapper and not has_fundamentals:
        return 0.4
    return 1.0

def closed_source_penalty(candidate: dict) -> float:
    """
    JD: People whose work has been entirely on closed-source proprietary systems 
    for 5+ years without external validation (papers, talks, open-source).
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    github_score = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    
    if yoe > 5 and (github_score == -1 or github_score == 0):
        return 0.8  # Slight penalty
    return 1.0


# Behavioral Multiplier

def get_behavioral_multiplier(candidate: dict) -> float:
    """
    Compute a behavioral quality multiplier from the 23 Redrob signals.
    Penalizes unresponsive and inactive candidates.
    """
    signals = candidate.get("redrob_signals", {})
    mult = 1.0

    # Penalize Ghosters (inactive > 180 days AND low response rate)
    response_rate = signals.get("recruiter_response_rate", 0.5)
    last_active = signals.get("last_active_date")
    days_inactive = 0
    if last_active:
        try:
            active_date = datetime.strptime(last_active, "%Y-%m-%d")
            days_inactive = (datetime.now() - active_date).days
        except (ValueError, TypeError):
            pass

    if days_inactive > 180 and response_rate < 0.10:
        return 0.0  # Instant drop for Ghosters

    # Normal activity penalties
    if days_inactive > 180:
        mult *= 0.5
    elif days_inactive > 90:
        mult *= 0.8

    # Normal response rate penalties
    if response_rate < 0.2:
        mult *= 0.6
        
    # Profile Completeness
    completeness = signals.get("profile_completeness_score", 50)
    if completeness < 50:
        mult *= 0.7
    elif completeness > 90:
        mult *= 1.05

    # Reward open-to-work candidates
    if signals.get("open_to_work_flag", False):
        mult *= 1.05

    # Interview completion rate
    interview_rate = signals.get("interview_completion_rate", 0.5)
    if interview_rate < 0.4:
        mult *= 0.6
    elif interview_rate > 0.85:
        mult *= 1.12

        # POSITIVE BOOST LOGIC
        
    # Boost 1: High demand (saved by recruiters)
    saves = signals.get("saved_by_recruiters_30d", 0)
    if saves > 10:
        mult *= 1.15
    elif saves > 0:
        mult *= 1.05

    # Boost 2: Active open-source/coding footprint
    github_score = signals.get("github_activity_score", -1)
    if github_score > 80:
        mult *= 1.10
    elif github_score > 40:
        mult *= 1.05

    # Boost 3: Highly complete and verified profile
    if signals.get("verified_email", False) and signals.get("verified_phone", False):
        mult *= 1.02

    return mult


# MAIN RANKING PIPELINE

def rank_candidates(
    top_k: int = 100,
) -> list[dict]:
    """
    Load pre-built artifacts, compute all scoring layers, return ranked top-K.
    """

    # â”€â”€ 1. Load artifacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Loading artifacts ...")
    with open(CANDIDATES_META_PATH, "rb") as f:
        candidates = pickle.load(f)

    index = faiss.read_index(DENSE_INDEX_PATH)

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)

    jd_vector = np.load(JD_VECTOR_PATH)

    n = len(candidates)
    print(f"  Loaded {n} candidates, FAISS index, BM25 index")

    # â”€â”€ 2. Compute semantic scores (dense) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Computing semantic scores ...")
    D, I = index.search(jd_vector, n)
    semantic_scores = np.zeros(n)
    for score, idx in zip(D[0], I[0]):
        semantic_scores[idx] = score

    # â”€â”€ 3. Compute BM25 scores (sparse) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Computing BM25 keyword scores ...")
    tokenized_query = tokenize(BM25_JD_QUERY)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 to [0, 1]
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    # â”€â”€ 4. Apply 8 scoring layers + combine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Applying multi-layer scoring (8 layers) ...")
    scored = []
    honeypot_count = 0

    for i, candidate in enumerate(candidates):
        hp = is_honeypot(candidate)
        if hp:
            honeypot_count += 1

        t_score = title_relevance_score(candidate)
        c_score = career_trajectory_score(candidate)
        e_score = experience_range_score(candidate)
        l_score = location_score(candidate)
        b_mult = get_behavioral_multiplier(candidate)

        # New soft penalties from JD disqualifiers
        research_pen = pure_researcher_penalty(candidate)
        notice_pen = notice_period_penalty(candidate)
        chaser_pen = title_chaser_penalty(candidate)
        
        arch_pen = non_coding_architect_penalty(candidate)
        wrapper_pen = langchain_wrapper_penalty(candidate)
        closed_pen = closed_source_penalty(candidate)

        f_score = compute_final_score(
            semantic_sim=semantic_scores[i],
            bm25_sim=bm25_scores[i],
            honeypot=hp,
            title_sc=t_score,
            trajectory_sc=c_score,
            experience_sc=e_score,
            location_sc=l_score,
            behavioral_mult=b_mult,
        )

        # Apply soft penalties on top of the base score
        f_score *= research_pen
        f_score *= notice_pen
        f_score *= chaser_pen
        f_score *= arch_pen
        f_score *= wrapper_pen
        f_score *= closed_pen

        scored.append({
            "candidate_id": candidate["candidate_id"],
            "final_score": f_score,
            "semantic_score": float(semantic_scores[i]),
            "bm25_score": float(bm25_scores[i]),
            "is_honeypot": hp,
            "title_score": t_score,
            "trajectory_score": c_score,
            "experience_score": e_score,
            "location_score": l_score,
            "behavioral_mult": b_mult,
            "research_penalty": research_pen,
            "notice_penalty": notice_pen,
            "chaser_penalty": chaser_pen,
            # Keep reference for reasoning generation (Role 3)
            "profile": candidate.get("profile", {}),
            "skills": candidate.get("skills", []),
            "career_history": candidate.get("career_history", []),
            "redrob_signals": candidate.get("redrob_signals", {}),
        })

    # â”€â”€ Normalize final scores to [0, 1] range â”€â”€â”€â”€â”€
    # Divide by the actual maximum score in the batch to prevent any clipping at 1.0
    # This preserves the exact relative differences between all candidates.
    max_score = max(c["final_score"] for c in scored)
    if max_score > 0:
        for c in scored:
            c["final_score"] = round(c["final_score"] / max_score, 4)

    # â”€â”€ 5. Sort: score descending, tie-break on candidate_id ascending â”€â”€â”€
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # â”€â”€ 6. Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n-- Results --")
    print(f"  Total candidates scored: {n}")
    print(f"  Honeypots detected: {honeypot_count}")

    top = scored[:min(10, len(scored))]
    print(f"\n  Top {len(top)} candidates:")
    print(f"  {'Rank':<5} {'ID':<15} {'Score':>8} {'Sem':>6} {'KW':>6} "
          f"{'Title':>6} {'Traj':>6} {'Exp':>5} {'Loc':>5} {'Beh':>5} {'HP'}")
    for rank, c in enumerate(top, 1):
        print(
            f"  {rank:<5} {c['candidate_id']:<15} {c['final_score']:>8.4f} "
            f"{c['semantic_score']:>6.3f} {c['bm25_score']:>6.3f} "
            f"{c['title_score']:>6.2f} {c['trajectory_score']:>6.2f} "
            f"{c['experience_score']:>5.2f} {c['location_score']:>5.2f} "
            f"{c['behavioral_mult']:>5.2f} {'!HP' if c['is_honeypot'] else 'OK'}"
        )

    # Check honeypot rate in top 100
    top_100 = scored[:100]
    hp_in_top = sum(1 for c in top_100 if c["is_honeypot"])
    hp_rate = hp_in_top / len(top_100) * 100 if top_100 else 0
    if hp_rate > 10:
        print(f"\n  [DANGER] {hp_rate:.1f}% honeypots in top 100 -- DISQUALIFICATION RISK!")
    else:
        print(f"\n  [OK] Honeypot rate in top 100: {hp_rate:.1f}% (safe)")

    return scored[:top_k]


if __name__ == "__main__":
    rank_candidates()


"""
rank.py -- Final Ranking Script (Role 3)

Produces the submission CSV from pre-computed artifacts.
This is the script judges will run inside the 5-minute sandbox.

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
"""

import os
import sys
import csv
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from src.ml.scorer import rank_candidates


# REASONING ENGINE

# JD core requirements for connecting reasoning to the job description
JD_CORE_SKILLS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "sentence-transformers", "bge", "e5", "embeddings",
    "ranking", "retrieval", "recommendation", "search",
    "ndcg", "mrr", "map", "a/b testing", "evaluation",
    "lora", "qlora", "peft", "rag", "fine-tuning",
    "pytorch", "tensorflow", "scikit-learn",
    "nlp", "natural language processing",
    "python", "machine learning", "deep learning", "neural network",
}


def generate_reasoning(candidate: dict, rank: int) -> str:
    """
    Generate a specific, honest, varied reasoning string for each candidate.

    Rules we follow (from hackathon Stage 4 manual review):
    - Reference specific facts from the candidate's profile
    - Connect to JD requirements
    - Acknowledge concerns honestly
    - Never hallucinate (only cite skills that exist)
    - Vary the reasoning across candidates
    - Match tone to rank position
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", []) if "career_history" in candidate else []

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")
    company = profile.get("current_company", "Unknown")
    industry = profile.get("current_industry", "")

    # Extract actual skill names from the candidate's profile (no hallucination)
    skill_names = [s.get("name", "").lower() for s in skills]
    skill_names_display = [s.get("name", "") for s in skills]

    # Find JD-relevant skills this candidate actually has
    matched_jd_skills = []
    for s in skill_names_display:
        if s.lower() in JD_CORE_SKILLS:
            matched_jd_skills.append(s)
    # Also check partial matches (e.g., "Machine Learning" contains "machine learning")
    for s in skill_names_display:
        sl = s.lower()
        for jd_skill in JD_CORE_SKILLS:
            if jd_skill in sl or sl in jd_skill:
                if s not in matched_jd_skills:
                    matched_jd_skills.append(s)
                    break

    # Behavioral signals
    response_rate = signals.get("recruiter_response_rate", -1)
    notice_period = signals.get("notice_period_days", -1)
    open_to_work = signals.get("open_to_work_flag", False)
    last_active = signals.get("last_active_date", "")
    interview_rate = signals.get("interview_completion_rate", -1)
    github_score = signals.get("github_activity_score", -1)

    # Career info
    recent_roles = []
    for role in career[:2]:
        r_title = role.get("title", "")
        r_company = role.get("company", "")
        r_duration = role.get("duration_months", 0)
        if r_title and r_company:
            recent_roles.append(f"{r_title} at {r_company} ({r_duration}mo)")

    # â”€â”€ Build reasoning parts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    parts = []

    # Part 1: Title + Experience + Company (always included)
    if rank <= 10:
        parts.append(f"{title} with {yoe:.1f} years of experience currently at {company}")
    elif rank <= 50:
        parts.append(f"{title} with {yoe:.1f} years based in {location}")
    else:
        parts.append(f"{title} ({yoe:.1f} yrs) in {location} {country}")

    # Part 2: JD-relevant skills (specific facts, no hallucination)
    if matched_jd_skills:
        if rank <= 20:
            skill_str = " | ".join(matched_jd_skills[:4])
            parts.append(f"directly relevant skills include {skill_str}")
        elif rank <= 50:
            skill_str = " | ".join(matched_jd_skills[:3])
            parts.append(f"relevant skills: {skill_str}")
        else:
            skill_str = " | ".join(matched_jd_skills[:2])
            parts.append(f"some relevant skills ({skill_str})")
    else:
        if rank <= 50:
            parts.append("limited direct overlap with core JD skill requirements (ranking, retrieval, embeddings)")
        else:
            parts.append("minimal JD skill alignment")

    # Part 3: Career trajectory (for top candidates)
    if recent_roles and rank <= 30:
        parts.append(f"recent experience: {recent_roles[0]}")

    # Part 4: Strengths (tone matches rank)
    strengths = []
    concerns = []

    # Location
    if country.lower() == "india":
        city_lower = location.lower()
        if any(c in city_lower for c in ["pune", "noida", "delhi", "gurgaon", "gurugram"]):
            strengths.append("based in preferred JD location")
        elif any(c in city_lower for c in ["bangalore", "bengaluru", "hyderabad", "mumbai", "chennai"]):
            strengths.append(f"India-based ({location})")
    else:
        concerns.append(f"located outside India ({country})")

    # Experience fit
    if 5 <= yoe <= 9:
        strengths.append("experience within JD's 5-9 year sweet spot")
    elif yoe < 5:
        concerns.append(f"under the JD's 5-year minimum ({yoe:.1f} yrs)")
    elif yoe > 12:
        concerns.append(f"significantly over JD's 9-year upper range ({yoe:.1f} yrs)")
    elif yoe > 9:
        concerns.append(f"slightly over JD's preferred range ({yoe:.1f} yrs)")

    # Behavioral signals
    if response_rate >= 0:
        if response_rate >= 0.6:
            strengths.append(f"strong recruiter engagement ({response_rate:.0%} response rate)")
        elif response_rate < 0.15:
            concerns.append(f"very low recruiter response rate ({response_rate:.0%})")
        elif response_rate < 0.3:
            concerns.append(f"below-average recruiter response rate ({response_rate:.0%})")

    if notice_period >= 0:
        if notice_period <= 30:
            strengths.append(f"short notice period ({notice_period}d)")
        elif notice_period >= 90:
            concerns.append(f"long notice period ({notice_period} days)")

    if open_to_work:
        strengths.append("actively open to new opportunities")

    if github_score >= 0:
        if github_score >= 60:
            strengths.append(f"strong GitHub presence (score: {github_score})")
        elif github_score <= 10 and github_score >= 0:
            concerns.append("minimal GitHub activity")

    if interview_rate >= 0 and interview_rate < 0.5:
        concerns.append(f"low interview completion rate ({interview_rate:.0%})")

    # Part 5: Honeypot flag
    if candidate.get("is_honeypot", False):
        concerns.append("profile contains timeline inconsistencies suggesting data integrity issues")

    # â”€â”€ Assemble final reasoning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # Add strengths
    if strengths and rank <= 60:
        top_strengths = strengths[:2] if rank <= 30 else strengths[:1]
        parts.append("; ".join(top_strengths))

    # Add concerns (honest acknowledgment)
    if concerns:
        if rank <= 10:
            # Top candidates: minor concern framing
            if len(concerns) == 1:
                parts.append(f"minor consideration: {concerns[0]}")
        elif rank <= 50:
            # Mid-range: balanced concern
            parts.append(f"however, {concerns[0]}")
        else:
            # Bottom half: concerns are the main story
            concern_str = "; ".join(concerns[:2])
            parts.append(f"concerns: {concern_str}")

    reasoning = "; ".join(parts) + "."

    # Capitalize first letter
    reasoning = reasoning[0].upper() + reasoning[1:]

    return reasoning


# SUBMISSION CSV WRITER

def write_submission_csv(scored_candidates: list, output_path: str):
    """
    Write the final submission CSV with exactly 100 rows.
    Enforces all hackathon format rules.
    """
    top_100 = scored_candidates[:100]

    # Validate we have exactly 100
    if len(top_100) < 100:
        print(f"[WARN] Only {len(top_100)} candidates available (need 100)")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        prev_score = float("inf")
        for rank, candidate in enumerate(top_100, 1):
            cid = candidate["candidate_id"]
            score = round(candidate["final_score"], 4)

            # Enforce non-increasing scores
            if score > prev_score:
                score = prev_score
            prev_score = score

            # Remove any accidental commas to prevent CSV quotes
            reasoning = generate_reasoning(candidate, rank).replace(",", ";")

            # Format score to exactly 4 decimal places
            score_str = f"{score:.4f}"

            writer.writerow([cid, rank, score_str, reasoning])

    print(f"\n[OK] Submission written to: {output_path}")
    print(f"     Rows: {len(top_100)} + 1 header = {len(top_100) + 1} total")


# VALIDATION

def validate_submission(output_path: str):
    """Run local validation checks matching the hackathon auto-validator."""
    print("\n-- Submission Validation --")
    errors = []

    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check 1: Exactly 100 rows
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")

    # Check 2: Required columns
    expected_cols = {"candidate_id", "rank", "score", "reasoning"}
    if rows:
        actual_cols = set(rows[0].keys())
        missing = expected_cols - actual_cols
        if missing:
            errors.append(f"Missing columns: {missing}")

    # Check 3: Ranks 1-100, each exactly once
    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be exactly 1-100, each used once")

    # Check 4: Unique candidate_ids
    cids = [r["candidate_id"] for r in rows]
    if len(set(cids)) != len(cids):
        errors.append("Duplicate candidate_ids found")

    # Check 5: candidate_id format
    import re
    for cid in cids:
        if not re.match(r"^CAND_\d{7}$", cid):
            errors.append(f"Invalid candidate_id format: {cid}")
            break

    # Check 6: Non-increasing scores
    scores = [float(r["score"]) for r in rows]
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            errors.append(f"Score increases at rank {i+1}: {scores[i]} > {scores[i-1]}")
            break

    # Check 7: No empty reasoning
    empty_count = sum(1 for r in rows if not r.get("reasoning", "").strip())
    if empty_count > 0:
        errors.append(f"{empty_count} rows have empty reasoning")

    # Check 8: Reasoning variation (no identical strings)
    reasonings = [r.get("reasoning", "") for r in rows]
    unique_reasonings = len(set(reasonings))
    if unique_reasonings < 90:  # Allow a tiny bit of collision
        errors.append(f"Only {unique_reasonings}/100 unique reasonings (too templated)")

    if errors:
        print("  [FAIL] Validation errors:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  [PASS] All 8 checks passed!")
        print(f"  Score range: {scores[0]:.4f} (rank 1) to {scores[-1]:.4f} (rank 100)")
        print(f"  Unique reasonings: {unique_reasonings}/100")

        # Sample 3 reasonings to eyeball
        print("\n  -- Sample Reasonings --")
        for sample_rank in [1, 50, 100]:
            r = rows[sample_rank - 1]
            print(f"  Rank {sample_rank}: {r['reasoning'][:120]}...")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon - Produce submission CSV")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates.jsonl (not used if artifacts exist)")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV filename")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of candidates to rank (default: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("  REDROB HACKATHON - SUBMISSION GENERATOR")
    print("=" * 60)

    start = time.time()

    # Step 1: Run the scorer to get ranked candidates
    print("\n[Step 1] Running scoring engine ...")
    scored = rank_candidates(top_k=args.top_k)

    # Step 2: Generate reasoning and write CSV
    print("\n[Step 2] Generating reasonings and writing CSV ...")
    write_submission_csv(scored, args.out)

    # Step 3: Validate
    print("\n[Step 3] Validating submission ...")
    validate_submission(args.out)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  TOTAL TIME: {elapsed:.1f} seconds")
    if elapsed > 300:
        print("  [WARN] Exceeded 5-minute limit!")
    else:
        print(f"  [OK] Within 5-minute limit ({elapsed/300*100:.1f}% used)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
