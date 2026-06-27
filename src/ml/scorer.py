"""
scorer.py — Multi-Layer Scoring Engine (Steps 5-6)

Runs during the strict 5-minute inference window.
Loads pre-built artifacts, computes 5 scoring layers, produces ranked top-100.
"""

import re
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import faiss

from src.ml.config import (
    BM25_JD_QUERY,
    SEMANTIC_WEIGHT,
    BM25_WEIGHT,
    CONSULTING_FIRMS,
    TITLE_TIER_PERFECT,
    TITLE_TIER_STRONG_ADJACENT,
    TITLE_TIER_WEAK_ADJACENT,
    TITLE_SCORES,
    EXPERIENCE_SCORES,
    PREFERRED_CITIES_TIER1,
    PREFERRED_CITIES_TIER2,
    LOCATION_SCORES,
    HONEYPOT_DATE_MISMATCH_TOLERANCE,
    HONEYPOT_CAREER_OVERLAP_DAYS,
    HONEYPOT_EXPERT_COUNT_THRESHOLD,
    HONEYPOT_EXPERT_AVG_DURATION_MIN,
    HONEYPOT_EXPERIENCE_RATIO,
    DENSE_INDEX_PATH,
    BM25_INDEX_PATH,
    CANDIDATES_META_PATH,
    JD_VECTOR_PATH,
)


# ── BM25 tokenizer (must match indexer.py) ───────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# =============================================================================
# STEP 5A: Honeypot Detection
# =============================================================================

def is_honeypot(candidate: dict) -> bool:
    """
    Detect candidates with subtly impossible profiles.
    Returns True if any honeypot signal fires.
    """
    # ── Check 1: Expert skills with 0 duration_months ────────────────────
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 1) == 0:
            return True

    # ── Check 2: Mass expert claims with low durations ───────────────────
    expert_skills = [
        s for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    ]
    if len(expert_skills) >= HONEYPOT_EXPERT_COUNT_THRESHOLD:
        durations = [s.get("duration_months", 0) for s in expert_skills]
        avg_duration = sum(durations) / len(durations) if durations else 0
        if avg_duration < HONEYPOT_EXPERT_AVG_DURATION_MIN:
            return True

    # ── Check 3: Career duration exceeds stated experience ───────────────
    history = candidate.get("career_history", [])
    total_career_months = sum(h.get("duration_months", 0) for h in history)
    stated_years = candidate.get("profile", {}).get("years_of_experience", 0)
    stated_months = stated_years * 12
    if stated_months > 0 and total_career_months > stated_months * HONEYPOT_EXPERIENCE_RATIO:
        return True

    # ── Check 4: Date-math mismatch ──────────────────────────────────────
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

    # ── Check 5: Overlapping career timelines ────────────────────────────
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


# =============================================================================
# STEP 5B: Title Relevance Score
# =============================================================================

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


# =============================================================================
# STEP 5C: Career Trajectory Score
# =============================================================================

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


# =============================================================================
# STEP 5D: Experience Range Score
# =============================================================================

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


# =============================================================================
# STEP 5E: Location Score
# =============================================================================

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


# =============================================================================
# STEP 6: Final Score Combination
# =============================================================================

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


# =============================================================================
# STEP 5F: Pure Researcher Penalty (Soft Filter)
# =============================================================================

def pure_researcher_penalty(candidate: dict) -> float:
    """
    Penalize candidates whose entire career is in academic/research roles
    with no production engineering experience.
    JD explicitly says: 'If you've spent your career in pure research
    environments without any production deployment — we will not move forward.'
    """
    history = candidate.get("career_history", [])
    if not history:
        return 1.0  # No data — don't penalize

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


# =============================================================================
# STEP 5G: Notice Period Penalty (Soft Filter)
# =============================================================================

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
        return 0.5   # Very long — significant penalty


# =============================================================================
# STEP 5H: Title Chaser Penalty (Soft Filter)
# =============================================================================

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


# =============================================================================
# NEW JD-SPECIFIC SOFT FILTERS
# =============================================================================

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


# =============================================================================
# Behavioral Multiplier
# =============================================================================

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

    # ==========================================
    # POSITIVE BOOST LOGIC
    # ==========================================
    
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


# =============================================================================
# MAIN RANKING PIPELINE
# =============================================================================

def rank_candidates(
    top_k: int = 100,
) -> list[dict]:
    """
    Load pre-built artifacts, compute all scoring layers, return ranked top-K.
    """

    # ── 1. Load artifacts ────────────────────────────────────────────────
    print("Loading artifacts ...")
    with open(CANDIDATES_META_PATH, "rb") as f:
        candidates = pickle.load(f)

    index = faiss.read_index(DENSE_INDEX_PATH)

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)

    jd_vector = np.load(JD_VECTOR_PATH)

    n = len(candidates)
    print(f"  Loaded {n} candidates, FAISS index, BM25 index")

    # ── 2. Compute semantic scores (dense) ───────────────────────────────
    print("Computing semantic scores ...")
    D, I = index.search(jd_vector, n)
    semantic_scores = np.zeros(n)
    for score, idx in zip(D[0], I[0]):
        semantic_scores[idx] = score

    # ── 3. Compute BM25 scores (sparse) ──────────────────────────────────
    print("Computing BM25 keyword scores ...")
    tokenized_query = tokenize(BM25_JD_QUERY)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 to [0, 1]
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max

    # ── 4. Apply 8 scoring layers + combine ──────────────────────────────
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

    # ── Normalize final scores to [0, 1] range ─────
    # Divide by the actual maximum score in the batch to prevent any clipping at 1.0
    # This preserves the exact relative differences between all candidates.
    max_score = max(c["final_score"] for c in scored)
    if max_score > 0:
        for c in scored:
            c["final_score"] = round(c["final_score"] / max_score, 4)

    # ── 5. Sort: score descending, tie-break on candidate_id ascending ───
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # ── 6. Report ────────────────────────────────────────────────────────
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
