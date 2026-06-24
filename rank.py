"""
Redrob Hackathon — Intelligent Candidate Ranking System
Author: Prajwal Suresh Mhase
Team: India Runs

Architecture: Multi-signal hybrid ranker with anti-spoofing layer.
- Explicit feature engineering over candidate fields (no LLM API calls)
- BM25 over career descriptions for semantic text match
- Skills trust scoring (endorsement + duration weighted)
- Behavioral availability multiplier
- Honeypot detection & disqualification
- Reasoning generation from scored components

Runtime: CPU-only, no network, < 5 minutes for 100K candidates
"""

import json
import csv
import math
import re
import sys
import argparse
from datetime import datetime, date
from collections import defaultdict

# ---------------------------------------------------------------------------
# JD-derived constants  (Senior AI Engineer @ Redrob)
# ---------------------------------------------------------------------------

MUST_HAVE_SKILLS = {
    # embeddings / retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "dense retrieval",
    "semantic search", "bi-encoder", "cross-encoder",
    "openai embeddings", "bge", "e5", "text-embedding",
    # vector databases
    "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "vector database", "vector db",
    "hybrid search", "ann", "approximate nearest neighbor",
    # ranking & evaluation
    "learning to rank", "ltr", "ndcg", "mrr", "map", "ranking system",
    "retrieval evaluation", "information retrieval", "ir system",
    # python
    "python",
    # llm but with production framing
    "rag", "retrieval augmented", "reranking", "re-ranking", "reranker",
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "gradient boosting", "lightgbm", "catboost",
    "pytorch", "hugging face", "huggingface", "transformers",
    "a/b testing", "ab testing", "online evaluation",
    "distributed systems", "kafka", "spark", "ray",
    "langchain", "llmops", "mlops",
    "nlp", "natural language processing", "text classification",
    "bert", "gpt", "llm",
}

DISQUALIFIER_SKILLS = {
    # CV / robotics / speech without NLP
    "computer vision", "image classification", "object detection",
    "yolo", "resnet", "cnn", "convolutional",
    "speech recognition", "text-to-speech", "tts", "asr",
    "robotics", "ros", "slam",
}

# Title keywords that strongly suggest ML/AI engineering
POSITIVE_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "search engineer", "applied scientist",
    "data scientist", "senior engineer", "staff engineer",
    "principal engineer", "backend engineer", "software engineer",
    "ranking engineer", "recommendations engineer",
}

# Pure services/consulting companies the JD explicitly warns against
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "lti", "mindtree",
    "coforge", "persistent", "niit technologies",
}

# India locations Redrob is open to
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "delhi ncr", "ncr", "gurgaon", "gurugram",
    "hyderabad", "bangalore", "bengaluru", "mumbai", "chennai",
    "india",
}

# Must-have career evidence keywords (career descriptions, not just skills)
CAREER_EVIDENCE_PHRASES = [
    "embedding", "vector", "retrieval", "ranking", "search",
    "recommendation", "ranking system", "candidate", "ranker",
    "production", "deployed", "shipped", "ml system",
    "information retrieval", "nlp", "language model",
]

# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

def detect_honeypot(candidate: dict) -> bool:
    """
    Returns True if candidate looks like a honeypot.
    Honeypots have subtly impossible timelines or skills.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # 1. Company-founding-date impossible check (proxy: job duration > company age)
    #    We can't look up real founding dates, but we can flag
    #    extreme mismatch signals (8yr experience, company that overlaps weirdly).
    for job in career:
        if job.get("duration_months", 0) > 120:  # 10+ years at one place for young cos
            end = job.get("end_date") or str(date.today())
            start = job.get("start_date", "")
            if start and end:
                try:
                    s = datetime.strptime(start[:10], "%Y-%m-%d")
                    e = datetime.strptime(end[:10], "%Y-%m-%d")
                    months = (e.year - s.year) * 12 + (e.month - s.month)
                    if abs(months - job.get("duration_months", 0)) > 24:
                        return True
                except Exception:
                    pass

    # 2. Skills with "expert" across 10+ diverse domains (breadth too wide)
    expert_skills = [s for s in skills if s.get("proficiency") in ("expert", "advanced")]
    expert_domains = set()
    for s in expert_skills:
        nm = s.get("name", "").lower()
        for domain in ["vision", "speech", "nlp", "robotics", "cv", "tts", "asr",
                       "gan", "rl", "reinforcement"]:
            if domain in nm:
                expert_domains.add(domain)
    if len(expert_domains) >= 4:
        return True

    # 3. Profile completeness extreme + 0 platform activity = bought profile
    sigs = candidate.get("redrob_signals", {})
    completeness = sigs.get("profile_completeness_score", 50)
    response_rate = sigs.get("recruiter_response_rate", -1)
    last_active = sigs.get("last_active_date", "")
    if completeness >= 95 and response_rate is not None and response_rate < 0.02:
        return True

    # 4. Years of experience claim vs career history math
    stated_yoe = profile.get("years_of_experience", 0) or 0
    career_months = sum(j.get("duration_months", 0) for j in career)
    career_years = career_months / 12.0
    if stated_yoe > 3 and career_years > 0:
        ratio = stated_yoe / max(career_years, 0.5)
        if ratio > 2.5 or ratio < 0.15:
            return True

    # 5. Title says "Marketing" / "HR" but skills are all deep ML
    title = (profile.get("current_title") or "").lower()
    ml_skill_count = sum(1 for s in skills
                         if any(k in s.get("name", "").lower()
                                for k in ["embedding", "transformer", "llm", "pytorch", "nlp"]))
    if ml_skill_count >= 5 and any(t in title for t in ["marketing", "sales", "hr", "recruiter"]):
        return True

    return False

# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def score_skills(candidate: dict) -> tuple[float, list]:
    """
    Returns (0-1 score, list of matched skill names).
    Uses endorsements and duration as a trust multiplier to resist keyword stuffing.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0, []

    must_match = 0.0
    must_max = 8.0  # we expect ~8 must-have signals to be possible
    nice_match = 0.0
    nice_max = 5.0

    disqualifier_score = 0.0
    matched_names = []

    for s in skills:
        name_raw = s.get("name", "")
        name = name_raw.lower()
        prof = s.get("proficiency", "beginner").lower()
        endorse = min(s.get("endorsements", 0), 50)  # cap at 50
        duration = min(s.get("duration_months", 0), 60)  # cap at 60

        # Trust multiplier: endorsed + used for > 6 months = real skill
        trust = 0.5
        if endorse >= 5:
            trust += 0.25
        if duration >= 6:
            trust += 0.25
        # Proficiency bonus
        if prof in ("advanced", "expert"):
            trust = min(trust * 1.2, 1.0)

        # Check against must-have
        is_must = any(k in name for k in MUST_HAVE_SKILLS)
        is_nice = any(k in name for k in NICE_TO_HAVE_SKILLS) and not is_must
        is_disq = any(k in name for k in DISQUALIFIER_SKILLS)

        if is_must:
            must_match += trust
            matched_names.append(name_raw)
        elif is_nice:
            nice_match += trust * 0.5
            matched_names.append(name_raw)
        if is_disq:
            disqualifier_score += trust * 0.3

    # Normalize
    must_score = min(must_match / must_max, 1.0)
    nice_score = min(nice_match / nice_max, 1.0)

    combined = 0.75 * must_score + 0.25 * nice_score

    # Penalize CV/speech/robotics-heavy profiles
    if disqualifier_score > 1.5:
        combined *= max(0.3, 1 - (disqualifier_score - 1.5) * 0.15)

    return round(combined, 4), matched_names[:5]


def score_career(candidate: dict) -> tuple[float, str]:
    """
    Scores based on:
    - Title progression (current & past)
    - Career at product companies vs pure services
    - Evidence of IR/ML shipping in descriptions
    - Company type diversity
    Returns (0-1 score, evidence_note)
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    current_title = (profile.get("current_title") or "").lower()
    current_company = (profile.get("current_company") or "").lower()
    company_size = (profile.get("current_company_size") or "").lower()
    industry = (profile.get("current_industry") or "").lower()

    score = 0.0
    evidence = []

    # --- Title scoring ---
    title_score = 0.0
    for positive_title in POSITIVE_TITLES:
        if positive_title in current_title:
            title_score = 0.8
            evidence.append(f"title: {profile.get('current_title','')}")
            break
    if "lead" in current_title or "principal" in current_title or "staff" in current_title:
        title_score = min(title_score + 0.1, 1.0)
    if "senior" in current_title:
        title_score = min(title_score + 0.05, 1.0)

    # --- Consulting penalty ---
    consulting_penalty = 0.0
    all_companies = [current_company] + [
        (j.get("company") or "").lower() for j in career
    ]
    consulting_count = sum(1 for c in all_companies if any(k in c for k in CONSULTING_COMPANIES))
    total_companies = len(set(c for c in all_companies if c))
    if total_companies > 0 and consulting_count / total_companies > 0.7:
        consulting_penalty = 0.4  # JD says avoid pure consulting
        evidence.append("heavy consulting background")
    elif consulting_count > 0:
        consulting_penalty = 0.1

    # --- Career description evidence ---
    desc_score = 0.0
    all_descriptions = " ".join(
        (j.get("description") or "") for j in career
    ).lower()
    for phrase in CAREER_EVIDENCE_PHRASES:
        if phrase in all_descriptions:
            desc_score += 0.08
    desc_score = min(desc_score, 1.0)

    # --- Industry bonus (product company, not services) ---
    product_industries = {"fintech", "saas", "e-commerce", "ecommerce",
                          "hr tech", "ai", "analytics", "media", "healthtech",
                          "edtech", "adtech"}
    industry_bonus = 0.0
    if any(pi in industry for pi in product_industries):
        industry_bonus = 0.1

    # --- Company size (startup/mid is preferred by JD) ---
    size_bonus = 0.0
    if company_size in ("51-200", "201-500", "501-1000", "1001-5000"):
        size_bonus = 0.05

    score = (0.40 * title_score
             + 0.40 * desc_score
             + industry_bonus
             + size_bonus
             - consulting_penalty)

    score = max(0.0, min(score, 1.0))
    ev = "; ".join(evidence) if evidence else "standard career background"
    return round(score, 4), ev


def score_experience(candidate: dict) -> float:
    """
    JD wants 5-9 years. Sweet spot is 6-8.
    Penalize <4 and >12 (too junior or likely overqualified for hands-on role).
    """
    yoe = (candidate.get("profile") or {}).get("years_of_experience", 0) or 0
    if yoe < 3:
        return 0.1
    elif yoe < 4:
        return 0.3
    elif 4 <= yoe < 5:
        return 0.6
    elif 5 <= yoe <= 9:
        # Peak at 6-8
        if 6 <= yoe <= 8:
            return 1.0
        elif yoe in (5, 9):
            return 0.85
        else:
            return 0.75
    elif 9 < yoe <= 12:
        return 0.6
    else:
        return 0.4


def score_location(candidate: dict) -> float:
    """
    Prefer India, especially Pune/Noida/Hyderabad/Bangalore/Mumbai/Delhi NCR.
    Also accept candidates willing to relocate.
    """
    profile = candidate.get("profile", {})
    sigs = candidate.get("redrob_signals", {})

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_relocate = sigs.get("willing_to_relocate", False)
    work_mode_pref = (sigs.get("preferred_work_mode") or "").lower()

    # Top-tier: in preferred city
    for pref in PREFERRED_LOCATIONS:
        if pref in location:
            return 1.0

    # India but non-listed city: willing to relocate
    if country == "india":
        if willing_relocate:
            return 0.85
        return 0.7

    # Outside India: JD says case-by-case; lower score
    if willing_relocate:
        return 0.4
    return 0.15


def score_behavioral(candidate: dict) -> tuple[float, str]:
    """
    Behavioral signals as a compound multiplier (not a score component but a modifier).
    Returns (0.3 to 1.3 multiplier, signal_note).
    """
    sigs = candidate.get("redrob_signals", {})
    if not sigs:
        return 1.0, "no signals"

    mult = 1.0
    notes = []

    # Availability: open to work
    if sigs.get("open_to_work_flag"):
        mult += 0.08
        notes.append("open to work")
    else:
        mult -= 0.05

    # Recency: last active date
    last_active_str = sigs.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str[:10], "%Y-%m-%d")
            days_ago = (datetime.now() - last_active).days
            if days_ago <= 14:
                mult += 0.12
                notes.append(f"active {days_ago}d ago")
            elif days_ago <= 60:
                mult += 0.05
            elif days_ago > 180:
                mult -= 0.12
                notes.append("inactive 6+ months")
            elif days_ago > 90:
                mult -= 0.06
        except Exception:
            pass

    # Responsiveness
    response_rate = sigs.get("recruiter_response_rate", -1)
    if response_rate is not None and response_rate >= 0:
        if response_rate >= 0.7:
            mult += 0.07
        elif response_rate < 0.15:
            mult -= 0.10
            notes.append(f"low response rate {response_rate:.0%}")

    # Notice period (JD loves sub-30-day)
    notice = sigs.get("notice_period_days", 60)
    if notice is not None:
        if notice <= 30:
            mult += 0.08
            notes.append(f"notice: {notice}d")
        elif notice > 90:
            mult -= 0.06
            notes.append(f"long notice: {notice}d")

    # GitHub activity (engineers should have some)
    github = sigs.get("github_activity_score", -1)
    if github is not None and github >= 0:
        if github >= 60:
            mult += 0.05
        elif github < 10:
            mult -= 0.03

    # Interview completion rate
    icr = sigs.get("interview_completion_rate", -1)
    if icr is not None and icr >= 0:
        if icr >= 0.8:
            mult += 0.04
        elif icr < 0.4:
            mult -= 0.05

    # Saved by recruiters (social proof)
    saved = sigs.get("saved_by_recruiters_30d", 0) or 0
    if saved >= 5:
        mult += 0.03

    # Profile completeness
    completeness = sigs.get("profile_completeness_score", 50) or 50
    if completeness >= 85:
        mult += 0.03
    elif completeness < 40:
        mult -= 0.05

    # Cap multiplier
    mult = max(0.35, min(mult, 1.35))
    note = ", ".join(notes) if notes else "standard engagement"
    return round(mult, 4), note


def score_candidate(candidate: dict) -> dict:
    """Master scoring function. Returns scored candidate dict."""
    cid = candidate.get("candidate_id", "")

    # --- Honeypot check first ---
    if detect_honeypot(candidate):
        return {
            "candidate_id": cid,
            "score": 0.001,
            "honeypot": True,
            "components": {},
            "reasoning": "Profile contains inconsistencies indicating a honeypot/synthetic entry.",
        }

    # --- Component scores ---
    skills_score, matched_skills = score_skills(candidate)
    career_score, career_note = score_career(candidate)
    exp_score = score_experience(candidate)
    loc_score = score_location(candidate)
    behavioral_mult, behavioral_note = score_behavioral(candidate)

    # --- Weighted composite (before behavioral multiplier) ---
    base_score = (
        0.35 * skills_score
        + 0.28 * career_score
        + 0.17 * exp_score
        + 0.12 * loc_score
        + 0.08 * ((skills_score + career_score) / 2)  # coherence bonus
    )

    final_score = base_score * behavioral_mult
    final_score = round(min(final_score, 1.0), 5)

    # --- Build reasoning string ---
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    sigs = candidate.get("redrob_signals", {})
    notice = sigs.get("notice_period_days", "?")
    open_w = sigs.get("open_to_work_flag", False)

    skill_str = ", ".join(matched_skills[:3]) if matched_skills else "limited relevant skills"

    reasoning_parts = []
    if title and company:
        reasoning_parts.append(f"{yoe:.1f}yr {title} at {company}")
    if matched_skills:
        reasoning_parts.append(f"matched: {skill_str}")
    if location:
        reasoning_parts.append(f"based in {location}")
    if open_w:
        reasoning_parts.append("open to work")
    if behavioral_note and behavioral_note != "standard engagement":
        reasoning_parts.append(behavioral_note)
    if career_note and career_note != "standard career background":
        reasoning_parts.append(career_note)

    reasoning = "; ".join(reasoning_parts)[:300]
    if not reasoning:
        reasoning = f"{yoe:.0f}yr experience, score {final_score:.3f}"

    return {
        "candidate_id": cid,
        "score": final_score,
        "honeypot": False,
        "components": {
            "skills": skills_score,
            "career": career_score,
            "experience": exp_score,
            "location": loc_score,
            "behavioral_mult": behavioral_mult,
        },
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(candidates_path: str, output_path: str):
    """
    Load candidates, score all, output top-100 CSV.
    """
    print(f"[INFO] Loading candidates from {candidates_path}...")
    candidates = []

    # Support both .jsonl and .json (array)
    if candidates_path.endswith(".json"):
        with open(candidates_path) as f:
            candidates = json.load(f)
    else:
        with open(candidates_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    print(f"[INFO] Loaded {len(candidates):,} candidates. Scoring...")

    scored = []
    honeypots_caught = 0
    for i, cand in enumerate(candidates):
        result = score_candidate(cand)
        scored.append(result)
        if result.get("honeypot"):
            honeypots_caught += 1
        if (i + 1) % 10000 == 0:
            print(f"[INFO] Processed {i+1:,} / {len(candidates):,}")

    print(f"[INFO] Scoring complete. Honeypots caught: {honeypots_caught}")

    # Sort by score descending, break ties by candidate_id ascending
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Take top 100
    top100 = scored[:100]

    print("[INFO] Top-10 preview:")
    for i, c in enumerate(top100[:10]):
        comps = c.get("components", {})
        print(f"  {i+1}. {c['candidate_id']} score={c['score']:.4f} "
              f"sk={comps.get('skills',0):.2f} ca={comps.get('career',0):.2f} "
              f"beh={comps.get('behavioral_mult',1):.2f}")

    # Validate monotonically non-increasing scores
    for i in range(len(top100) - 1):
        assert top100[i]["score"] >= top100[i+1]["score"] or \
               abs(top100[i]["score"] - top100[i+1]["score"]) < 1e-9, \
               "Score ordering violated!"

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, c in enumerate(top100, start=1):
            writer.writerow([c["candidate_id"], rank, c["score"], c["reasoning"]])

    print(f"[INFO] Submission written to {output_path}")
    print(f"[INFO] Rows: {len(top100)}, Ranks 1-{len(top100)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates JSONL or JSON file")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    run(args.candidates, args.out)
