# Redrob Intelligent Candidate Ranking — India Runs Hackathon

**Track:** Data & AI Challenge  
**Author:** Prajwal Suresh Mhase  
**Runtime:** 28 seconds · CPU-only · No network · No pip installs

---

## What this does

Ranks 100,000 candidates for the **Senior AI Engineer** role at Redrob AI using a multi-signal hybrid scoring engine.

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
[01] Load JSONL (streaming, memory-efficient)
        │
        ▼
[02] Honeypot Filter
     - Timeline math check (career months vs. stated YoE)
     - Domain-expert overload detection
     - Behavior-profile coherence check
        │
        ▼
[03] 5-Signal Scorer
     ├── Skills Score (35%)    — trust-weighted by endorsements × duration
     ├── Career Score (28%)    — title match + description evidence + consulting penalty
     ├── Experience Score (17%)— peak at 6-8yr, JD-calibrated band
     ├── Location Score (12%)  — India-preferred, Pune/Noida/Hyderabad/Bangalore/Delhi
     └── Coherence Bonus (8%)  — reward skill-career consistency
        │
        ▼
[04] Behavioral Multiplier (0.35 – 1.35×)
     - open_to_work_flag, last_active_date, recruiter_response_rate
     - notice_period_days, github_activity_score, interview_completion_rate
        │
        ▼
[05] Sort descending, take top 100, generate per-candidate reasoning
        │
        ▼
team_infiniity.csv
```

## Run

```bash
python rank.py --candidates candidates.jsonl --out team_infiniity.csv
```

**No pip installs required** — uses only Python stdlib (`json`, `csv`, `datetime`, `argparse`).

## Validate

```bash
python validate_submission.py team_infiniity.csv
# → Submission is valid.
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Trust multiplier on skills | Prevents keyword stuffers — listing "Pinecone" with 0 endorsements and 0 months gains near-zero score |
| Consulting penalty | JD explicitly warns against pure TCS/Infosys/Wipro backgrounds |
| Behavioral as multiplier, not component | An unavailable genius is not a hire; this mirrors real recruiter logic |
| Career description text scan | Verifies claimed skills appear as actual work, not just in skills list |
| Honeypot detection | 5 independent checks; 236 suspicious profiles excluded |

## Files

| File | Purpose |
|---|---|
| `rank.py` | Main ranker — run this |
| `team_infiniity.csv` | Final submission (top-100 ranked candidates) |
| `submission_metadata.yaml` | Portal metadata |
| `validate_submission.py` | Official format validator |
