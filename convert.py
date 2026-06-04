"""
convert2.py
───────────
Converts dataset.csv with columns:
  Role, Resume, Decision, Reason_for_decision, Job_Description
into an ML-ready CSV that ml_model.py can train on.

Usage:
    python convert2.py
    python convert2.py --input dataset.csv --output resumes_ml_ready.csv
"""

import re
import os
import argparse
import pandas as pd

# ─────────────────────────────────────────────────────────
# SKILL KEYWORD LIBRARIES
# ─────────────────────────────────────────────────────────

TECH_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "r",
    "html", "css", "react", "node.js", "sql", "machine learning",
    "deep learning", "tensorflow", "pytorch", "scikit-learn", "nlp",
    "natural language processing", "data analysis", "data science",
    "cloud computing", "docker", "devops", "aws", "azure", "gcp",
    "linux", "network security", "cybersecurity", "embedded systems",
    "android", "ios", "flutter", "algorithms", "data structures",
    "microservices", "spring", "etl", "big data", "spark",
    "tableau", "power bi", "excel", "statistics", "data visualization",
    "automation", "selenium", "git", "agile", "scrum", ".net",
    "graphic design", "adobe", "seo", "digital marketing",
    "project management", "financial analysis", "econometrics",
]

SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "problem solving",
    "creativity", "negotiation", "research", "analytical",
    "presentation", "management", "storytelling", "critical thinking",
]

EDUCATION_KEYWORDS = {
    "phd":       ["ph.d", "phd", "doctor of philosophy"],
    "master":    ["master", "m.sc", "msc", "m.s.", "mba", "m.eng"],
    "bachelor":  ["bachelor", "b.sc", "bsc", "b.s.", "b.e.", "b.tech", "b.a."],
    "associate": ["associate", "diploma"],
}

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
    r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
    r"(\d+)\+?\s*yrs?\s*(?:of\s*)?experience",
]


def extract_education(text: str) -> str:
    t = text.lower()
    for level, keywords in EDUCATION_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return level
    return "unknown"


def extract_experience_years(text: str) -> int:
    t = text.lower()
    for pat in EXPERIENCE_PATTERNS:
        m = re.search(pat, t)
        if m:
            return min(int(m.group(1)), 40)
    return 0


def experience_level(years: int) -> str:
    if years == 0:   return "entry level"
    if years <= 2:   return "junior"
    if years <= 5:   return "mid level"
    if years <= 10:  return "senior"
    return "executive"


def extract_skills(text: str):
    t = text.lower()
    tech = [s for s in TECH_SKILLS if s in t]
    soft = [s for s in SOFT_SKILLS if s in t]
    return tech, soft


def clean_text_for_tfidf(text: str) -> str:
    """Keep only letters/numbers, collapse whitespace — used as TF-IDF input."""
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def convert(input_path: str, output_path: str):
    print(f"\n  📂  Loading: {input_path}")
    df = pd.read_csv(input_path)
    # Strip column name whitespace
    df.columns = df.columns.str.strip()
    print(f"  ✅  {len(df)} rows loaded")
    print(f"  📋  Columns: {df.columns.tolist()}")

    # Validate required columns
    required = {"Role", "Resume"}
    missing  = required - set(df.columns)
    if missing:
        print(f"\n  ❌  Missing required columns: {missing}")
        print(f"  Found columns: {df.columns.tolist()}\n")
        return

    rows = []
    for _, row in df.iterrows():
        role       = str(row.get("Role",               "") or "").strip()
        resume     = str(row.get("Resume",             "") or "")
        decision   = str(row.get("Decision",           "") or "").strip().lower()
        reason     = str(row.get("Reason_for_decision","") or "")
        job_desc   = str(row.get("Job_Description",    "") or "")

        # All text combined for deep feature extraction
        full_text  = f"{resume} {reason}"

        edu_level  = extract_education(full_text)
        exp_years  = extract_experience_years(full_text)
        exp_lvl    = experience_level(exp_years)
        tech, soft = extract_skills(full_text)

        # Binary: was candidate selected?
        hired = 1 if decision in ("selected", "hired", "accepted", "yes", "1", "true") else 0

        # Clean resume text for TF-IDF (stored as space-separated)
        resume_clean = clean_text_for_tfidf(resume[:2000])  # cap at 2000 chars

        rows.append({
            # Identity
            "source_file":            "dataset.csv",
            "role":                   role,

            # ML Numeric Features
            "total_experience_years": exp_years,
            "experience_level":       exp_lvl,
            "num_technical_skills":   len(tech),
            "num_soft_skills":        len(soft),
            "hired":                  hired,
            "resume_length":          len(resume.split()),
            "education_level":        edu_level,

            # Text features for TF-IDF (space-separated tokens)
            "technical_skills":       " ".join(tech),
            "soft_skills":            " ".join(soft),
            "resume_text":            resume_clean,

            # Career label — directly from Role column
            "career_category":        role,
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n  ✅  Saved → {output_path}")
    print(f"  📊  {len(out)} rows | {len(out.columns)} columns")
    print(f"\n  🎯  Career Category Breakdown:")
    print(out["career_category"].value_counts().head(20).to_string())
    print(f"\n  🎓  Education Breakdown:")
    print(out["education_level"].value_counts().to_string())
    print(f"\n  💼  Experience Level Breakdown:")
    print(out["experience_level"].value_counts().to_string())
    print(f"\n  📋  Hired/Not-Hired Breakdown:")
    print(out["hired"].value_counts().to_string())
    print(f"\n  🎉  Done! Now run:")
    print(f"      python ml_model.py {output_path} --save model.pkl\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset.csv",          help="Input CSV")
    parser.add_argument("--output", default="resumes_ml_ready.csv", help="Output CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"\n  ❌  File not found: {args.input}")
        print(f"  Save your dataset as '{args.input}' in the same folder.\n")
        return

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
