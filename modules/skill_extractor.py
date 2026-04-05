"""
Skill Extraction Module — Domain-Independent

Extraction pipeline:
1. DB-based matching   (optional, strong signal)
2. NLP noun-phrase extraction via spaCy
3. Frequency + linguistic filtering (no domain hardcoding)
4. Pattern matching for special tokens (C++, C#, .NET …)
5. Clean normalisation
"""

import json
import re
import os
from collections import Counter
from modules.preprocessor import extract_noun_phrases


# ─────────────────────────────────────────────
# LOAD SKILL DB (OPTIONAL)
# ─────────────────────────────────────────────
def load_skill_db() -> dict:
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'skill_db.json')
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# GENERIC STOPWORDS (NO DOMAIN TERMS)
# ─────────────────────────────────────────────
STOPWORDS = {
    "and", "or", "the", "a", "an", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "this", "that", "these", "those", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "not", "but", "if", "so", "as", "at", "from",
    "we", "you", "they", "it", "our", "your", "their", "its",
}

# Tokens that look like skills but are generic filler words
FILLER_TOKENS = {
    "ability", "skill", "skills", "knowledge", "experience", "understanding",
    "familiar", "familiarity", "proficiency", "working", "strong", "good",
    "excellent", "great", "solid", "hands", "plus", "bonus", "etc", "years",
    "year", "minimum", "least", "preferred", "required", "desired", "relevant",
    "related", "equivalent", "position", "role", "team", "company", "job",
    "work", "project", "system", "tool", "tools", "use", "using", "used",
    "new", "high", "best", "well", "level", "basic", "advanced", "senior",
    "junior", "lead", "own", "own", "candidate", "applicant",
}


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def is_valid_skill(phrase: str, freq: int) -> bool:
    phrase = phrase.lower().strip()
    words = phrase.split()

    if not (1 <= len(words) <= 4):
        return False

    if all(w in STOPWORDS for w in words):
        return False

    if any(w in FILLER_TOKENS for w in words):
        return False

    # Reject pure verb forms
    if len(words) == 1 and phrase.endswith(("ing", "tion", "ness", "ment")):
        return False

    if len(phrase) < 2:
        return False

    # Multi-word phrases need at least 2 occurrences
    if freq < 2 and len(words) > 1:
        return False

    # Single-letter tokens (except known abbreviations handled by pattern step)
    if len(phrase) == 1:
        return False

    return True


# ─────────────────────────────────────────────
# MAIN EXTRACTION
# ─────────────────────────────────────────────
def extract_skills(text: str, skill_db: dict = None) -> list:
    if not text:
        return []

    raw_text = text  # keep original for pattern matching
    text = text.lower()
    text = re.sub(r'[,\n•|/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    found_skills = set()

    # ── 1. DB-based (optional, strongest signal) ──────────────────────────────
    if skill_db:
        for category, skills in skill_db.items():
            for skill in skills:
                pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'
                if re.search(pattern, text):
                    found_skills.add(_normalize_skill(skill))

    # ── 2. NLP noun-phrase extraction ─────────────────────────────────────────
    phrases = extract_noun_phrases(text)
    phrase_counts = Counter(phrases)

    for phrase in phrases:
        if is_valid_skill(phrase, phrase_counts[phrase]):
            found_skills.add(_normalize_skill(phrase))

    # ── 3. Special token patterns (C++, C#, .NET, F#, etc.) ──────────────────
    special_patterns = re.findall(
        r'\b[a-zA-Z][a-zA-Z0-9]*(?:\+\+|#)\b'   # C++, C#, F#
        r'|\.NET\b'                                # .NET
        r'|\bASP\.NET\b'                           # ASP.NET
        r'|\bNode\.js\b'                           # Node.js
        r'|\bVue\.js\b'                            # Vue.js
        r'|\bReact\.js\b'                          # React.js
        r'|\bNext\.js\b',                          # Next.js
        raw_text,
        re.IGNORECASE,
    )
    for skill in special_patterns:
        found_skills.add(_normalize_skill(skill))

    # ── 4. Deduplicate substrings (remove "java" if "javascript" also present)
    found_skills = _remove_substring_duplicates(found_skills)

    return sorted(found_skills)


# ─────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────

# Canonical aliases — maps common variants to a preferred form
_ALIASES = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "ml": "Machine Learning",
    "ai": "AI",
    "dl": "Deep Learning",
    "nlp": "NLP",
    "cv": "Computer Vision",
    "oop": "OOP",
    "api": "API",
    "rest": "REST",
    "sql": "SQL",
    "nosql": "NoSQL",
    "aws": "AWS",
    "gcp": "GCP",
    "ci": "CI/CD",
    "cd": "CI/CD",
    "cicd": "CI/CD",
    "k8s": "Kubernetes",
    "nodejs": "Node.js",
    "reactjs": "React",
    "vuejs": "Vue.js",
    "angularjs": "Angular",
    "nextjs": "Next.js",
    "golang": "Go",
    "dotnet": ".NET",
    "dotnetcore": ".NET Core",
}


def _normalize_skill(skill: str) -> str:
    skill_clean = skill.lower().strip()
    skill_clean = re.sub(r'[^\w\+\#\. ]', '', skill_clean).strip()
    skill_clean = re.sub(r'\s+', ' ', skill_clean)

    # Check alias map first
    if skill_clean in _ALIASES:
        return _ALIASES[skill_clean]

    # Preserve well-known special tokens
    lower = skill_clean
    if "c++" in lower:
        return "C++"
    if "c#" in lower:
        return "C#"
    if "f#" in lower:
        return "F#"
    if ".net" in lower or lower == "dotnet":
        return ".NET"
    if "node.js" in lower or lower == "nodejs":
        return "Node.js"

    # Normalise version suffixes (python3 → Python)
    skill_clean = re.sub(r'(\b[a-z]+)\d+\b', r'\1', skill_clean)

    return skill_clean.title()


def _remove_substring_duplicates(skills: set) -> set:
    """
    Remove a skill if a longer skill that contains it is already present.
    E.g. if both "java" and "javascript" are found, keep "JavaScript" only.
    """
    skills_lower = {s.lower(): s for s in skills}
    to_remove = set()
    for s in skills_lower:
        for other in skills_lower:
            if s != other and s in other and len(s) <= len(other) - 2:
                to_remove.add(s)
    return {v for k, v in skills_lower.items() if k not in to_remove}


# ─────────────────────────────────────────────
# CATEGORY LOOKUP (OPTIONAL)
# ─────────────────────────────────────────────
def get_skill_category(skill: str, skill_db: dict) -> str:
    if not skill_db:
        return "other"
    skill_lower = skill.lower()
    for category, skills in skill_db.items():
        for s in skills:
            if s.lower() in skill_lower or skill_lower in s.lower():
                return category
    return "other"