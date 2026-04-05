"""
Matching & Scoring Module — Domain-Independent, Optimized

Score pipeline:
1. Skill overlap score      (weight 0.55) — fuzzy + semantic matching
2. TF-IDF cosine similarity (weight 0.25)
3. Keyword coverage         (weight 0.20)
"""

import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.preprocessor import preprocess_text


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC EXPANSION MAP
# Maps a canonical skill → set of related surface forms that count as a match.
# Keep entries lowercase. Add rows as needed.
# ─────────────────────────────────────────────────────────────────────────────
_SEMANTIC_MAP: dict[str, set[str]] = {
    "python": {"python programming", "python3", "python 3", "python scripting",
               "python development", "programming in python"},
    "java": {"java programming", "core java", "java development", "java ee", "java se"},
    "c++": {"cplusplus", "c plus plus", "c++ programming"},
    "c": {"c programming", "c language"},
    "javascript": {"js", "javascript programming", "vanilla js", "es6"},
    "typescript": {"ts", "typescript programming"},
    "machine learning": {"ml", "statistical learning", "predictive modeling",
                         "machine learning algorithms"},
    "deep learning": {"dl", "neural networks", "ann", "cnn", "rnn", "lstm",
                      "deep neural networks"},
    "ai": {"artificial intelligence", "intelligent systems"},
    "nlp": {"natural language processing", "text mining", "text analytics"},
    "computer vision": {"cv", "image processing", "image recognition", "object detection"},
    "cybersecurity": {"cyber security", "information security", "infosec", "security"},
    "network security": {"network security analysis", "network protection",
                          "network monitoring", "network defense"},
    "vulnerability assessment": {"vulnerability scanning", "vulnerability management",
                                  "security assessment", "risk assessment"},
    "penetration testing": {"pen testing", "ethical hacking", "pentest", "pentesting",
                             "offensive security"},
    "cryptography": {"cryptography basics", "encryption", "decryption",
                     "public key infrastructure", "pki", "hash functions"},
    "malware detection": {"malware analysis", "malware research", "threat detection",
                           "antivirus", "anti-malware"},
    "linux": {"linux systems administration", "linux administration",
              "linux system administration", "unix", "bash scripting",
              "shell scripting", "linux commands"},
    "operating systems": {"operating systems familiarity", "os concepts",
                           "system administration"},
    "mysql": {"mysql database", "sql", "relational databases"},
    "oracle": {"oracle db", "oracle database", "oracle sql",
               "database proficiency in oracle and mysql"},
    "postgresql": {"postgres", "psql"},
    "mongodb": {"mongo", "nosql", "document database"},
    "aws": {"amazon web services", "amazon aws", "cloud computing"},
    "gcp": {"google cloud", "google cloud platform"},
    "azure": {"microsoft azure", "azure cloud"},
    "docker": {"containerization", "containers"},
    "kubernetes": {"k8s", "container orchestration"},
    "git": {"version control", "github", "gitlab", "bitbucket", "source control"},
    "flask": {"flask framework", "flask api", "flask web"},
    "django": {"django framework", "django rest framework", "drf"},
    "react": {"reactjs", "react.js", "react framework", "react development"},
    "node.js": {"nodejs", "node js", "express.js", "expressjs"},
    "scikit-learn": {"sklearn", "scikit learn", "scikit"},
    "tensorflow": {"tf", "keras", "tensorflow keras"},
    "pytorch": {"torch", "pytorch framework"},
    "firebase": {"google firebase", "firebase realtime"},
    "iot": {"internet of things", "embedded systems", "connected devices"},
    "esp32": {"esp32-cam", "esp 32"},
    "wireshark": {"packet analysis", "network analysis", "packet capture"},
    "nmap": {"network scanning", "port scanning"},
    "kali linux": {"kali", "kali os"},
    "sonarqube": {"sonar", "static code analysis", "code quality"},
    "honeypot": {"honeypot development", "honeypot systems", "cyber deception"},
    "adversarial training": {"adversarial attacks", "adversarial ml",
                              "adversarial examples", "model robustness"},
    "secure systems": {"secure system building", "secure coding",
                        "secure software development"},
}

# Build reverse lookup: surface form → canonical
_SURFACE_TO_CANONICAL: dict[str, str] = {}
for canonical, surfaces in _SEMANTIC_MAP.items():
    _SURFACE_TO_CANONICAL[canonical] = canonical
    for surface in surfaces:
        _SURFACE_TO_CANONICAL[surface] = canonical


_GENERIC_STOP = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","by",
    "from","is","are","was","were","be","been","have","has","had","do","does",
    "did","will","would","could","should","may","might","must","shall",
    "we","you","they","it","this","that","which","who","not","your","our",
    "their","as","if","so"
}


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def _canonicalize(skill: str) -> str:
    """Map a skill to its canonical form via the semantic map."""
    norm = _normalize(skill)
    return _SURFACE_TO_CANONICAL.get(norm, norm)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL MATCHING — 3-LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _is_match(skill_a: str, skill_b: str, fuzzy_threshold: float = 0.82) -> bool:
    """
    True if skill_a and skill_b are considered the same skill.

    Layer 1 — Canonical match: both resolve to the same canonical key
    Layer 2 — Substring match: one contains the other (≥4 chars each)
    Layer 3 — Fuzzy ratio: SequenceMatcher ≥ fuzzy_threshold
    """
    a = _normalize(skill_a)
    b = _normalize(skill_b)

    if a == b:
        return True

    # Layer 1: canonical
    ca, cb = _canonicalize(a), _canonicalize(b)
    if ca == cb:
        return True

    # Layer 2: substring (avoid matching tiny tokens like "c" in "cybersecurity")
    if len(a) >= 4 and len(b) >= 4:
        if a in b or b in a:
            return True

    # Layer 3: fuzzy
    if len(a) >= 4 and len(b) >= 4:
        if _fuzzy_ratio(a, b) >= fuzzy_threshold:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# SKILL OVERLAP SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _skill_overlap_score(resume_skills: list, jd_skills: list) -> float:
    if not jd_skills:
        return 0.0

    matched_jd = set()
    for jd_skill in jd_skills:
        for res_skill in resume_skills:
            if _is_match(res_skill, jd_skill):
                matched_jd.add(_normalize(jd_skill))
                break

    return (len(matched_jd) / len(jd_skills)) * 100


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: SKILL MATCH CLASSIFICATION
# Returns matched, missing, extra as normalized lists
# ─────────────────────────────────────────────────────────────────────────────

def classify_skills(
    resume_skills: list, jd_skills: list
) -> tuple[list, list, list]:
    """
    Returns (matched_jd_skills, missing_jd_skills, extra_resume_skills)
    All lists use the JD/resume original casing for display.
    """
    matched_jd = []
    missing_jd = []

    for jd_skill in jd_skills:
        found = any(_is_match(res_skill, jd_skill) for res_skill in resume_skills)
        if found:
            matched_jd.append(jd_skill)
        else:
            missing_jd.append(jd_skill)

    extra_resume = [
        res_skill for res_skill in resume_skills
        if not any(_is_match(res_skill, jd_skill) for jd_skill in jd_skills)
    ]

    return matched_jd, missing_jd, extra_resume


# ─────────────────────────────────────────────────────────────────────────────
# TEXT FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def filter_relevant_text(resume_text: str, jd_skills: list) -> str:
    relevant = []
    for sentence in resume_text.split('.'):
        for skill in jd_skills:
            if _is_match(skill, sentence):
                relevant.append(sentence)
                break
    return ". ".join(relevant)


def find_irrelevant_sentences(resume_text: str, jd_skills: list) -> list:
    irrelevant = []
    for sentence in resume_text.split('.'):
        sentence_clean = sentence.strip()
        if len(sentence_clean) < 20:
            continue
        if not any(_is_match(skill, sentence_clean) for skill in jd_skills):
            irrelevant.append(sentence_clean)
    return irrelevant[:5]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF COSINE
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_cosine(text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
        )
        mat = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD COVERAGE
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_coverage(resume_text: str, jd_text: str) -> float:
    jd_words = set(re.sub(r'[^a-z\s]', '', jd_text.lower()).split()) - _GENERIC_STOP
    resume_words = set(re.sub(r'[^a-z\s]', '', resume_text.lower()).split())

    if not jd_words:
        return 0.5

    covered = jd_words & resume_words
    return min(0.98, len(covered) / len(jd_words))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_match_score(
    resume_text: str,
    jd_text: str,
    resume_skills: list,
    jd_skills: list,
) -> tuple[float, float]:
    """
    Returns (final_score, tfidf_score) — both 0–99.
    """
    skill_score = _skill_overlap_score(resume_skills, jd_skills)

    filtered_resume = filter_relevant_text(resume_text, jd_skills)
    if not filtered_resume.strip():
        filtered_resume = resume_text

    clean_resume = preprocess_text(filtered_resume)
    clean_jd     = preprocess_text(jd_text)

    tfidf_score = _tfidf_cosine(clean_resume, clean_jd) * 100
    kw_score    = _keyword_coverage(resume_text, jd_text) * 100

    if jd_skills:
        w_skill, w_tfidf, w_kw = 0.55, 0.25, 0.20
    else:
        w_skill, w_tfidf, w_kw = 0.0, 0.55, 0.45

    final_score = (
        w_skill * skill_score +
        w_tfidf * tfidf_score +
        w_kw    * kw_score
    )

    return round(min(final_score, 99.0), 1), round(tfidf_score, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION SCORES
# ─────────────────────────────────────────────────────────────────────────────

def _extract_section_text(text: str, section_type: str) -> str:
    patterns = {
        "skills":     r"(skills?|competenc(y|ies)|tools?)",
        "experience": r"(experience|employment)",
        "education":  r"(education|degree|qualification)",
    }
    pattern = patterns.get(section_type)
    if not pattern:
        return text

    lines = text.split('\n')
    capture = False
    collected = []

    for line in lines:
        line_lower = line.lower()
        if re.search(pattern, line_lower):
            capture = True
            continue
        if capture:
            if re.search(r"(summary|project|award|language)", line_lower):
                break
            collected.append(line)

    return "\n".join(collected) if collected else text


def _section_score(resume_text: str, jd_text: str, section_type: str) -> float:
    jd_section  = _extract_section_text(jd_text, section_type)
    res_section = _extract_section_text(resume_text, section_type)

    jd_words = [
        w for w in re.sub(r'[^a-z\s]', '', jd_section.lower()).split()
        if w not in _GENERIC_STOP and len(w) > 2
    ]
    if not jd_words:
        return 1.0

    matched = sum(1 for w in jd_words if w in res_section.lower())
    return min(0.95, max(0.10, matched / len(jd_words)))


def detailed_section_scores(resume_text: str, jd_text: str) -> dict:
    return {
        "Technical Skills": _section_score(resume_text, jd_text, "skills"),
        "Experience":       _section_score(resume_text, jd_text, "experience"),
        "Education":        _section_score(resume_text, jd_text, "education"),
        "Keyword Coverage": _keyword_coverage(resume_text, jd_text),
    }