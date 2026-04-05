"""
llm_skill_extractor.py — Precise, domain-independent skill extraction via local LLM.
"""

import re
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL NORMALIZATION MAP
# Maps common surface variants → single canonical form
# Add more as you discover edge cases.
# ─────────────────────────────────────────────────────────────────────────────
_ALIAS_MAP = {
    # Python variants
    "python3": "Python", "python 3": "Python", "python2": "Python",
    "python programming": "Python", "programming in python": "Python",

    # C / C++
    "c programming": "C", "c language": "C",
    "c++ programming": "C++", "cplusplus": "C++",

    # Java
    "java programming": "Java", "core java": "Java",

    # JS / TS
    "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript", "node.js": "Node.js",
    "nodejs": "Node.js", "react.js": "React", "reactjs": "React",
    "vue.js": "Vue", "vuejs": "Vue", "next.js": "Next.js", "nextjs": "Next.js",

    # ML / AI
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "artificial intelligence": "AI", "natural language processing": "NLP",
    "computer vision": "Computer Vision", "cv": "Computer Vision",
    "scikit learn": "Scikit-learn", "sklearn": "Scikit-learn",
    "tensorflow": "TensorFlow", "tf": "TensorFlow",
    "pytorch": "PyTorch", "keras": "Keras",

    # Databases
    "mysql": "MySQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "oracle db": "Oracle", "oracle database": "Oracle",
    "mongodb": "MongoDB", "mongo": "MongoDB", "sqlite": "SQLite",
    "database proficiency in oracle and mysql": "Oracle,MySQL",

    # Security
    "cybersecurity": "Cybersecurity", "cyber security": "Cybersecurity",
    "information security": "Information Security", "infosec": "Information Security",
    "network security": "Network Security", "network security analysis": "Network Security",
    "vulnerability assessment": "Vulnerability Assessment",
    "penetration testing": "Penetration Testing", "pen testing": "Penetration Testing",
    "malware analysis": "Malware Analysis", "malware detection": "Malware Detection",
    "cryptography basics": "Cryptography", "cryptography": "Cryptography",

    # Cloud / DevOps
    "amazon web services": "AWS", "aws": "AWS",
    "google cloud platform": "GCP", "gcp": "GCP",
    "microsoft azure": "Azure", "azure": "Azure",
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "ci/cd": "CI/CD", "cicd": "CI/CD", "git": "Git", "github": "GitHub",

    # OS / Networking
    "linux systems administration": "Linux", "linux administration": "Linux",
    "operating systems familiarity": "Operating Systems",
    "operating system": "Operating Systems",
    "wireshark": "Wireshark", "nmap": "Nmap",

    # Frameworks / Tools
    "flask": "Flask", "django": "Django", "fastapi": "FastAPI",
    "spring boot": "Spring Boot", "spring": "Spring",
    "firebase": "Firebase", "sonarqube": "SonarQube",
    "pestudio": "PEStudio", "kali linux": "Kali Linux",

    # IoT / Embedded
    "internet of things": "IoT", "iot": "IoT",
    "esp32": "ESP32", "esp32-cam": "ESP32-CAM", "arduino": "Arduino",

    # Misc
    "secure system building": "Secure Systems",
    "honeypot development": "Honeypot", "honeypot": "Honeypot",
    "adversarial training": "Adversarial Training",
}

_JUNK_WORDS = {
    "responsibilities", "requirements", "qualification", "experience",
    "years", "year", "knowledge", "understanding", "familiarity",
    "ability", "skills", "skill", "proficiency", "basic", "strong",
    "good", "excellent", "demonstrated", "proven", "team", "work",
    "role", "position", "candidate", "applicant", "bachelor", "master",
    "degree", "etc", "including", "such", "as", "various", "multiple",
}


def _normalize_skill(raw: str) -> str:
    """Lowercase, strip punctuation, collapse spaces, look up alias."""
    cleaned = re.sub(r"[^a-z0-9\s\+\#\./\-]", "", raw.lower().strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Direct alias lookup
    if cleaned in _ALIAS_MAP:
        return _ALIAS_MAP[cleaned]

    # Partial alias lookup (if the cleaned text contains a known alias key)
    for key, val in sorted(_ALIAS_MAP.items(), key=lambda x: -len(x[0])):
        if key in cleaned:
            return val

    # Title-case fallback
    return raw.strip().title()


def _split_compound(skill: str) -> list[str]:
    """Handle cases like 'Oracle,MySQL' returned by alias map."""
    return [s.strip() for s in skill.split(",") if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """You are a strict professional skill extraction system.

TASK: Extract ONLY concrete professional skills from the text below.

STRICT RULES:
1. Output ONLY a comma-separated list — no sentences, no bullets, no headings, no explanation.
2. Include: programming languages, frameworks, libraries, tools, platforms, databases, protocols, security concepts, cloud services, hardware/embedded tech, methodologies.
3. Exclude: soft skills (teamwork, communication), generic words (knowledge of, experience with, understanding of), job titles, company names, years of experience, education degrees.
4. Normalize: "Python 3" → "Python", "ML" → "Machine Learning", "K8s" → "Kubernetes".
5. No duplicates. No trailing punctuation. Concise (1–3 words per skill).
6. If a phrase like "Database proficiency in Oracle and MySQL" appears, emit: "Oracle, MySQL".
7. Do NOT wrap output in quotes or brackets.

TEXT:
{text}

OUTPUT (comma-separated skills only):"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills_from_text(text: str) -> list[str]:
    """
    Extract, normalize, and deduplicate skills from free text.
    Falls back to empty list on LLM error.
    """
    prompt = _PROMPT_TEMPLATE.format(text=text[:3500])

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,   # deterministic
                    "top_p": 1.0,
                    "num_predict": 300,
                }
            },
            timeout=120,
        )
        raw_output = response.json().get("response", "").strip()
    except Exception as e:
        print(f"[LLM Error] {e}")
        return []

    # ── Parse raw output ──────────────────────────────────────────────────────
    # Strip any accidental markdown fences
    raw_output = re.sub(r"```.*?```", "", raw_output, flags=re.DOTALL)
    raw_output = re.sub(r"`", "", raw_output)

    # Split on commas or newlines
    raw_skills = re.split(r"[,\n]+", raw_output)

    seen = set()
    final = []

    for raw in raw_skills:
        raw = raw.strip().strip("•-–—*·").strip()
        if not raw or len(raw) < 2:
            continue

        # Drop junk words
        if raw.lower() in _JUNK_WORDS:
            continue

        # Normalize
        normalized_result = _normalize_skill(raw)

        # Handle compound results (e.g. "Oracle,MySQL")
        for skill in _split_compound(normalized_result):
            skill_lower = skill.lower()
            if skill_lower not in seen and len(skill) > 1:
                seen.add(skill_lower)
                final.append(skill)

    return final