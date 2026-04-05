"""
Job Description Processing Module — Domain-Independent

Extracts structured information from any job description:
- Sections (requirements, responsibilities, benefits, about)
- Experience requirements
- Education requirements (dynamically detected, no field hardcoding)
- Clean text for NLP
"""
import re
from modules.preprocessor import preprocess_text


# ─────────────────────────────────────────────
# SECTION HEADER PATTERNS (generic)
# ─────────────────────────────────────────────
SECTION_PATTERNS = {
    "requirements": [
        r"(requirements?|qualifications?|what you.ll need|must.have|required skills?|we.re looking for)",
    ],
    "responsibilities": [
        r"(responsibilities|what you.ll do|role overview|job duties|about the role|key responsibilities)",
    ],
    "benefits": [
        r"(benefits?|perks?|what we offer|compensation|why join)",
    ],
    "about": [
        r"(about us|about the company|who we are|company overview|our mission)",
    ],
}

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s+of\s+experience",
    r"(\d+)\+?\s*years?\s+experience",
    r"experience\s+of\s+(\d+)\+?\s*years?",
    r"minimum\s+(\d+)\s+years?",
    r"at\s+least\s+(\d+)\s+years?",
    r"(\d+)\+\s*yrs?\b",
]

# Universal degree-level markers — NO field names here
EDUCATION_KEYWORDS = [
    # Degree levels (universal)
    "bachelor", "master", "phd", "doctorate", "degree",
    "undergraduate", "postgraduate", "diploma", "associate",
    # Generic credential markers
    "gpa", "honours", "honors", "thesis", "dissertation",
    # Common field-neutral abbreviations
    "mba", "msc", "bsc", "ma", "ba",
]
# NOTE: IT-specific codes (b.tech, m.tech) and field names
# (computer science, engineering, etc.) are intentionally excluded.
# Field names are discovered dynamically from context in _extract_education().


# ─────────────────────────────────────────────
# MAIN PROCESSOR
# ─────────────────────────────────────────────
def process_job_description(jd_text: str) -> dict:
    """Process raw JD text and return structured data."""
    return {
        "raw_text":        jd_text,
        "clean_text":      preprocess_text(jd_text),
        "sections":        _extract_sections(jd_text),
        "experience_years": _extract_experience(jd_text),
        "education":       _extract_education(jd_text),
        "word_count":      len(jd_text.split()),
    }


def _extract_sections(text: str) -> dict:
    """Split JD into logical sections using generic header patterns."""
    sections = {}
    lines = text.split('\n')
    current_section = "general"
    current_content = []

    for line in lines:
        line_lower = line.lower().strip()
        matched_section = None

        for section_name, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    matched_section = section_name
                    break
            if matched_section:
                break

        if matched_section:
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = matched_section
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def _extract_experience(text: str) -> int:
    """Extract minimum years of experience required."""
    years_found = []
    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                years_found.append(int(match))
            except (ValueError, TypeError):
                pass
    return min(years_found) if years_found else 0


def _extract_education(text: str) -> list:
    """
    Extract education requirements from JD text.

    Two-pass, domain-independent approach:
    1. Match universal degree-level keywords from EDUCATION_KEYWORDS.
    2. Dynamically extract the study field from surrounding context
       (e.g. "degree in nursing", "master of accounting") — no field
       names are hardcoded.
    """
    text_lower = text.lower()
    found = set()

    # Pass 1 — universal degree markers
    for keyword in EDUCATION_KEYWORDS:
        if keyword in text_lower:
            found.add(keyword)

    # Pass 2 — dynamic field detection via context regex
    # Stops before conjunctions / filler words so "nursing or" → "nursing"
    _STOP = r"(?:or|and|is|are|a|an|the|preferred|required|equivalent|related|relevant|any)"
    field_pattern = re.compile(
        r"(?:degree|bachelor|master|phd|doctorate|diploma|associate)"
        r"(?:'s|s)?\s+(?:in|of)\s+"
        r"(?!" + _STOP + r"\b)"          # first word must not be a stop word
        r"([a-z]+)"                       # first word of the field name
        r"(?:\s+(?!" + _STOP + r"\b)([a-z]+))?"  # optional second word
    )
    for match in field_pattern.finditer(text_lower):
        parts = [match.group(1)]
        if match.group(2):
            parts.append(match.group(2))
        field = " ".join(parts).strip()
        found.add(field)

    return sorted(found)


def extract_bullet_points(text: str) -> list:
    """Extract bullet point items from text."""
    bullets = []
    for line in text.split('\n'):
        line = line.strip()
        if re.match(r'^[•\-\*\+◦▪]\s*', line):
            cleaned = re.sub(r'^[•\-\*\+◦▪]\s*', '', line).strip()
            if cleaned:
                bullets.append(cleaned)
        elif re.match(r'^\d+\.\s+', line):
            cleaned = re.sub(r'^\d+\.\s+', '', line).strip()
            if cleaned:
                bullets.append(cleaned)
    return bullets