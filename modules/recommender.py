"""
Recommendation Engine
Generates prioritised, actionable resume improvement suggestions.
Domain-independent — all recommendations derived from runtime data.
"""
import os
import requests
from modules.matcher import find_irrelevant_sentences


def generate_recommendations(
    missing_skills: list,
    matched_skills: list,
    score: float,
    resume_text: str,
    jd_skills: list,
) -> list:
    """
    Return a prioritised list of recommendation dicts:
      {"text": str, "priority": "high" | "medium" | "low"}
    """
    recs = []

    # ── Critical: missing skills ──────────────────────────────────────────────
    top_missing = missing_skills[:6]
    if top_missing:
        skills_str = ", ".join(top_missing)
        recs.append({
            "text": f"Add these missing skills to your resume (with project or work context): {skills_str}.",
            "priority": "high",
        })

    # ── Overall alignment warning ─────────────────────────────────────────────
    if score < 40:
        recs.append({
            "text": "Your resume has very low alignment with this job. Consider rewriting your summary "
                    "section to reflect the role's core requirements.",
            "priority": "high",
        })
    elif score < 60:
        recs.append({
            "text": "Resume alignment is moderate. Tailor your bullet points to mirror the language "
                    "and priorities used in this job description.",
            "priority": "high",
        })

    # ── Irrelevant content warning ────────────────────────────────────────────
    if jd_skills:
        irrelevant = find_irrelevant_sentences(resume_text, jd_skills)
        if len(irrelevant) >= 3:
            recs.append({
                "text": "Several resume sentences don't relate to this job. "
                        "Replace weak/irrelevant bullet points with examples that showcase required skills.",
                "priority": "medium",
            })

    # ── Highlight matched skills ──────────────────────────────────────────────
    if matched_skills:
        top_matched = ", ".join(sorted(matched_skills)[:5])
        recs.append({
            "text": f"You already have key matching skills ({top_matched}). "
                    "Make sure they appear prominently near the top of your resume.",
            "priority": "medium",
        })

    # ── Skill ratio insight ───────────────────────────────────────────────────
    if jd_skills and len(jd_skills) > 0:
        match_ratio = len(matched_skills) / len(jd_skills)
        if match_ratio < 0.3:
            recs.append({
                "text": f"You match only {len(matched_skills)} of {len(jd_skills)} required skills. "
                        "Consider adding projects or certifications to bridge the gap.",
                "priority": "high",
            })
        elif match_ratio >= 0.7:
            recs.append({
                "text": "Strong skill match! Focus on quantifying your achievements "
                        "(e.g. 'reduced processing time by 30%') to stand out further.",
                "priority": "low",
            })

    # ── Format / length tip ───────────────────────────────────────────────────
    word_count = len(resume_text.split())
    if word_count < 300:
        recs.append({
            "text": "Your resume appears short. Add more detail about your experience, "
                    "projects, and measurable achievements.",
            "priority": "medium",
        })
    elif word_count > 1200:
        recs.append({
            "text": "Your resume is quite long. Consider trimming it to 1–2 pages, "
                    "focusing on the most relevant experience for this role.",
            "priority": "low",
        })

    return recs


# ─────────────────────────────────────────────
# LLM SUGGESTIONS (local Ollama or Anthropic)
# ─────────────────────────────────────────────

def get_llm_suggestions(
    resume_text: str,
    jd_text: str,
    missing_skills: list,
    score: float,
) -> str:
    """
    Try Anthropic API first (if key is set), fall back to local Ollama.
    Returns a plain-text string of suggestions.
    """
    prompt = (
        f"Resume Match Score: {score:.0f}%\n"
        f"Missing Skills: {', '.join(missing_skills[:8])}\n\n"
        "Based on the above, give 4–5 short, specific, actionable resume improvement "
        "tips. Focus on what the candidate should ADD or CHANGE. Be concise."
    )

    # 1. Try Anthropic API
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
        result = _call_anthropic(prompt, anthropic_key)
        if result:
            return result

    # 2. Fallback to local Ollama
    return _call_ollama(prompt)


def _call_anthropic(prompt: str, api_key: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        return ""


def _call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": prompt, "stream": False},
            timeout=120,
        )
        data = response.json()
        return data.get("response", "No suggestions generated.")
    except Exception as e:
        return f"⚠️ AI suggestions unavailable: {str(e)}"