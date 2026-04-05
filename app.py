import streamlit as st
import os
import json
import pandas as pd

from modules.resume_parser import parse_resume
from modules.preprocessor import preprocess_text
from modules.skill_extractor import extract_skills, load_skill_db
from modules.jd_processor import process_job_description
from modules.matcher import compute_match_score, detailed_section_scores, classify_skills
from modules.recommender import generate_recommendations, get_llm_suggestions
from modules.association_miner import mine_skill_associations
from modules.visualizer import (
    plot_match_gauge,
    plot_skill_gap_chart,
    plot_section_scores,
    plot_score_breakdown,
    plot_skill_wordcloud,
    plot_association_network,
    plot_skill_categories,
)

from modules.llm_skill_extractor import extract_skills_from_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SKILLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #08090d;
    --surface:   #0f1117;
    --border:    #1c2030;
    --border2:   #252b3b;
    --text:      #d4d8e2;
    --muted:     #5a6278;
    --blue:      #4d7cfe;
    --blue-dim:  #1a2a55;
    --red:       #e05c5c;
    --red-dim:   #2a1414;
    --green:     #3ecf8e;
    --green-dim: #0d2820;
    --amber:     #f5a623;
    --purple:    #a78bfa;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
}

div[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
div[data-testid="stSidebar"] * { color: var(--text); }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; }

/* ── Wordmark ── */
.wordmark {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
}
.wordmark span { color: var(--blue); }
.wordmark-sub {
    font-family: var(--mono);
    font-size: 0.63rem;
    color: var(--muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── KPI Row ── */
.kpi-row { display: flex; gap: 10px; margin-bottom: 1.4rem; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 100px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}
.kpi .val {
    font-family: var(--mono);
    font-size: 1.75rem;
    font-weight: 600;
    line-height: 1;
    color: var(--blue);
}
.kpi .val.red  { color: var(--red); }
.kpi .val.grn  { color: var(--green); }
.kpi .val.amb  { color: var(--amber); }
.kpi .lbl {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
    font-family: var(--mono);
}

/* ── Section header ── */
.sh {
    font-family: var(--mono);
    font-size: 0.66rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
    margin: 1.1rem 0 0.7rem 0;
}

/* ── Skill tags ── */
.tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.71rem;
    padding: 2px 9px;
    border-radius: 3px;
    margin: 2px;
    background: var(--blue-dim);
    color: var(--blue);
    border: 1px solid #1f3470;
}
.tag.miss  { background: var(--red-dim);   color: var(--red);   border-color: #3d1a1a; }
.tag.match { background: var(--green-dim); color: var(--green); border-color: #1a3d2a; }

/* ── Recommendation cards ── */
.rec {
    background: var(--surface);
    border-left: 2px solid var(--border2);
    padding: 0.7rem 1rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 6px;
    font-size: 0.84rem;
    line-height: 1.65;
    color: var(--text);
}
.rec.hi  { border-color: var(--red); }
.rec.med { border-color: var(--amber); }
.rec.low { border-color: var(--green); }
.rec .badge {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 1px 6px;
    border-radius: 2px;
    margin-right: 6px;
}
.rec .badge.hi  { background: var(--red-dim);   color: var(--red); }
.rec .badge.med { background: #2a1e08;           color: var(--amber); }
.rec .badge.low { background: var(--green-dim);  color: var(--green); }

/* ── AI block ── */
.ai-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.87rem;
    line-height: 1.75;
    color: var(--text);
    white-space: pre-wrap;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--blue) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom: 2px solid var(--blue) !important;
}

div[data-testid="stFileUploader"] {
    border: 1px dashed var(--border2) !important;
    border-radius: 6px !important;
    background: var(--surface) !important;
}
textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.84rem !important;
}
textarea:focus { border-color: var(--blue) !important; }

hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

/* ── Hero ── */
.hero-container {
    max-width: 540px;
    margin: 4rem auto;
    text-align: center;
}
.hero-title {
    font-family: var(--mono);
    font-size: 3rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.04em;
    margin-bottom: 0.5rem;
}
.hero-title span { color: var(--blue); }
.hero-sub {
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-family: var(--mono);
    margin-bottom: 2.5rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    text-align: left;
    margin-top: 1.5rem;
}
.feature {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}
.feature .f-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--blue);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.feature .f-desc { font-size: 0.77rem; color: var(--muted); line-height: 1.4; }
.hint { font-size: 0.73rem; color: var(--muted); margin-top: 1.5rem; font-family: var(--mono); }

/* Hide Plotly toolbar */
.js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached computations ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_load_skill_db():
    return load_skill_db()


@st.cache_data(show_spinner=False)
def cached_parse_resume(file_bytes: bytes, filename: str) -> str:
    class _FakeFile:
        def __init__(self, b, n):
            self._b, self.name = b, n
        def read(self):
            return self._b
    return parse_resume(_FakeFile(file_bytes, filename))


@st.cache_data(show_spinner=False)
def cached_process_jd(jd_text: str) -> dict:
    return process_job_description(jd_text)


@st.cache_data(show_spinner=False)
def cached_llm_skills(text: str) -> list:
    return extract_skills_from_text(text)


@st.cache_data(show_spinner=False)
def cached_extract_skills(text: str, skill_db_json: str) -> list:
    return extract_skills(text, json.loads(skill_db_json))


@st.cache_data(show_spinner=False)
def cached_match(resume_text, jd_text, resume_skills_tuple, jd_skills_tuple):
    return compute_match_score(
        resume_text, jd_text,
        list(resume_skills_tuple), list(jd_skills_tuple),
    )


@st.cache_data(show_spinner=False)
def cached_section_scores(resume_text: str, jd_text: str) -> dict:
    return detailed_section_scores(resume_text, jd_text)


@st.cache_data(show_spinner=False)
def cached_classify(resume_skills_tuple, jd_skills_tuple):
    return classify_skills(list(resume_skills_tuple), list(jd_skills_tuple))


@st.cache_data(show_spinner=False)
def cached_recs(missing_t, matched_t, score, resume_text, jd_skills_t):
    return generate_recommendations(
        list(missing_t), list(matched_t), score, resume_text, list(jd_skills_t)
    )


@st.cache_data(show_spinner=False)
def cached_llm(resume_text, jd_text, missing_t, score):
    return get_llm_suggestions(resume_text, jd_text, list(missing_t), score)


@st.cache_data(show_spinner=False)
def cached_assoc(resume_t, jd_t):
    return mine_skill_associations([list(resume_t), list(jd_t)])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="wordmark">SKILL<span>ens</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="wordmark-sub">Resume Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    _lbl = lambda t: st.markdown(
        f'<div style="font-family:var(--mono);font-size:0.68rem;color:var(--muted);'
        f'text-transform:uppercase;letter-spacing:0.15em;margin:12px 0 5px 0">{t}</div>',
        unsafe_allow_html=True
    )

    _lbl("Resume")
    uploaded_file = st.file_uploader(
        "Upload resume", type=["pdf", "docx"], label_visibility="collapsed"
    )

    _lbl("Job Description")
    jd_input = st.text_area(
        "Job description", height=210,
        placeholder="Paste the full job description…",
        label_visibility="collapsed",
    )

    _lbl("Options")
    use_llm   = st.toggle("AI Suggestions", value=True)
    run_assoc = st.toggle("Skill Association Mining", value=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    analyze_btn = st.button("ANALYZE", use_container_width=True)
    st.markdown(
        '<p style="font-family:var(--mono);font-size:0.6rem;color:var(--muted);'
        'text-align:center;margin-top:1rem">SKILLens v2.0 · Domain-Independent NLP</p>',
        unsafe_allow_html=True,
    )


# ── Hero (shown before analysis) ─────────────────────────────────────────────
if not uploaded_file or not jd_input.strip():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">SKILL<span>ens</span></div>
        <div class="hero-sub">Resume Skill Intelligence</div>
        <div class="feature-grid">
            <div class="feature">
                <div class="f-label">Parse</div>
                <div class="f-desc">PDF and DOCX resume extraction with multi-library cascade</div>
            </div>
            <div class="feature">
                <div class="f-label">Extract</div>
                <div class="f-desc">Domain-independent NLP skill detection via LLM</div>
            </div>
            <div class="feature">
                <div class="f-label">Match</div>
                <div class="f-desc">3-layer scoring: canonical · substring · fuzzy matching</div>
            </div>
            <div class="feature">
                <div class="f-label">Recommend</div>
                <div class="f-desc">Prioritised, actionable improvement suggestions</div>
            </div>
        </div>
        <div class="hint">Upload a resume and paste a job description to begin →</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Analysis pipeline ─────────────────────────────────────────────────────────
if analyze_btn or st.session_state.get("analyzed"):
    st.session_state["analyzed"] = True

    NO_TOOLBAR = {"displayModeBar": False}

    prog = st.progress(0, text="Parsing resume…")

    # Read file bytes once and cache
    file_bytes = uploaded_file.read()
    resume_text = cached_parse_resume(file_bytes, uploaded_file.name)

    prog.progress(15, text="Processing job description…")
    jd_data = cached_process_jd(jd_input)

    prog.progress(30, text="Loading skill database…")
    skill_db = cached_load_skill_db()
    skill_db_json = json.dumps(skill_db)

    prog.progress(45, text="Extracting skills via LLM…")

    # AI-based skill extraction
    resume_skills = cached_llm_skills(resume_text)
    jd_skills     = cached_llm_skills(jd_input)

    prog.progress(60, text="Computing match score…")
    score, tfidf_score = cached_match(
        resume_text, jd_input,
        tuple(resume_skills), tuple(jd_skills),
    )
    section_scores = cached_section_scores(resume_text, jd_input)

    # ── FIXED: Use fuzzy + semantic classify_skills instead of naive set ops ──
    matched_skills, missing_skills, extra_skills = cached_classify(
        tuple(resume_skills), tuple(jd_skills)
    )
    matched_skills = sorted(matched_skills)
    missing_skills = sorted(missing_skills)
    extra_skills   = sorted(extra_skills)

    # Derive component scores for breakdown chart
    skill_component = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
    kw_cov = section_scores.get("Keyword Coverage", 0) * 100

    prog.progress(75, text="Generating recommendations…")
    recs = cached_recs(
        tuple(missing_skills), tuple(matched_skills),
        score, resume_text, tuple(jd_skills),
    )

    llm_text = ""
    if use_llm:
        prog.progress(88, text="Running AI suggestions…")
        llm_text = cached_llm(resume_text, jd_input, tuple(missing_skills), score)

    assoc_rules = None
    if run_assoc and len(resume_skills) >= 3:
        assoc_rules = cached_assoc(tuple(resume_skills), tuple(jd_skills))

    prog.progress(100, text="Done.")
    prog.empty()

    # ── Score colour helper ───────────────────────────────────────────────────
    def _score_class(s):
        return "grn" if s >= 70 else ("amb" if s >= 45 else "red")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi">
            <div class="val {_score_class(score)}">{score:.0f}%</div>
            <div class="lbl">Match Score</div>
        </div>
        <div class="kpi">
            <div class="val {_score_class(tfidf_score)}">{tfidf_score:.0f}%</div>
            <div class="lbl">TF-IDF Sim.</div>
        </div>
        <div class="kpi">
            <div class="val grn">{len(matched_skills)}</div>
            <div class="lbl">Skills Matched</div>
        </div>
        <div class="kpi">
            <div class="val red">{len(missing_skills)}</div>
            <div class="lbl">Skills Missing</div>
        </div>
        <div class="kpi">
            <div class="val">{len(extra_skills)}</div>
            <div class="lbl">Bonus Skills</div>
        </div>
        <div class="kpi">
            <div class="val">{jd_data.get('experience_years', 0)}yr</div>
            <div class="lbl">Exp. Required</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Overview", "Skills", "Visualizations", "Recommendations", "Associations"])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="sh">Match Score Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_match_gauge(score), use_container_width=True, config=NO_TOOLBAR)

        with col2:
            st.markdown('<div class="sh">Section-wise Scores</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_section_scores(section_scores), use_container_width=True, config=NO_TOOLBAR)

        st.markdown('<div class="sh">Score Component Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_score_breakdown(skill_component, tfidf_score, kw_cov),
            use_container_width=True, config=NO_TOOLBAR,
        )

        st.markdown('<div class="sh">Resume Preview</div>', unsafe_allow_html=True)
        with st.expander("View extracted resume text"):
            preview = resume_text[:3000] + ("…" if len(resume_text) > 3000 else "")
            st.text_area("", preview, height=200, label_visibility="collapsed")

        st.markdown('<div class="sh">JD Summary</div>', unsafe_allow_html=True)
        jd_col1, jd_col2, jd_col3 = st.columns(3)
        jd_col1.metric("Word Count", jd_data.get("word_count", 0))
        jd_col2.metric("Experience Required", f"{jd_data.get('experience_years', 0)} yrs")
        jd_col3.metric("Education Terms", len(jd_data.get("education", [])))

    # ── Tab 2: Skills ─────────────────────────────────────────────────────────
    with tabs[1]:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="sh">Matched Skills</div>', unsafe_allow_html=True)
            if matched_skills:
                st.markdown(
                    "".join(f'<span class="tag match">{s}</span>' for s in matched_skills),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No matched skills found.")

        with c2:
            st.markdown('<div class="sh">Missing Skills</div>', unsafe_allow_html=True)
            if missing_skills:
                st.markdown(
                    "".join(f'<span class="tag miss">{s}</span>' for s in missing_skills),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No critical skills missing. 🎉")

        with c3:
            st.markdown('<div class="sh">Bonus Skills</div>', unsafe_allow_html=True)
            if extra_skills:
                st.markdown(
                    "".join(f'<span class="tag">{s}</span>' for s in extra_skills),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No extra skills detected.")

        st.markdown('<div class="sh">Skill Gap Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_skill_gap_chart(matched_skills, missing_skills, extra_skills),
            use_container_width=True, config=NO_TOOLBAR,
        )

        st.markdown('<div class="sh">All Extracted Skills</div>', unsafe_allow_html=True)
        col_r, col_j = st.columns(2)
        with col_r:
            st.caption(f"Resume — {len(resume_skills)} skills")
            st.dataframe(
                pd.DataFrame({"Skill": sorted(resume_skills)}),
                use_container_width=True, hide_index=True,
            )
        with col_j:
            st.caption(f"Job Description — {len(jd_skills)} skills")
            st.dataframe(
                pd.DataFrame({"Skill": sorted(jd_skills)}),
                use_container_width=True, hide_index=True,
            )

    # ── Tab 3: Visualizations ─────────────────────────────────────────────────
    with tabs[2]:
        v1, v2 = st.columns(2)

        with v1:
            st.markdown('<div class="sh">Resume Skill Word Cloud</div>', unsafe_allow_html=True)
            wc = plot_skill_wordcloud(resume_skills)
            if wc:
                st.pyplot(wc)
            else:
                st.caption("Not enough skills for word cloud.")

        with v2:
            st.markdown('<div class="sh">Skill Category Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(
                plot_skill_categories(resume_skills, skill_db),
                use_container_width=True, config=NO_TOOLBAR,
            )

        st.markdown('<div class="sh">Skill Overlap Summary</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({
                "Category":   ["Matched", "Missing from Resume", "Extra in Resume"],
                "Count":      [len(matched_skills), len(missing_skills), len(extra_skills)],
                "Percentage": [
                    f"{len(matched_skills) / max(len(jd_skills), 1) * 100:.0f}%",
                    f"{len(missing_skills) / max(len(jd_skills), 1) * 100:.0f}%",
                    "-",
                ],
            }),
            use_container_width=True, hide_index=True,
        )

        if jd_data.get("education"):
            st.markdown('<div class="sh">Education Requirements Detected</div>', unsafe_allow_html=True)
            st.markdown(
                "".join(f'<span class="tag">{e}</span>' for e in jd_data["education"]),
                unsafe_allow_html=True,
            )

    # ── Tab 4: Recommendations ────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="sh">Prioritized Recommendations</div>', unsafe_allow_html=True)

        if recs:
            for rec in recs:
                p = rec.get("priority", "medium").lower()
                css_p = "hi" if p == "high" else ("low" if p == "low" else "med")
                label = p.upper()
                st.markdown(
                    f'<div class="rec {css_p}"><span class="badge {css_p}">{label}</span>{rec["text"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No recommendations generated.")

        if use_llm and llm_text:
            st.markdown('<div class="sh">AI-Powered Suggestions</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-block">{llm_text}</div>', unsafe_allow_html=True)

    # ── Tab 5: Associations ───────────────────────────────────────────────────
    with tabs[4]:
        if assoc_rules is not None and not assoc_rules.empty:
            st.markdown('<div class="sh">Skill Association Rules</div>', unsafe_allow_html=True)
            st.dataframe(
                assoc_rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(15),
                use_container_width=True, hide_index=True,
            )
            st.markdown('<div class="sh">Association Network</div>', unsafe_allow_html=True)
            fig_net = plot_association_network(assoc_rules)
            if fig_net:
                st.plotly_chart(fig_net, use_container_width=True, config=NO_TOOLBAR)
        else:
            st.info(
                "Enable 'Skill Association Mining' and ensure at least 3 skills are detected "
                "to view this tab.",
                icon="ℹ️",
            )