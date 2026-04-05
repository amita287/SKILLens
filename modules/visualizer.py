"""
Visualizer Module
All Plotly charts and matplotlib word cloud for SKILLens.
Dark theme consistent with app design.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_COLOR    = "#08090d"
PAPER_BG    = "#0f1117"
GRID_COLOR  = "#1c2030"
TEXT_COLOR  = "#d4d8e2"
MUTED_COLOR = "#5a6278"
BLUE        = "#4d7cfe"
GREEN       = "#3ecf8e"
RED         = "#e05c5c"
AMBER       = "#f5a623"
PURPLE      = "#a78bfa"
CYAN        = "#22d3ee"

PALETTE = [BLUE, GREEN, PURPLE, AMBER, RED, CYAN,
           "#f472b6", "#34d399", "#fb923c", "#60a5fa"]


def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="IBM Plex Mono, monospace", size=11),
        margin=dict(l=16, r=16, t=44, b=16),
    )
    base.update(kwargs)
    return base


# ── 1. Match Score Gauge ──────────────────────────────────────────────────────
def plot_match_gauge(score: float) -> go.Figure:
    if score >= 75:
        bar_color, status = GREEN, "STRONG MATCH"
    elif score >= 50:
        bar_color, status = AMBER, "MODERATE MATCH"
    else:
        bar_color, status = RED, "WEAK MATCH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 40, "color": bar_color, "family": "IBM Plex Mono"}},
        title={"text": f"<b>{status}</b>", "font": {"color": TEXT_COLOR, "size": 11}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": GRID_COLOR,
                "tickfont": {"color": MUTED_COLOR, "size": 9},
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": BG_COLOR,
            "borderwidth": 1,
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [0,  40], "color": "#1a0a0a"},
                {"range": [40, 70], "color": "#1a150a"},
                {"range": [70, 100], "color": "#0a1a10"},
            ],
            "threshold": {
                "line": {"color": BLUE, "width": 2},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))
    fig.update_layout(**_base_layout(height=260, title_text=""))
    return fig


# ── 2. Skill Gap Bar Chart ────────────────────────────────────────────────────
def plot_skill_gap_chart(matched: list, missing: list, extra: list) -> go.Figure:
    categories, values, colors, labels = [], [], [], []

    for s in sorted(matched)[:14]:
        categories.append(s); values.append(1)
        colors.append(GREEN); labels.append("✓ Matched")

    for s in sorted(missing)[:14]:
        categories.append(s); values.append(-1)
        colors.append(RED); labels.append("✗ Missing")

    for s in sorted(extra)[:8]:
        categories.append(s); values.append(0.55)
        colors.append(BLUE); labels.append("★ Bonus")

    if not categories:
        fig = go.Figure()
        fig.update_layout(**_base_layout(height=200))
        fig.add_annotation(text="No skills detected", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=MUTED_COLOR))
        return fig

    fig = go.Figure(go.Bar(
        x=values, y=categories, orientation='h',
        marker=dict(color=colors, line=dict(color=GRID_COLOR, width=0.4)),
        text=labels, textposition='outside',
        textfont=dict(size=9, color=TEXT_COLOR),
        hovertemplate='<b>%{y}</b><extra></extra>',
    ))

    fig.update_layout(**_base_layout(
        title_text="Skill Gap Analysis",
        height=max(280, len(categories) * 26 + 60),
        xaxis=dict(showticklabels=False, showgrid=False,
                   zeroline=True, zerolinecolor=GRID_COLOR, range=[-1.6, 1.6]),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        bargap=0.32,
    ))
    return fig


# ── 3. Section Score Bar Chart ────────────────────────────────────────────────
def plot_section_scores(section_scores: dict) -> go.Figure:
    categories = list(section_scores.keys())
    values = [round(v * 100, 1) for v in section_scores.values()]

    colors = [GREEN if v >= 70 else (AMBER if v >= 45 else RED) for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=categories, orientation='h',
        marker=dict(color=colors, opacity=0.9, line=dict(color=GRID_COLOR, width=0.4)),
        text=[f"{v:.0f}%" for v in values],
        textposition='inside',
        textfont=dict(size=11, color="white", family="IBM Plex Mono"),
        hovertemplate='<b>%{y}</b>: %{x:.0f}%<extra></extra>',
    ))

    fig.update_layout(**_base_layout(
        title_text="Section-wise Match",
        height=260,
        xaxis=dict(range=[0, 108], showgrid=True, gridcolor=GRID_COLOR,
                   ticksuffix="%", tickfont=dict(size=9)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        showlegend=False, bargap=0.28,
    ))
    return fig


# ── 4. Score Breakdown Stacked Chart ─────────────────────────────────────────
def plot_score_breakdown(skill_score: float, tfidf_score: float, kw_score: float) -> go.Figure:
    """Show the three component scores side by side."""
    labels = ["Skill Match", "TF-IDF Sim.", "Keyword Cov."]
    values = [skill_score, tfidf_score, kw_score]
    colors = [BLUE, GREEN, PURPLE]

    fig = go.Figure()
    for label, val, color in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            name=label, x=[label], y=[val],
            marker_color=color,
            text=[f"{val:.0f}%"], textposition='inside',
            textfont=dict(size=12, color="white", family="IBM Plex Mono"),
            hovertemplate=f'<b>{label}</b>: {val:.1f}%<extra></extra>',
        ))

    fig.update_layout(**_base_layout(
        title_text="Score Breakdown",
        height=220,
        xaxis=dict(showgrid=False),
        yaxis=dict(range=[0, 110], showgrid=True, gridcolor=GRID_COLOR,
                   ticksuffix="%", tickfont=dict(size=9)),
        showlegend=False, bargap=0.4, barmode='group',
    ))
    return fig


# ── 5. Word Cloud ─────────────────────────────────────────────────────────────
def plot_skill_wordcloud(skills: list):
    if not skills or len(skills) < 3:
        return None
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        word_freq = {skill: max(1, len(skill)) for skill in skills}
        wc = WordCloud(
            width=700, height=320,
            background_color=PAPER_BG,
            colormap="Blues",
            max_words=60, prefer_horizontal=0.8,
            collocations=False, min_font_size=10,
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=PAPER_BG)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    except (ImportError, ValueError):
        return _skill_bar_fallback(skills)


def _skill_bar_fallback(skills: list):
    import matplotlib.pyplot as plt
    display = skills[:15]
    counts = list(range(len(display), 0, -1))
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PAPER_BG)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(display)))
    ax.barh(display, counts, color=colors)
    ax.set_facecolor(PAPER_BG)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    return fig


# ── 6. Association Network ────────────────────────────────────────────────────
def plot_association_network(rules_df) -> go.Figure:
    try:
        if rules_df is None or rules_df.empty:
            return None

        top_rules = rules_df.head(12)
        nodes = set()
        edges = []

        for _, row in top_rules.iterrows():
            ant, con = str(row["antecedents"]), str(row["consequents"])
            nodes.add(ant); nodes.add(con)
            edges.append((ant, con, float(row["lift"])))

        node_list = list(nodes)
        n = len(node_list)
        if n == 0:
            return None

        angles = [2 * np.pi * i / n for i in range(n)]
        nx = {nd: np.cos(a) for nd, a in zip(node_list, angles)}
        ny = {nd: np.sin(a) for nd, a in zip(node_list, angles)}

        edge_traces = [
            go.Scatter(
                x=[nx[a], nx[c], None], y=[ny[a], ny[c], None],
                mode='lines',
                line=dict(width=min(lift * 0.8, 3), color=BLUE),
                opacity=0.5, hoverinfo='none', showlegend=False,
            )
            for a, c, lift in edges
        ]

        node_trace = go.Scatter(
            x=list(nx.values()), y=list(ny.values()),
            mode='markers+text',
            marker=dict(size=18, color=PURPLE, line=dict(width=1, color=GRID_COLOR)),
            text=node_list, textposition="top center",
            textfont=dict(size=8, color=TEXT_COLOR),
            hoverinfo='text', name="Skills",
        )

        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(**_base_layout(
            title_text="Skill Association Network",
            height=380,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
        ))
        return fig
    except Exception:
        return None


# ── 7. Skill Category Donut ───────────────────────────────────────────────────
def plot_skill_categories(skills: list, skill_db: dict) -> go.Figure:
    from modules.skill_extractor import get_skill_category

    cat_counts: dict = {}
    for skill in skills:
        cat = get_skill_category(skill, skill_db).replace("_", " ").title()
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    if not cat_counts:
        fig = go.Figure()
        fig.update_layout(**_base_layout(height=280))
        fig.add_annotation(text="No skills to categorize", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=MUTED_COLOR))
        return fig

    labels = list(cat_counts.keys())
    values = list(cat_counts.values())

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.52,
        marker=dict(colors=PALETTE[:len(labels)], line=dict(color=BG_COLOR, width=2)),
        textinfo="label+percent",
        textfont=dict(size=9, color=TEXT_COLOR),
        hovertemplate='<b>%{label}</b>: %{value} skills (%{percent})<extra></extra>',
    ))
    fig.update_layout(**_base_layout(title_text="Skill Categories", height=300, showlegend=False))
    return fig