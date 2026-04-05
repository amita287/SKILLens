"""
Skill Association Mining Module
Finds co-occurrence patterns in skill sets using simple pair counting.
"""
import pandas as pd
from itertools import combinations


def mine_skill_associations(skill_lists: list) -> pd.DataFrame:
    """
    Given a list of skill lists, compute pairwise co-occurrence stats.

    Returns a DataFrame with columns:
        antecedents, consequents, support, confidence, lift
    """
    try:
        # Filter out empty lists
        skill_lists = [s for s in skill_lists if s]
        if not skill_lists:
            return pd.DataFrame()

        total = len(skill_lists)
        pair_counts: dict = {}
        single_counts: dict = {}

        for skill_set in skill_lists:
            unique = list(set(skill_set))
            for s in unique:
                single_counts[s] = single_counts.get(s, 0) + 1
            for combo in combinations(sorted(unique), 2):
                pair_counts[combo] = pair_counts.get(combo, 0) + 1

        if not pair_counts:
            return pd.DataFrame()

        rows = []
        for (ant, con), count in pair_counts.items():
            support    = count / total
            ant_sup    = single_counts.get(ant, 1) / total
            confidence = support / ant_sup if ant_sup > 0 else 0
            lift       = confidence / (single_counts.get(con, 1) / total) if single_counts.get(con, 1) > 0 else 1
            rows.append({
                "antecedents": ant,
                "consequents": con,
                "support":     round(support, 4),
                "confidence":  round(confidence, 4),
                "lift":        round(lift, 4),
            })

        df = pd.DataFrame(rows)
        return df.sort_values("support", ascending=False).reset_index(drop=True)

    except Exception:
        return pd.DataFrame()