"""
stats.py
--------
Statistical tests on the scored dataset.

For each demographic category (gender, race, religion, …) we ask:
    "Are the model scores meaningfully different across the identity terms
     within this category, holding the template constant?"

Tests used:
    1. Kruskal-Wallis H-test  —  non-parametric omnibus test per category.
       Null: all groups have the same score distribution.
    2. Pairwise Wilcoxon (Mann-Whitney U)  —  for every pair of terms
       inside a category, with Bonferroni correction.

Output: outputs/stats_summary.csv   — one row per category with H, p-value, effect size
        outputs/pairwise.csv        — every pairwise comparison

Usage:
    python stats.py
    python stats.py --input outputs/scores.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
from pathlib import Path
from itertools import combinations

SCORES_DEFAULT = Path(__file__).resolve().parent.parent / "outputs" / "scores.csv"
OUTPUT_DIR     = Path(__file__).resolve().parent.parent / "outputs"


def detect_score_column(df: pd.DataFrame) -> str:
    """Auto-detect which score column is present (sentiment_score, toxicity_score, …)."""
    candidates = [c for c in df.columns if c.endswith("_score")]
    if not candidates:
        raise ValueError("No score column found in the CSV. Did you run run_audit.py first?")
    return candidates[0]   # if multiple, grab the first one


def effect_size_eta_squared(groups: list[np.ndarray]) -> float:
    """
    Eta-squared approximation for Kruskal-Wallis.
    η² = (H - k + 1) / (N - k)   where k = number of groups, N = total obs.
    Clamped to [0, 1].
    """
    k = len(groups)
    N = sum(len(g) for g in groups)
    # Re-run H to grab the stat (cheap)
    H, _ = kruskal(*groups)
    eta2 = (H - k + 1) / (N - k)
    return float(np.clip(eta2, 0, 1))


def run_omnibus(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    One Kruskal-Wallis test per demographic category.
    Returns a summary DataFrame.
    """
    rows = []
    for category, grp in df.groupby("category"):
        groups = [g[score_col].values for _, g in grp.groupby("term")]
        terms  = [name for name, _ in grp.groupby("term")]

        if len(groups) < 2:
            continue

        H, p = kruskal(*groups)
        eta2 = effect_size_eta_squared(groups)

        rows.append({
            "category":      category,
            "n_terms":       len(terms),
            "n_sentences":   len(grp),
            "H_statistic":   round(H, 4),
            "p_value":       round(p, 6),
            "eta_squared":   round(eta2, 4),
            "significant":   p < 0.05,
            "terms":         ", ".join(sorted(terms)),
        })

    return pd.DataFrame(rows).sort_values("p_value")


def run_pairwise(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Mann-Whitney U for every pair of terms within each category.
    Applies Bonferroni correction per category.
    """
    rows = []
    for category, grp in df.groupby("category"):
        term_groups = {name: g[score_col].values for name, g in grp.groupby("term")}
        pairs = list(combinations(sorted(term_groups.keys()), 2))
        n_pairs = len(pairs)   # for Bonferroni

        for term_a, term_b in pairs:
            a_scores = term_groups[term_a]
            b_scores = term_groups[term_b]

            if len(a_scores) == 0 or len(b_scores) == 0:
                continue

            U, p_raw = mannwhitneyu(a_scores, b_scores, alternative="two-sided")
            p_corrected = min(p_raw * n_pairs, 1.0)   # Bonferroni

            # Cohen's d (approximate, using pooled std)
            pooled_std = np.sqrt(
                ((len(a_scores)-1)*np.var(a_scores, ddof=1) +
                 (len(b_scores)-1)*np.var(b_scores, ddof=1)) /
                (len(a_scores) + len(b_scores) - 2)
            )
            cohens_d = (np.mean(a_scores) - np.mean(b_scores)) / pooled_std if pooled_std > 0 else 0.0

            rows.append({
                "category":         category,
                "term_a":           term_a,
                "term_b":           term_b,
                "mean_a":           round(float(np.mean(a_scores)), 4),
                "mean_b":           round(float(np.mean(b_scores)), 4),
                "U_statistic":      round(U, 2),
                "p_raw":            round(p_raw, 6),
                "p_bonferroni":     round(p_corrected, 6),
                "cohens_d":         round(float(cohens_d), 4),
                "significant":      p_corrected < 0.05,
            })

    return pd.DataFrame(rows).sort_values("p_bonferroni")


def main():
    parser = argparse.ArgumentParser(description="Run statistical tests on scored sentences.")
    parser.add_argument("--input", default=str(SCORES_DEFAULT),
                        help="Path to scores.csv (output of run_audit.py).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Scores file not found: {input_path}")
        print("    Run  python run_audit.py  first.")
        raise SystemExit(1)

    df = pd.read_csv(input_path)
    score_col = detect_score_column(df)
    print(f"Using score column: {score_col}\n")

    # --- Omnibus ---
    summary = run_omnibus(df, score_col)
    summary_path = OUTPUT_DIR / "stats_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("=== Omnibus (Kruskal-Wallis) per category ===")
    print(summary.to_string(index=False))
    print(f"\nSaved → {summary_path}\n")

    # --- Pairwise ---
    pairwise = run_pairwise(df, score_col)
    pairwise_path = OUTPUT_DIR / "pairwise.csv"
    pairwise.to_csv(pairwise_path, index=False)
    print("=== Top 15 pairwise comparisons (by p-value) ===")
    print(pairwise.head(15).to_string(index=False))
    print(f"\nSaved → {pairwise_path}")


if __name__ == "__main__":
    main()
