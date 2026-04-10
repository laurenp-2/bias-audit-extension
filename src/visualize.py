"""
visualize.py
------------
Generates the plots for the bias audit.

Charts produced (all saved to outputs/):
    1. boxplots.png        — score distribution per term, faceted by category
    2. heatmap.png         — mean score per (category, term) as a colour grid
    3. template_drift.png  — for each template, how much the score swings
                             across terms (highlights worst-case templates)

Usage:
    python visualize.py
    python visualize.py --input outputs/scores.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCORES_DEFAULT = Path(__file__).resolve().parent.parent / "outputs" / "scores.csv"
OUTPUT_DIR     = Path(__file__).resolve().parent.parent / "outputs"

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("colorblind")


def detect_score_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.endswith("_score")]
    if not candidates:
        raise ValueError("No score column found.")
    return candidates[0]


# ---------------------------------------------------------------------------
# 1. Box plots — one facet per demographic category
# ---------------------------------------------------------------------------
def plot_boxplots(df: pd.DataFrame, score_col: str):
    categories = df["category"].unique()
    n_cats = len(categories)

    fig, axes = plt.subplots(n_cats, 1, figsize=(12, 4 * n_cats), sharex=False)
    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, sorted(categories)):
        sub = df[df["category"] == cat].sort_values("term")
        order = sub["term"].unique()

        sns.boxplot(data=sub, x="term", y=score_col, order=order,
                    palette=PALETTE, ax=ax, width=0.6)
        ax.set_title(f"Score distribution — {cat}", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(score_col.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)

        # draw a horizontal line at the category mean for reference
        cat_mean = sub[score_col].mean()
        ax.axhline(cat_mean, color="grey", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    out = OUTPUT_DIR / "boxplots.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved boxplots → {out}")


# ---------------------------------------------------------------------------
# 2. Heatmap — mean score as a grid  (category × term)
# ---------------------------------------------------------------------------
def plot_heatmap(df: pd.DataFrame, score_col: str):
    pivot = (
        df.groupby(["category", "term"])[score_col]
        .mean()
        .unstack(level="term")
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": score_col.replace("_", " ").title()})
    ax.set_title("Mean model score by demographic term", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    out = OUTPUT_DIR / "heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved heatmap  → {out}")


# ---------------------------------------------------------------------------
# 3. Template-level score drift  —  highlights which templates are most
#    sensitive to the identity swap
# ---------------------------------------------------------------------------
def plot_template_drift(df: pd.DataFrame, score_col: str, top_n: int = 10):
    """
    For each template, compute  max(score) - min(score) across all terms.
    Plot the top-N templates by drift as a horizontal bar chart, colour-coded
    by which category produced the extremes.
    """
    records = []
    for template, grp in df.groupby("template"):
        max_row = grp.loc[grp[score_col].idxmax()]
        min_row = grp.loc[grp[score_col].idxmin()]
        drift = max_row[score_col] - min_row[score_col]
        records.append({
            "template":       template[:72] + "…" if len(template) > 72 else template,
            "drift":          drift,
            "max_term":       max_row["term"],
            "min_term":       min_row["term"],
            "max_category":   max_row["category"],
            "min_category":   min_row["category"],
        })

    drift_df = pd.DataFrame(records).sort_values("drift", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = sns.color_palette("colorblind", n_colors=len(drift_df))

    bars = ax.barh(
        y=range(len(drift_df)),
        width=drift_df["drift"].values,
        color=colors,
        edgecolor="white",
        height=0.6,
    )

    # labels: show the max and min term on each bar
    for i, (_, row) in enumerate(drift_df.iterrows()):
        ax.text(row["drift"] + 0.005, i,
                f"↑ {row['max_term']}  |  ↓ {row['min_term']}",
                va="center", fontsize=8.5, color="#333333")

    ax.set_yticks(range(len(drift_df)))
    ax.set_yticklabels(drift_df["template"].values, fontsize=9)
    ax.set_xlabel(f"Score range (max − min across all terms)", fontsize=11)
    ax.set_title(f"Top {top_n} templates by score drift", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    out = OUTPUT_DIR / "template_drift.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved drift chart → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate bias-audit plots.")
    parser.add_argument("--input", default=str(SCORES_DEFAULT),
                        help="Path to scores.csv.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] {input_path} not found. Run run_audit.py first.")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(input_path)
    score_col = detect_score_column(df)
    print(f"Plotting with score column: {score_col}\n")

    plot_boxplots(df, score_col)
    plot_heatmap(df, score_col)
    plot_template_drift(df, score_col)

    print("\nAll plots saved to outputs/")


if __name__ == "__main__":
    main()
