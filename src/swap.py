"""
swap.py
-------
Generates the full swapped-sentence dataset.

Takes every (template, category, term) combination and produces a flat
pandas DataFrame that gets saved to CSV. This is the input to run_audit.py.

Usage:
    python swap.py
    # writes outputs/sentences.csv
"""

import pandas as pd
from pathlib import Path
from templates import TEMPLATES, DEMOGRAPHIC_GROUPS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generate_dataset() -> pd.DataFrame:
    """
    Cross-product of all templates × all demographic terms.
    Each row is one sentence to score.
    """
    rows = []
    for template in TEMPLATES:
        for category, terms in DEMOGRAPHIC_GROUPS.items():
            for term in terms:
                sentence = template.format(group=term)
                rows.append({
                    "template": template,
                    "category": category,
                    "term": term,
                    "sentence": sentence,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = generate_dataset()
    out_path = OUTPUT_DIR / "sentences.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} sentences across {len(DEMOGRAPHIC_GROUPS)} categories.")
    print(f"Saved to {out_path}")
    print()
    print(df.head(10).to_string(index=False))
