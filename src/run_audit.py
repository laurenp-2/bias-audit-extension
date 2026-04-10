"""
run_audit.py
------------
Runs sentiment / toxicity inference over the swapped sentence dataset.

Supports two model modes out of the box:
    --model sentiment   →  distilbert-base-uncased-finetuned-sst-2-english
    --model toxicity    →  unitary/toxic-bert

You can also pass any HuggingFace pipeline-compatible model slug directly:
    --model facebook/bart-large-mnli   (zero-shot classification)

Output: outputs/scores.csv  (sentences.csv + model scores appended)

Usage:
    python run_audit.py --model sentiment
    python run_audit.py --model toxicity
    python run_audit.py --model sentiment --input outputs/sentences.csv
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

# ---------------------------------------------------------------------------
# Model registry — maps short names to HuggingFace model slugs + config
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "sentiment": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "sentiment-analysis",
        # This model outputs POSITIVE / NEGATIVE with a score.
        # We convert to a single [-1, 1] scale: POSITIVE → +score, NEGATIVE → -score
        "score_fn": lambda out: out["score"] if out["label"] == "POSITIVE" else -out["score"],
        "score_label": "sentiment_score",   # column name in output CSV
    },
    "toxicity": {
        "model_name": "unitary/toxic-bert",
        "task": "text-classification",
        # toxic-bert outputs a single label with probability. Higher = more toxic.
        "score_fn": lambda out: out["score"],
        "score_label": "toxicity_score",
    },
}

INPUT_DEFAULT  = Path(__file__).resolve().parent.parent / "outputs" / "sentences.csv"
OUTPUT_PATH    = Path(__file__).resolve().parent.parent / "outputs" / "scores.csv"


def load_pipeline(model_key: str):
    """Instantiate the HuggingFace pipeline for the chosen model."""
    if model_key in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[model_key]
        print(f"Loading model: {cfg['model_name']}  (task={cfg['task']})")
        pipe = pipeline(cfg["task"], model=cfg["model_name"],
                        device=0 if torch.cuda.is_available() else -1)
        return pipe, cfg["score_fn"], cfg["score_label"]
    else:
        # Treat model_key as a raw HuggingFace slug — default to text-classification
        print(f"Loading custom model: {model_key}")
        pipe = pipeline("text-classification", model=model_key,
                        device=0 if torch.cuda.is_available() else -1)
        score_fn = lambda out: out["score"]
        return pipe, score_fn, "custom_score"


def score_sentences(df: pd.DataFrame, pipe, score_fn, score_label: str,
                    batch_size: int = 32) -> pd.DataFrame:
    """
    Run inference in batches. Appends the score column to the DataFrame.
    """
    sentences = df["sentence"].tolist()
    scores = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Scoring"):
        batch = sentences[i:i + batch_size]
        # pipeline returns a list of dicts; grab first result per sentence
        results = pipe(batch, truncation=True, max_length=512)
        for res in results:
            # Some pipelines return a list-of-lists; flatten if needed
            if isinstance(res, list):
                res = res[0]
            scores.append(score_fn(res))

    df[score_label] = scores
    # Also store raw label if sentiment model (handy for sanity checks)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run bias audit inference.")
    parser.add_argument("--model", default="sentiment",
                        help="Model key (sentiment | toxicity) or a HuggingFace slug.")
    parser.add_argument("--input", default=str(INPUT_DEFAULT),
                        help="Path to sentences.csv (output of swap.py).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference.")
    args = parser.parse_args()

    # --- load sentences ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Input file not found: {input_path}")
        print("    Run  python swap.py  first to generate it.")
        raise SystemExit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} sentences from {input_path}")

    # --- load model & score ---
    pipe, score_fn, score_label = load_pipeline(args.model)
    df = score_sentences(df, pipe, score_fn, score_label, batch_size=args.batch_size)

    # --- save ---
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. Results saved to {OUTPUT_PATH}")
    print(df[["category", "term", score_label]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
