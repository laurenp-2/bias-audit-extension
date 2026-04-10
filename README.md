# bias-swap-audit

A tool for auditing demographic bias in pre-trained sentiment models using counterfactual term swapping.

Takes neutral sentence templates, swaps in identity terms from different demographic categories (gender, race, religion, etc.), runs a pre-trained sentiment model, and checks whether scores shift depending on the subject. 

---

## Results summary

The model used is `distilbert-base-uncased-finetuned-sst-2-english`, a DistilBERT model fine-tuned on SST-2 (Stanford Sentiment Treebank). Scores are mapped to a **[-1, 1] scale**: positive values = model reads the sentence as positive, negative values = negative.

A few things that stood out:

**Religion** had the largest within-category gap. "atheist" had a mean score of **-0.45**, while "Christian" scored **-0.07** — nearly a 0.4 difference on sentences that are otherwise identical. Muslim and Jewish person also scored meaningfully lower than Christian and Hindu.

**Race** showed a consistent pattern where "Black person" and "Latino person" clustered at the bottom of the category (means around -0.36 to -0.39), while "Asian person" scored notably higher (-0.11). "White person" was also on the lower end (-0.38).

**Socioeconomic** terms showed a gap between "wealthy person" (-0.05) and "person from a low-income family" (-0.38). The model reads sentences about wealthy people as more positive on average.

**Template sensitivity** was one of the more interesting outputs. Some sentence templates barely moved when you swapped terms ("The {group} has been in the headlines recently" stayed pretty stable) while others spanned the entire scale ("The {group} received a promotion at work last month" had a drift of ~1.94 out of a possible 2.0, with "senior citizen" getting a promotion as positive and "Black person" getting a promotion as negative).

**Gender and nationality** showed smaller and less consistent differences. Gender in particular was pretty flat across terms — the model didn't seem to have strong sentiment associations between male and female terms in this template set.

---

## What's in here

| File | What it does |
|---|---|
| `src/templates.py` | Sentence templates and demographic term lists |
| `src/swap.py` | Generates the swapped-sentence dataset → `outputs/sentences.csv` |
| `src/run_audit.py` | Runs a HuggingFace sentiment/toxicity model over every sentence → `outputs/scores.csv` |
| `src/stats.py` | Kruskal-Wallis + pairwise Wilcoxon tests per category → `outputs/stats_summary.csv`, `pairwise.csv` |
| `src/visualize.py` | Generates box plots, a heatmap, and a template-drift chart → `outputs/*.png` |

---

## Setup

```bash
git clone https://github.com/your-username/bias-swap-audit.git
cd bias-swap-audit
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Models download automatically on first run (~250 MB for the sentiment model). They cache in `~/.cache/huggingface/` so you only download once.

**Note:** use `python3` to run scripts, not `python` — on some setups `python` aliases to a system interpreter outside the venv.

---

## Running the pipeline

From the `src/` directory, run these in order:

```bash
cd src

# 1. Generate sentences
python3 swap.py

# 2. Score them
python3 run_audit.py --model sentiment

# 3. Run stats
python3 stats.py

# 4. Make plots
python3 visualize.py
```

Everything writes to `outputs/`. You can also swap in a toxicity model with `--model toxicity`, or pass any HuggingFace text-classification model slug directly.

---

## How to read the outputs

**`scores.csv`** — one row per sentence. The `sentiment_score` column is the model's output mapped to [-1, 1]. Positive = model reads it as positive sentiment, negative = negative. Rows are tagged with the template, demographic category, and identity term.

**`stats_summary.csv`** — one row per demographic category. `H_statistic` and `p_value` are from a Kruskal-Wallis test (tests whether score distributions differ across terms in that category). `eta_squared` is the effect size — 0.01 is small, 0.06 is medium, 0.14 is large.

**`pairwise.csv`** — every term-vs-term comparison within each category. `cohens_d` gives the effect size for each pair. `p_bonferroni` is corrected for multiple comparisons.

**Plots:**
- `boxplots.png` — score distributions per term, one panel per category. The grey dashed line is the category mean.
- `heatmap.png` — mean score as a colour grid. Darker blue = more negative. 
- `template_drift.png` — which templates are most sensitive to the swap. The labels on each bar show which terms hit the max and min. These are the cases where the model's behavior is most clearly tied to the identity term rather than the sentence content.

---

## Extending it

**New demographic category:** add a key + term list to `DEMOGRAPHIC_GROUPS` in `src/templates.py`. 

**New templates:** append to `TEMPLATES` in the same file. 

**Different model:** pass any HuggingFace text-classification slug to `--model`. E.g. `python3 run_audit.py --model facebook/bart-large-mnli`.

---

## References

- Bolukbasi et al., "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings" (NeurIPS 2016)
- Lu et al., "Measuring and Reducing Gendered Correlations in Pre-trained Models" (2020)
- Nadeem et al., "Intrinsic Bias Metrics Beyond Co-occurrence" (2021)

---

## License

MIT
