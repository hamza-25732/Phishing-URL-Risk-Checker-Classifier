# AI-Assisted Phishing URL Classifier

A machine learning system that scores a URL's phishing risk using lexical
and structural features only (no live browsing, no content fetch) — trained
on ~188K URLs and deployed as an interactive Streamlit app.

## What this project demonstrates

- End-to-end ML pipeline: data sourcing → feature engineering → model
  training/comparison → deployment
- **Catching and fixing two rounds of data leakage** (see below) — the kind
  of debugging judgment that separates a real project from a copy-pasted
  notebook
- Security-relevant feature engineering grounded in known phishing-URL
  literature (HTTPS usage, subdomain abuse, hyphenation, IP-as-domain,
  character entropy, suspicious keywords)
- Honest evaluation: precision/recall/F1/ROC-AUC and false-positive rate,
  not just accuracy

## Pipeline

1. **Data** — Combined multiple public sources (PhiUSIIL, UNB benign
   corpus, PhishTank-derived and manually-verified phishing URL lists) into
   ~188K labeled URLs, plus 30K synthetically generated realistic legitimate
   deep links across 140+ well-known domains (see "Data leakage" below for
   why).
2. **Feature engineering** (`src/features.py`) — 27 features extracted
   directly from the raw URL string: HTTPS usage, subdomain count, hyphen/
   digit/special-char counts and ratios, domain character entropy, IP-as-
   domain detection, known shortener detection, suspicious keyword count,
   and more.
3. **Modeling** (`src/train.py`) — Random Forest and XGBoost, compared on
   accuracy, precision, recall, F1, ROC-AUC, and false-positive/negative
   rate. Regularized (shallow trees, min leaf size, L1/L2 penalties) after
   an early version overfit — see below.
4. **App** (`app/app.py`) — Streamlit interface: paste a URL, get a risk
   score, a low/medium/high verdict, and the top features driving the
   score.

## Data leakage — two real issues I found and fixed

**Round 1:** My first dataset had legitimate URLs from PhiUSIIL, which only
provides bare homepage links (`https://example.com`) with no path. Phishing
URLs, by contrast, almost always had paths. The model hit 99.5%+ accuracy
by learning "has a path → phishing" — a dataset artifact that would fail on
any real legitimate deep link. **Fix:** added the UNB benign URL corpus
(35K legitimate URLs with real paths) and rebalanced path-presence across
classes.

**Round 2:** Even after that fix, the model still flagged ordinary
legitimate URLs like `github.com/anthropics` or `stackoverflow.com/
questions/...` as phishing with near-100% confidence. Digging in, raw
length-based features (`url_length`, `path_length`, `domain_length`,
`num_slashes`) turned out to be confounded with *which source dataset* a
URL came from — my legit corpus (torrent/wiki/news sites) happened to have
*longer* paths on average than my phishing corpus, the opposite of common
intuition. **Fix:** dropped raw-magnitude features in favor of
scale-invariant ratios and bounded counts, generated 30K additional
synthetic-but-realistic legitimate URLs across 140+ well-known domains to
close the diversity gap, and regularized both models more heavily.

**Result after both fixes:** accuracy dropped from an artificially inflated
99.5%+ to a more honest ~93-94% — with the model now correctly handling 15
of 17 hand-picked real-world sanity-check URLs it had never seen, including
GitHub, Amazon, Reddit, and LinkedIn deep links, while still catching every
phishing example with high confidence.

## Known limitations

- **Lexical-only.** The model never looks at page content, so a
  well-crafted phishing URL that mimics normal URL structure (or a
  legitimate URL with an unusual structure) can fool it in either
  direction. Treat the score as one signal among several, not a verdict.
- **Dataset drift.** Public phishing-URL datasets skew toward a particular
  era and style of phishing kit; performance on brand-new phishing
  campaigns will likely be somewhat lower than the held-out test score.
- **Not a production tool.** No rate limiting, no live blocklist
  integration, no monitoring. Built for learning and portfolio purposes.

## Project structure

```
phishing_url_classifier/
├── data/                   # raw + processed datasets
├── src/
│   ├── features.py         # feature extraction (shared with the app)
│   ├── augment_data.py     # synthetic legit URL generator
│   └── train.py            # training + evaluation + plots
├── models/                 # trained model artifacts
├── reports/                # confusion matrices, feature importance, metrics
└── app/
    ├── app.py               # Streamlit app
    └── features.py           # (copy of src/features.py for import)
```

## Running the app

```bash
cd app
pip install -r ../requirements.txt
streamlit run app.py
```

## Possible extensions

- Fine-tune a small transformer (DistilBERT) directly on raw URL text
- Browser extension wrapper
- Add a live PhishTank feed for continuously updated training data
