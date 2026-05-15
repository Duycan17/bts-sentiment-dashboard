"""Project configuration: paths, label maps, hparams, aspect taxonomy & rules."""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "all_reviews_cleaned.csv"

ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
FIGURES_DIR = ARTIFACTS / "figures"
LOGS_DIR = ARTIFACTS / "logs"
REPORTS_DIR = ARTIFACTS / "reports"

BASELINE_PATH = MODELS_DIR / "baseline.joblib"
TRANSFORMER_DIR = MODELS_DIR / "distilbert"

for d in (MODELS_DIR, FIGURES_DIR, LOGS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment labels
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_LABELS: list[str] = ["Negative", "Neutral", "Positive"]
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
ID2LABEL: dict[int, str] = {i: l for l, i in LABEL2ID.items()}

# ─────────────────────────────────────────────────────────────────────────────
# Aspect taxonomy (10 canonical aspects, matches all_reviews_cleaned.csv)
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_ASPECTS: list[str] = [
    "Fare & Payment System",
    "Crowding & Comfort",
    "Punctuality & Reliability",
    "Route Coverage & Connectivity",
    "Facilities & Accessibility",
    "Safety & Security",
    "Information & Navigation",
    "Staff & Service Quality",
    "Cleanliness & Hygiene",
    "Overall Experience & Convenience",
]

# Rule-based aspect extraction patterns. Order matters — first match wins.
# Used by both training data cleaning and runtime inference.
ASPECT_RULES: list[tuple[str, str]] = [
    ("Fare & Payment System",
        r"fare|price|cost|ticket|payment|pay|cheap|expensive|rabbit card|top.?up|fee|baht|surcharge"),
    ("Crowding & Comfort",
        r"crowd|crowded|packed|rush hour|peak|seat|standing|space|comfort|comfortable|squeeze|full|busy"),
    ("Cleanliness & Hygiene",
        r"clean|dirty|hygiene|smell|odor|trash|garbage|litter|sanit"),
    ("Staff & Service Quality",
        r"staff|officer|guard|employee|service|rude|helpful|friendly|assist|attitude|personnel"),
    ("Punctuality & Reliability",
        r"delay|late|on time|punctual|reliable|schedule|frequency|wait|waiting|interval|breakdown|cancel"),
    ("Safety & Security",
        r"safe|safety|security|accident|crime|theft|steal|cctv|police|emergency|danger|hazard"),
    ("Facilities & Accessibility",
        r"accessib|disable|wheelchair|elevator|lift|escalator|ramp|elderly|handicap|barrier|"
        r"facilit|infrastructure|platform|toilet|restroom|wifi|air.?con|bench|repair|maintain|maintenance"),
    ("Information & Navigation",
        r"sign|signage|map|direction|navigate|navigation|confus|lost|label|display|board|announcement|"
        r"data|information|info|app|website|online|real.?time|update|timetable|screen"),
    ("Route Coverage & Connectivity",
        r"route|coverage|connect|extend|extension|line|network|interchange|transfer|reach|destination|suburb"),
    ("Overall Experience & Convenience",
        r"convenient|convenience|easy|simple|quick|fast|efficient|hassle|smooth|seamless|"
        r"overall|experience|general|impression|recommend|worth|value|enjoy|love|hate|terrible|excellent|"
        r"great|good|bad|poor|amazing|awful"),
]

ASPECT_FALLBACK: str = "Overall Experience & Convenience"

# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
SPLIT_N_FOLDS: int = 5

# ─────────────────────────────────────────────────────────────────────────────
# Baseline hparams
# ─────────────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int = 80_000
TFIDF_NGRAM: tuple[int, int] = (1, 2)
LOGREG_C: float = 2.0
LOGREG_MAX_ITER: int = 2000

# ─────────────────────────────────────────────────────────────────────────────
# Transformer hparams
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMER_MODEL: str = "distilbert-base-uncased"
TRANSFORMER_EPOCHS: float = 3.0
TRANSFORMER_BATCH: int = 16
TRANSFORMER_LR: float = 2e-5
TRANSFORMER_MAX_LEN: int = 256
TRANSFORMER_PATIENCE: int = 2

# ─────────────────────────────────────────────────────────────────────────────
# Quick mode (smoke test)
# ─────────────────────────────────────────────────────────────────────────────
QUICK_MAX_ROWS: int = 5_000
QUICK_TRANSFORMER_EPOCHS: float = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
LDA_N_TOPICS: int = 6
LDA_N_TOP_WORDS: int = 12
WORDCLOUD_MAX_WORDS: int = 150
