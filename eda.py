"""
EDA for all_reviews_merged.csv
Bangkok BTS Skytrain Reviews - Sentiment Analysis Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import re
import warnings
import os

warnings.filterwarnings("ignore")

# Output directory for charts
os.makedirs("eda_output", exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

df = pd.read_csv("all_reviews_cleaned.csv")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

# ─────────────────────────────────────────────
# 2. BASIC INFO & MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
print(missing_df)

print(f"\nDescriptive stats:\n{df.describe(include='all').to_string()}")

# ─────────────────────────────────────────────
# 3. TARGET VARIABLE DISTRIBUTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. SENTIMENT & RATING DISTRIBUTIONS")
print("=" * 60)

print(f"\nSentiment counts:\n{df['sentiment'].value_counts()}")
print(f"\nRating counts:\n{df['review_rating'].value_counts().sort_index()}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Sentiment bar
sent_counts = df["sentiment"].value_counts()
axes[0].bar(sent_counts.index, sent_counts.values, color=["#4CAF50", "#F44336"])
axes[0].set_title("Sentiment Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(sent_counts.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

# Rating histogram
rating_counts = df["review_rating"].value_counts().sort_index()
axes[1].bar(rating_counts.index.astype(str), rating_counts.values, color="#2196F3")
axes[1].set_title("Review Rating Distribution")
axes[1].set_xlabel("Rating")
axes[1].set_ylabel("Count")
for i, v in enumerate(rating_counts.values):
    axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")

# Rating × Sentiment heatmap
ct = pd.crosstab(df["review_rating"], df["sentiment"])
sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=axes[2])
axes[2].set_title("Rating × Sentiment Crosstab")

plt.tight_layout()
plt.savefig("eda_output/01_sentiment_rating.png")
plt.close()
print("Saved: eda_output/01_sentiment_rating.png")

# ─────────────────────────────────────────────
# 4. SOURCE & METADATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. SOURCE & METADATA")
print("=" * 60)

print(f"\nSource counts:\n{df['source'].value_counts()}")
print(f"\nBTS line counts:\n{df['bts_line'].value_counts()}")
print(f"\nAspect counts:\n{df['aspect'].value_counts()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Source
src = df["source"].value_counts()
axes[0].bar(src.index, src.values, color="#9C27B0")
axes[0].set_title("Reviews by Source")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)

# BTS line
bts = df["bts_line"].value_counts().dropna()
axes[1].barh(bts.index, bts.values, color="#FF9800")
axes[1].set_title("Reviews by BTS Line")
axes[1].set_xlabel("Count")

# Aspect
asp = df["aspect"].value_counts()
axes[2].barh(asp.index, asp.values, color="#00BCD4")
axes[2].set_title("Reviews by Aspect")
axes[2].set_xlabel("Count")

plt.tight_layout()
plt.savefig("eda_output/02_source_metadata.png")
plt.close()
print("Saved: eda_output/02_source_metadata.png")

# ─────────────────────────────────────────────
# 5. TEMPORAL ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TEMPORAL ANALYSIS")
print("=" * 60)

df["created_at_date"] = pd.to_datetime(df["created_at_date"], errors="coerce")
df["year_month"] = df["created_at_date"].dt.to_period("M")

monthly = df.groupby("year_month").size()
print(f"\nDate range: {df['created_at_date'].min()} → {df['created_at_date'].max()}")
print(f"Months with data: {len(monthly)}")

# Monthly sentiment trend
monthly_sent = df.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)
monthly_sent["NSS"] = (
    (monthly_sent.get("Positive", 0) - monthly_sent.get("Negative", 0))
    / monthly_sent.sum(axis=1)
    * 100
).round(2)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Review volume
x = [str(p) for p in monthly.index]
axes[0].plot(x, monthly.values, marker="o", color="#2196F3", linewidth=1.5)
axes[0].set_title("Monthly Review Volume")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)
step = max(1, len(x) // 12)
axes[0].set_xticks(range(0, len(x), step))
axes[0].set_xticklabels(x[::step], rotation=45)

# NSS trend
nss_x = [str(p) for p in monthly_sent.index]
axes[1].plot(nss_x, monthly_sent["NSS"].values, marker="o", color="#4CAF50", linewidth=1.5)
axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_title("Monthly Net Sentiment Score (NSS)")
axes[1].set_ylabel("NSS (%)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].set_xticks(range(0, len(nss_x), step))
axes[1].set_xticklabels(nss_x[::step], rotation=45)

plt.tight_layout()
plt.savefig("eda_output/03_temporal.png")
plt.close()
print("Saved: eda_output/03_temporal.png")

# ─────────────────────────────────────────────
# 6. TEXT LENGTH ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. TEXT LENGTH ANALYSIS")
print("=" * 60)

df["text_length"] = df["review_text"].astype(str).apply(len)
df["word_count"] = df["review_text"].astype(str).apply(lambda x: len(x.split()))

print(f"\nText length stats:\n{df['text_length'].describe().round(1)}")
print(f"\nWord count stats:\n{df['word_count'].describe().round(1)}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Text length histogram
axes[0, 0].hist(df["text_length"].clip(upper=2000), bins=50, color="#3F51B5", edgecolor="white")
axes[0, 0].set_title("Text Length Distribution (chars, clipped at 2000)")
axes[0, 0].set_xlabel("Characters")
axes[0, 0].set_ylabel("Count")

# Word count histogram
axes[0, 1].hist(df["word_count"].clip(upper=300), bins=50, color="#E91E63", edgecolor="white")
axes[0, 1].set_title("Word Count Distribution (clipped at 300)")
axes[0, 1].set_xlabel("Words")
axes[0, 1].set_ylabel("Count")

# Text length by sentiment
df.boxplot(column="text_length", by="sentiment", ax=axes[1, 0])
axes[1, 0].set_title("Text Length by Sentiment")
axes[1, 0].set_xlabel("Sentiment")
axes[1, 0].set_ylabel("Characters")
plt.sca(axes[1, 0])
plt.title("Text Length by Sentiment")

# Word count by sentiment
df.boxplot(column="word_count", by="sentiment", ax=axes[1, 1])
axes[1, 1].set_title("Word Count by Sentiment")
axes[1, 1].set_xlabel("Sentiment")
axes[1, 1].set_ylabel("Words")
plt.sca(axes[1, 1])
plt.title("Word Count by Sentiment")

plt.suptitle("")
plt.tight_layout()
plt.savefig("eda_output/04_text_length.png")
plt.close()
print("Saved: eda_output/04_text_length.png")

# ─────────────────────────────────────────────
# 7. VOCABULARY ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. VOCABULARY ANALYSIS")
print("=" * 60)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "i", "you", "we", "they", "this", "that",
    "was", "are", "be", "have", "has", "had", "do", "did", "not", "from",
    "by", "as", "so", "if", "my", "me", "he", "she", "his", "her", "our",
    "your", "their", "its", "will", "would", "can", "could", "should",
    "there", "here", "what", "which", "who", "how", "when", "where", "why",
    "all", "any", "some", "no", "more", "also", "just", "up", "out", "about",
    "than", "then", "been", "were", "am", "into", "very", "get", "got",
    "one", "two", "like", "go", "use", "used", "amp", "re", "s", "t", "m",
}

def tokenize(text):
    tokens = re.findall(r"\b[a-z]{3,}\b", str(text).lower())
    return [t for t in tokens if t not in STOPWORDS]

all_tokens = []
for text in df["review_text"]:
    all_tokens.extend(tokenize(text))

vocab_size = len(set(all_tokens))
print(f"Total tokens: {len(all_tokens):,}")
print(f"Vocabulary size: {vocab_size:,}")

top30 = Counter(all_tokens).most_common(30)
print(f"\nTop 30 words: {[w for w, _ in top30]}")

# Top words by sentiment
pos_tokens, neg_tokens = [], []
for _, row in df.iterrows():
    tokens = tokenize(row["review_text"])
    if row["sentiment"] == "Positive":
        pos_tokens.extend(tokens)
    else:
        neg_tokens.extend(tokens)

top_pos = Counter(pos_tokens).most_common(20)
top_neg = Counter(neg_tokens).most_common(20)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# All top 30
words, counts = zip(*top30)
axes[0].barh(list(reversed(words)), list(reversed(counts)), color="#607D8B")
axes[0].set_title("Top 30 Words (All Reviews)")
axes[0].set_xlabel("Frequency")

# Top positive
pw, pc = zip(*top_pos)
axes[1].barh(list(reversed(pw)), list(reversed(pc)), color="#4CAF50")
axes[1].set_title("Top 20 Words — Positive Reviews")
axes[1].set_xlabel("Frequency")

# Top negative
nw, nc = zip(*top_neg)
axes[2].barh(list(reversed(nw)), list(reversed(nc)), color="#F44336")
axes[2].set_title("Top 20 Words — Negative Reviews")
axes[2].set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("eda_output/05_vocabulary.png")
plt.close()
print("Saved: eda_output/05_vocabulary.png")

# ─────────────────────────────────────────────
# 8. WORD CLOUDS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. WORD CLOUDS")
print("=" * 60)

def make_wordcloud(tokens, title, color, path):
    freq = Counter(tokens)
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap=color,
        max_words=150,
        collocations=False,
    ).generate_from_frequencies(freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

make_wordcloud(all_tokens, "Word Cloud — All Reviews", "viridis", "eda_output/06a_wc_all.png")
make_wordcloud(pos_tokens, "Word Cloud — Positive Reviews", "Greens", "eda_output/06b_wc_positive.png")
make_wordcloud(neg_tokens, "Word Cloud — Negative Reviews", "Reds", "eda_output/06c_wc_negative.png")

# ─────────────────────────────────────────────
# 9. ASPECT ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. ASPECT ANALYSIS")
print("=" * 60)

aspect_sent = pd.crosstab(df["aspect"], df["sentiment"])
aspect_sent["total"] = aspect_sent.sum(axis=1)
aspect_sent["neg_pct"] = (
    aspect_sent.get("Negative", 0) / aspect_sent["total"] * 100
).round(1)
print(f"\nAspect × Sentiment:\n{aspect_sent.sort_values('neg_pct', ascending=False).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
pct_table = aspect_sent.drop(columns=["total", "neg_pct"])
pct_norm = pct_table.div(pct_table.sum(axis=1), axis=0) * 100
sns.heatmap(pct_norm, annot=True, fmt=".1f", cmap="RdYlGn", ax=axes[0], vmin=0, vmax=100)
axes[0].set_title("Aspect × Sentiment (% of aspect total)")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Aspect")

# Negative % bar
neg_sorted = aspect_sent["neg_pct"].sort_values(ascending=True)
colors = ["#F44336" if v > 30 else "#FF9800" if v > 15 else "#4CAF50" for v in neg_sorted]
axes[1].barh(neg_sorted.index, neg_sorted.values, color=colors)
axes[1].set_title("Negative Sentiment % by Aspect")
axes[1].set_xlabel("Negative %")
axes[1].axvline(neg_sorted.mean(), color="black", linestyle="--", label=f"Mean: {neg_sorted.mean():.1f}%")
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_output/07_aspect_analysis.png")
plt.close()
print("Saved: eda_output/07_aspect_analysis.png")

# ─────────────────────────────────────────────
# 10. SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. SUMMARY STATISTICS")
print("=" * 60)

total = len(df)
pos_count = (df["sentiment"] == "Positive").sum()
neg_count = (df["sentiment"] == "Negative").sum()
nss_overall = round((pos_count - neg_count) / total * 100, 2)

summary = {
    "Total reviews": f"{total:,}",
    "Positive reviews": f"{pos_count:,} ({pos_count/total*100:.1f}%)",
    "Negative reviews": f"{neg_count:,} ({neg_count/total*100:.1f}%)",
    "Overall NSS": f"{nss_overall}%",
    "Avg text length (chars)": f"{df['text_length'].mean():.1f}",
    "Median text length (chars)": f"{df['text_length'].median():.1f}",
    "Avg word count": f"{df['word_count'].mean():.1f}",
    "Median word count": f"{df['word_count'].median():.1f}",
    "Vocabulary size": f"{vocab_size:,}",
    "Total tokens": f"{len(all_tokens):,}",
    "Date range": f"{df['created_at_date'].min().date()} → {df['created_at_date'].max().date()}",
    "Sources": ", ".join(df["source"].unique()),
    "BTS lines": ", ".join(df["bts_line"].dropna().unique()),
    "Aspects": str(df["aspect"].nunique()),
}

print()
for k, v in summary.items():
    print(f"  {k:<35} {v}")

print("\n" + "=" * 60)
print("EDA COMPLETE — charts saved to eda_output/")
print("=" * 60)
