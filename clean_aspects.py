"""
Aspect cleaning & merging for all_reviews_merged.csv

Steps:
1. Merge near-duplicate aspect labels
2. For rows where aspect is 'Positive', 'Negative', 'Neutral', or 'Irrelevant',
   re-assign the correct aspect using keyword-based rules on review_text
3. Further merge overlapping aspects
4. Save cleaned CSV as all_reviews_cleaned.csv
"""

import pandas as pd
import re

df = pd.read_csv("all_reviews_merged.csv")

print(f"Original shape: {df.shape}")
print(f"\nOriginal aspect counts:\n{df['aspect'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# STEP 1: Merge near-duplicate aspects
# ─────────────────────────────────────────────
ASPECT_MAP = {
    # Staff
    "Staff quality":                    "Staff & Service Quality",
    "Staff & Customer Service":         "Staff & Service Quality",
    # Cleanliness
    "Cleanliness":                      "Cleanliness & Hygiene",
    # Safety
    "Safety":                           "Safety & Security",
    # Punctuality
    "Punctuality":                      "Punctuality & Reliability",
    # Facilities
    "Infrastructure & Facilities":      "Facilities & Accessibility",
    "Facilities":                       "Facilities & Accessibility",
    "Accessibility":                    "Facilities & Accessibility",
    # Fare / Price
    "Price fairness":                   "Fare & Payment System",
    # Information
    "Signage & Navigation":             "Information & Navigation",
    "Data availability":                "Information & Navigation",
    # Convenience + Overall
    "Convenience":                      "Overall Experience & Convenience",
    "Overall Experience":               "Overall Experience & Convenience",
}

df["aspect"] = df["aspect"].replace(ASPECT_MAP)

print(f"\nAfter merge (before rule-based fix):\n{df['aspect'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# STEP 2: Rule-based re-assignment for artifact
#         labels AND all Irrelevant rows
# ─────────────────────────────────────────────

# Keyword rules: ordered by specificity (first match wins)
RULES = [
    ("Fare & Payment System",               r"fare|price|cost|ticket|payment|pay|cheap|expensive|rabbit card|top.?up|fee|baht|surcharge"),
    ("Crowding & Comfort",                  r"crowd|crowded|packed|rush hour|peak|seat|standing|space|comfort|comfortable|squeeze|full|busy"),
    ("Cleanliness & Hygiene",               r"clean|dirty|hygiene|smell|odor|trash|garbage|litter|sanit"),
    ("Staff & Service Quality",             r"staff|officer|guard|employee|service|rude|helpful|friendly|assist|attitude|personnel"),
    ("Punctuality & Reliability",           r"delay|late|on time|punctual|reliable|schedule|frequency|wait|waiting|interval|breakdown|cancel"),
    ("Safety & Security",                   r"safe|safety|security|accident|crime|theft|steal|cctv|police|emergency|danger|hazard"),
    ("Facilities & Accessibility",          r"accessib|disable|wheelchair|elevator|lift|escalator|ramp|elderly|handicap|barrier|facilit|infrastructure|platform|toilet|restroom|wifi|air.?con|bench|repair|maintain|maintenance"),
    ("Information & Navigation",            r"sign|signage|map|direction|navigate|navigation|confus|lost|label|display|board|announcement|data|information|info|app|website|online|real.?time|update|timetable|screen"),
    ("Route Coverage & Connectivity",       r"route|coverage|connect|extend|extension|line|network|interchange|transfer|reach|destination|suburb"),
    ("Overall Experience & Convenience",    r"convenient|convenience|easy|simple|quick|fast|efficient|hassle|smooth|seamless|overall|experience|general|impression|recommend|worth|value|enjoy|love|hate|terrible|excellent|great|good|bad|poor|amazing|awful"),
]

def infer_aspect(text: str) -> str:
    text_lower = str(text).lower()
    for aspect, pattern in RULES:
        if re.search(pattern, text_lower):
            return aspect
    return "Irrelevant"

# Fix artifact labels AND all Irrelevant rows
reassign_mask = df["aspect"].isin(["Positive", "Negative", "Neutral", "Irrelevant"])
reassign_count = reassign_mask.sum()
print(f"\nRows to re-assign (artifacts + Irrelevant): {reassign_count}")

df.loc[reassign_mask, "aspect"] = df.loc[reassign_mask, "review_text"].apply(infer_aspect)

print(f"\nRe-assigned rows distribution:")
print(df.loc[reassign_mask, "aspect"].value_counts().to_string())

# ─────────────────────────────────────────────
# STEP 3: Final counts & save
# ─────────────────────────────────────────────
print(f"\nFinal aspect counts:\n{df['aspect'].value_counts().to_string()}")
print(f"\nFinal unique aspects: {df['aspect'].nunique()}")
print(f"\nRemaining Irrelevant: {(df['aspect'] == 'Irrelevant').sum()}")

# Drop remaining Irrelevant rows — genuinely off-topic content
before = len(df)
df = df[df["aspect"] != "Irrelevant"].reset_index(drop=True)
print(f"\nDropped {before - len(df)} remaining Irrelevant rows.")
print(f"Final shape: {df.shape}")

df.to_csv("all_reviews_cleaned.csv", index=False)
print(f"\nSaved: all_reviews_cleaned.csv  (shape: {df.shape})")
