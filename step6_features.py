import pandas as pd
import numpy as np

df = pd.read_csv("merged_dataset.csv", dtype={"date": str})
df = df.sort_values("date").reset_index(drop=True)

# convert price columns to numeric
for col in ["REMX", "BEL.NS", "DIXON.NS"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# fix zero REMX values
df["REMX"] = df["REMX"].replace(0, np.nan)

print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# ── FEATURE 1: 7-day rolling GDELT event count ────────────────────
df["gdelt_7d_count"] = df["event_count"].rolling(7).mean()

# ── FEATURE 2: 7-day rolling GDELT tone ──────────────────────────
df["gdelt_7d_tone"] = df["avg_tone"].rolling(7).mean()

# ── FEATURE 3: tone velocity (rate of change) ─────────────────────
df["tone_velocity"] = df["avg_tone"].diff(7)

# ── FEATURE 4: 7-day rolling Chinese media count ─────────────────
df["media_7d_count"] = df["china_media_count"].rolling(7).mean()

# ── FEATURE 5: 7-day rolling Chinese media tone ──────────────────
df["media_7d_tone"] = df["china_media_tone"].rolling(7).mean()

# ── FEATURE 6: spike indicator ────────────────────────────────────
rolling_mean = df["event_count"].rolling(90).mean()
rolling_std  = df["event_count"].rolling(90).std()
df["spike_indicator"] = (
    df["event_count"] > rolling_mean + 2 * rolling_std
).astype(int)

# ── FEATURE 7: Xinhua-specific count ─────────────────────────────
df["xinhua_7d"] = df["xinhua_count"].rolling(7).mean()

# ── FEATURE 8: REMX 30-day momentum ──────────────────────────────
df["remx_momentum_30d"] = df["REMX"].pct_change(30)

# ── FEATURE 9: BEL 14-day return ─────────────────────────────────
df["bel_return_14d"] = df["BEL.NS"].pct_change(14)

# ── FEATURE 10: Dixon 14-day return ──────────────────────────────
df["dixon_return_14d"] = df["DIXON.NS"].pct_change(14)

# ── DROP ROWS WITH NaN ────────────────────────────────────────────
feature_cols = [
    "gdelt_7d_count", "gdelt_7d_tone", "tone_velocity",
    "media_7d_count", "media_7d_tone", "spike_indicator",
    "xinhua_7d", "remx_momentum_30d", "bel_return_14d",
    "dixon_return_14d"
]

df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

print(f"\nAfter dropping NaN rows: {len(df_clean)} rows")
print(f"\nFeature summary:")
print(df_clean[feature_cols].describe().round(3))

# ── SAVE ──────────────────────────────────────────────────────────
final_cols = ["date"] + feature_cols + ["label"]
df_clean[final_cols].to_csv("features.csv", index=False)
print(f"\nLabel distribution:")
print(df_clean["label"].value_counts())
print(f"\nSaved to features.csv")