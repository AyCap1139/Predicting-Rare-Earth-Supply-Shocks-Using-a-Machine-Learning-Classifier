import pandas as pd
import numpy as np

# ── LOAD ALL THREE DATASETS ───────────────────────────────────────
gdelt  = pd.read_csv("gdelt_ree_daily.csv",    dtype={"date": str})
media  = pd.read_csv("chinese_media_daily.csv", dtype={"date": str})
prices = pd.read_csv("price_data_daily.csv",    dtype={"date": str})

print(f"GDELT rows:  {len(gdelt)}")
print(f"Media rows:  {len(media)}")
print(f"Prices rows: {len(prices)}")

# ── MERGE ON DATE ─────────────────────────────────────────────────
df = prices.merge(gdelt,  on="date", how="left")
df = df.merge(media, on="date", how="left")

# ── CLEAN ─────────────────────────────────────────────────────────
df["date"] = df["date"].astype(str)
df = df.sort_values("date").reset_index(drop=True)
df = df.ffill()
df = df.fillna(0)

print(f"\nMerged dataset: {len(df)} rows, {len(df.columns)} columns")
print(df.head(5))

# ── CREATE LABEL ──────────────────────────────────────────────────
# label = 1 if REMX rises more than 8% in the next 14 days
df["REMX"] = pd.to_numeric(df["REMX"], errors="coerce")
df["REMX_fwd_14d"] = (df["REMX"].shift(-14) / df["REMX"] - 1)
df["label"] = (df["REMX_fwd_14d"] > 0.15).astype(int)

# ── CHECK LABELS AROUND CRISIS EVENTS ────────────────────────────
crisis_dates = ["20230703", "20250404", "20251009"]

print("\n── Label check around crisis events ──")
for cd in crisis_dates:
    cd_dt = pd.to_datetime(cd, format="%Y%m%d")
    start = (cd_dt - pd.Timedelta(days=14)).strftime("%Y%m%d")
    end   = (cd_dt + pd.Timedelta(days=21)).strftime("%Y%m%d")
    window = df[df["date"].between(start, end)]
    label_1_count = window["label"].sum()
    print(f"\nAround {cd} (window {start} to {end}):")
    print(f"  Rows in window: {len(window)}")
    print(f"  label=1 count:  {label_1_count}")
    print(window[["date","REMX","REMX_fwd_14d","label"]].to_string())

# ── SUMMARY ───────────────────────────────────────────────────────
print(f"\nTotal label=1 (crisis) rows: {df['label'].sum()}")
print(f"Total label=0 (normal) rows: {(df['label']==0).sum()}")
print(f"Crisis rate: {df['label'].mean()*100:.1f}%")

# ── SAVE ──────────────────────────────────────────────────────────
df.to_csv("merged_dataset.csv", index=False)
print("\nSaved to merged_dataset.csv")