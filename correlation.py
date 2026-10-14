import pandas as pd
import numpy as np

df = pd.read_csv("merged_dataset.csv", dtype={"date": str})

for col in ["REMX", "BEL.NS", "DIXON.NS"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["REMX"] = df["REMX"].replace(0, np.nan)

# compute 14-day returns for all three
df["remx_ret"]  = df["REMX"].pct_change(14)
df["bel_ret"]   = df["BEL.NS"].pct_change(14)
df["dixon_ret"] = df["DIXON.NS"].pct_change(14)

df_clean = df.dropna(subset=["remx_ret","bel_ret","dixon_ret"])

# correlation matrix
corr = df_clean[["remx_ret","bel_ret","dixon_ret"]].corr()
print("── 14-day return correlation matrix ──")
print(corr.round(3))

# also split by label
print("\n── Correlation during NORMAL days (label=0) ──")
print(df_clean[df_clean["label"]==0][["remx_ret","bel_ret","dixon_ret"]].corr().round(3))

print("\n── Correlation during CRISIS days (label=1) ──")
print(df_clean[df_clean["label"]==1][["remx_ret","bel_ret","dixon_ret"]].corr().round(3))