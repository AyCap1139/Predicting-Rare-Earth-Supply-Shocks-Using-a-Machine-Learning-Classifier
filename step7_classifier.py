import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

df = pd.read_csv("features.csv", dtype={"date": str})
df = df.sort_values("date").reset_index(drop=True)

feature_cols = [
    "gdelt_7d_count", "gdelt_7d_tone", "tone_velocity",
    "media_7d_count", "media_7d_tone", "spike_indicator",
    "xinhua_7d", "remx_momentum_30d", "bel_return_14d",
    "dixon_return_14d"
]

X = df[feature_cols].values
y = df["label"].values
dates = df["date"].values

# ── TEMPORAL SPLIT ────────────────────────────────────────────────
split_date = "20250101"
train_mask = df["date"] < split_date
test_mask  = df["date"] >= split_date

X_train = X[train_mask]
y_train = y[train_mask]
X_test  = X[test_mask]
y_test  = y[test_mask]
dates_test = dates[test_mask]

print(f"Train: {len(X_train)} rows ({y_train.sum()} crisis)")
print(f"Test:  {len(X_test)} rows ({y_test.sum()} crisis)")

# ── STANDARDISE ───────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── TRAIN ─────────────────────────────────────────────────────────
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# ── PREDICT with threshold=0.20 ───────────────────────────────────
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred      = (y_pred_prob >= 0.20).astype(int)

# ── EVALUATE ──────────────────────────────────────────────────────
print("\n── Confusion Matrix ──")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nTP={cm[1,1]}  FP={cm[0,1]}  FN={cm[1,0]}  TN={cm[0,0]}")

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred,
      target_names=["Normal", "Crisis"]))

auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC: {auc:.4f}")

# ── FEATURE COEFFICIENTS ──────────────────────────────────────────
print("\n── Feature Coefficients ──")
coefs = pd.DataFrame({
    "feature":     feature_cols,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", ascending=False)
print(coefs.to_string())

# ── THRESHOLD TUNING TABLE ────────────────────────────────────────
print("\n── Threshold Tuning ──")
for threshold in [0.5, 0.4, 0.3, 0.2, 0.15]:
    y_t = (y_pred_prob >= threshold).astype(int)
    cm_t = confusion_matrix(y_test, y_t)
    tp = cm_t[1,1]
    fp = cm_t[0,1]
    fn = cm_t[1,0]
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"Threshold={threshold:.2f} | "
          f"Recall={recall:.2f} | "
          f"Precision={precision:.2f} | "
          f"TP={tp} FP={fp} FN={fn}")

# ── ROC CURVE ─────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(7,5))
ax.set_facecolor("#111")
fig.patch.set_facecolor("#1a1a1a")
ax.plot(fpr, tpr, color="#534AB7", lw=2,
        label=f"ROC curve (AUC={auc:.3f})")
ax.plot([0,1],[0,1], color="#555", ls="--", label="Random guess")
ax.set_xlabel("False Positive Rate", color="#888")
ax.set_ylabel("True Positive Rate", color="#888")
ax.set_title("REE Crisis Classifier — ROC Curve", color="#eee")
ax.legend(facecolor="#2a2a2a", labelcolor="#ccc")
ax.tick_params(colors="#666")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("ROC curve saved to roc_curve.png")

# ── SAVE PREDICTIONS ──────────────────────────────────────────────
results = pd.DataFrame({
    "date":        dates_test,
    "actual":      y_test,
    "predicted":   y_pred,
    "probability": y_pred_prob
})
results.to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")