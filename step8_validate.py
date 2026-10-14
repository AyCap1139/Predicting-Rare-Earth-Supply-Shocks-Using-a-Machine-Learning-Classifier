import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── LOAD DATA ─────────────────────────────────────────────────────
predictions = pd.read_csv("predictions.csv", dtype={"date": str})
features    = pd.read_csv("features.csv",    dtype={"date": str})
prices      = pd.read_csv("price_data_daily.csv", dtype={"date": str})

# merge predictions with prices
df = predictions.merge(prices[["date","REMX"]], on="date", how="left")
df["REMX"] = pd.to_numeric(df["REMX"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

# ── CRISIS EVENTS ─────────────────────────────────────────────────
crisis_events = {
    "Apr 2025 — 7 REEs restricted":  "20250404",
    "Oct 2025 — 12 REEs restricted": "20251009",
}

print("── Validation Results ──\n")

for event_name, crisis_date in crisis_events.items():
    cd_dt = pd.to_datetime(crisis_date, format="%Y%m%d")
    start = (cd_dt - pd.Timedelta(days=30)).strftime("%Y%m%d")
    end   = (cd_dt + pd.Timedelta(days=21)).strftime("%Y%m%d")

    window = df[df["date"].between(start, end)].copy()

    if len(window) == 0:
        print(f"{event_name}: no data in window")
        continue

    # find first classifier alert before crisis date
    before_crisis = window[window["date"] <= crisis_date]
    alerts_before = before_crisis[before_crisis["predicted"] == 1]

    if len(alerts_before) > 0:
        first_alert = alerts_before.iloc[0]["date"]
        first_alert_dt = pd.to_datetime(first_alert, format="%Y%m%d")
        lead_days = (cd_dt - first_alert_dt).days
        print(f"✅ {event_name}")
        print(f"   Crisis date:   {crisis_date}")
        print(f"   First alert:   {first_alert}")
        print(f"   Lead time:     {lead_days} days before crisis")
        print(f"   Alerts in window: {len(alerts_before)}")
    else:
        print(f"❌ {event_name}")
        print(f"   Crisis date: {crisis_date}")
        print(f"   No alerts fired before crisis date")

    # REMX move
    crisis_row = window[window["date"] <= crisis_date]
    if len(crisis_row) > 0:
        remx_at_crisis = crisis_row.iloc[-1]["REMX"]
        remx_21d_later = window[window["date"] > crisis_date]["REMX"]
        if len(remx_21d_later) > 0:
            remx_peak = remx_21d_later.max()
            remx_move = (remx_peak - remx_at_crisis) / remx_at_crisis
            print(f"   REMX at crisis: ${remx_at_crisis:.2f}")
            print(f"   REMX peak +21d: ${remx_peak:.2f}")
            print(f"   REMX move:      {remx_move*100:.1f}%")
    print()

# ── PLOT: classifier score vs REMX price ─────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#1a1a1a")

for idx, (event_name, crisis_date) in enumerate(crisis_events.items()):
    cd_dt = pd.to_datetime(crisis_date, format="%Y%m%d")
    start = (cd_dt - pd.Timedelta(days=30)).strftime("%Y%m%d")
    end   = (cd_dt + pd.Timedelta(days=21)).strftime("%Y%m%d")
    window = df[df["date"].between(start, end)].copy()

    if len(window) == 0:
        continue

    ax1 = fig.add_subplot(2, 1, idx+1)
    ax1.set_facecolor("#111")
    ax2 = ax1.twinx()

    # REMX price
    ax1.plot(range(len(window)), window["REMX"].values,
             color="#1D9E75", lw=2, label="REMX price")
    ax1.set_ylabel("REMX Price ($)", color="#1D9E75")
    ax1.tick_params(axis="y", colors="#1D9E75")

    # classifier probability
    ax2.plot(range(len(window)), window["probability"].values,
             color="#534AB7", lw=1.5, alpha=0.8, label="Crisis probability")
    ax2.axhline(0.20, color="#EF9F27", ls="--", lw=1,
                label="Threshold (0.20)")
    ax2.set_ylabel("P(Crisis)", color="#534AB7")
    ax2.tick_params(axis="y", colors="#534AB7")
    ax2.set_ylim(0, 1)

    # crisis date vertical line
    crisis_idx = window[window["date"] == crisis_date].index
    if len(crisis_idx) > 0:
        ci = window.index.get_loc(crisis_idx[0])
        ax1.axvline(ci, color="#E24B4A", lw=2, ls="-",
                    label=f"Crisis: {crisis_date}")

    # alert markers
    alerts = window[window["predicted"] == 1]
    for _, alert_row in alerts.iterrows():
        ai = window.index.get_loc(alert_row.name)
        ax1.axvline(ai, color="#EF9F27", lw=0.5, alpha=0.4)

    # x ticks
    tick_step = max(1, len(window)//8)
    ax1.set_xticks(range(0, len(window), tick_step))
    ax1.set_xticklabels(
        window["date"].values[::tick_step],
        rotation=45, color="#666", fontsize=8
    )
    ax1.set_title(event_name, color="#eee", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2,
               facecolor="#2a2a2a", labelcolor="#ccc", fontsize=8)

plt.tight_layout()
plt.savefig("event_validation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Event validation chart saved to event_validation.png")