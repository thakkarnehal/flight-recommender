"""Explore BTS flight data: print schema, distributions, and key stats."""

import warnings
warnings.filterwarnings("ignore")

import glob
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

NEEDED_COLS = [
    "FlightDate", "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepDelay", "ArrDelay", "Distance",
    "DayOfWeek", "Month", "Year", "Cancelled", "Diverted",
]


def load_sample(n_files: int = 2) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("bts_*.csv"))[:n_files]
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in NEEDED_COLS, low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_all() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("bts_*.csv"))
    print(f"Loading {len(files)} files...")
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in NEEDED_COLS, low_memory=False)
        dfs.append(df)
        print(f"  {f.name}: {len(df):,} rows")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(combined):,} rows\n")
    return combined


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    # ── Schema peek (one file) ──────────────────────────────────
    sep("SCHEMA (first file)")
    sample_file = sorted(RAW_DIR.glob("bts_*.csv"))[0]
    peek = pd.read_csv(sample_file, nrows=3)
    print("All columns in raw file:")
    for i, c in enumerate(peek.columns.tolist()):
        print(f"  {i:3d}  {c}")
    print(f"\nRows sampled: {len(peek)}")
    print(peek.to_string())

    # ── Load all data ───────────────────────────────────────────
    sep("LOADING ALL DATA")
    df = load_all()

    # ── Basic shape & nulls ────────────────────────────────────
    sep("SHAPE & MISSING VALUES")
    print(f"Shape: {df.shape}")
    null_pct = (df.isnull().sum() / len(df) * 100).round(2)
    print("\nNull % per column:")
    print(null_pct[null_pct > 0].to_string())

    # Drop cancelled/diverted for most stats
    active = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy()
    print(f"\nNon-cancelled, non-diverted flights: {len(active):,} ({len(active)/len(df)*100:.1f}%)")

    # ── Carriers ───────────────────────────────────────────────
    sep("CARRIER DISTRIBUTION")
    carrier_counts = active["Reporting_Airline"].value_counts()
    print(carrier_counts.to_string())
    print(f"\nTotal unique carriers: {active['Reporting_Airline'].nunique()}")

    # ── Delay statistics ───────────────────────────────────────
    sep("DELAY STATISTICS (active flights)")
    for col in ["DepDelay", "ArrDelay"]:
        d = active[col].dropna()
        print(f"\n{col}:")
        print(f"  mean={d.mean():.1f} min, median={d.median():.1f} min, "
              f"std={d.std():.1f} min")
        print(f"  pct delayed (>0 min): {(d > 0).mean()*100:.1f}%")
        print(f"  pct delayed (>15 min): {(d > 15).mean()*100:.1f}%")
        print(f"  pct delayed (>30 min): {(d > 30).mean()*100:.1f}%")
        print(f"  pct early (<0 min):  {(d < 0).mean()*100:.1f}%")

    # ── Delay by carrier ───────────────────────────────────────
    sep("MEAN ARRIVAL DELAY BY CARRIER")
    delay_by_carrier = (
        active.groupby("Reporting_Airline")["ArrDelay"]
        .agg(["mean", "count", lambda x: (x > 30).mean() * 100])
        .rename(columns={"mean": "avg_delay_min", "count": "n_flights",
                         "<lambda_0>": "pct_delay_30min"})
        .sort_values("avg_delay_min", ascending=False)
    )
    print(delay_by_carrier.round(2).to_string())

    # ── Top routes ─────────────────────────────────────────────
    sep("TOP 30 ROUTES BY FLIGHT VOLUME")
    active["route"] = active["Origin"] + "-" + active["Dest"]
    route_counts = active["route"].value_counts().head(30)
    print(route_counts.to_string())
    print(f"\nTotal unique routes: {active['route'].nunique()}")

    # ── Time of day ────────────────────────────────────────────
    sep("FLIGHT VOLUME BY TIME OF DAY")
    dep = active["CRSDepTime"].dropna().astype(int)

    def time_bucket(t):
        h = t // 100
        if 5 <= h < 12:
            return "morning (5am-12pm)"
        elif 12 <= h < 18:
            return "afternoon (12pm-6pm)"
        elif 18 <= h < 22:
            return "evening (6pm-10pm)"
        else:
            return "redeye (10pm-5am)"

    active["time_bucket"] = dep.map(time_bucket)
    tod = active["time_bucket"].value_counts()
    print(tod.to_string())
    print("\nPercentages:")
    print((tod / tod.sum() * 100).round(1).to_string())

    # ── Avg delay by time of day ────────────────────────────────
    sep("MEAN ARRIVAL DELAY BY TIME OF DAY")
    delay_tod = active.groupby("time_bucket")["ArrDelay"].agg(["mean", "count"])
    print(delay_tod.round(2).to_string())

    # ── Season ─────────────────────────────────────────────────
    sep("FLIGHT VOLUME BY SEASON")
    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall",   10: "fall",  11: "fall",
    }
    active["season"] = active["Month"].map(season_map)
    season = active["season"].value_counts()
    print(season.to_string())
    delay_season = active.groupby("season")["ArrDelay"].mean().round(2)
    print("\nMean arrival delay by season:")
    print(delay_season.to_string())

    # ── Distance tiers ─────────────────────────────────────────
    sep("DISTANCE TIER DISTRIBUTION")
    bins = [0, 500, 1500, 10000]
    labels = ["short (<500mi)", "medium (500-1500mi)", "long (>1500mi)"]
    active["dist_tier"] = pd.cut(active["Distance"], bins=bins, labels=labels)
    dist = active["dist_tier"].value_counts()
    print(dist.to_string())
    delay_dist = active.groupby("dist_tier", observed=True)["ArrDelay"].mean().round(2)
    print("\nMean arrival delay by distance tier:")
    print(delay_dist.to_string())

    # ── Monthly volume ─────────────────────────────────────────
    sep("MONTHLY FLIGHT VOLUME")
    active["year_month"] = active["Year"].astype(str) + "-" + active["Month"].astype(str).str.zfill(2)
    monthly = active.groupby("year_month")["route"].count().sort_index()
    print(monthly.to_string())

    # ── Summary stats ──────────────────────────────────────────
    sep("SUMMARY")
    print(f"Total flights (incl. cancelled): {len(df):,}")
    print(f"Active flights (used for model): {len(active):,}")
    print(f"Date range: {active['FlightDate'].min()} → {active['FlightDate'].max()}")
    print(f"Unique carriers: {active['Reporting_Airline'].nunique()}")
    print(f"Unique airports: {pd.concat([active['Origin'], active['Dest']]).nunique()}")
    print(f"Unique routes:   {active['route'].nunique()}")
    print(f"Overall delay rate (>30 min): {(active['ArrDelay'] > 30).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
