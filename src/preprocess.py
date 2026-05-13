"""
Feature engineering, cohort construction, interaction matrix, and train/val/test split.
Saves processed data to data/processed/.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

NEEDED_COLS = [
    "FlightDate", "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepDelay", "ArrDelay", "Distance",
    "DayOfWeek", "Month", "Year", "Cancelled", "Diverted",
]

# ── Carrier → budget tier ──────────────────────────────────────────────────────
LCC_CARRIERS       = {"NK", "F9", "G4", "WN", "B6"}           # economy
HYBRID_CARRIERS    = {"AS", "MQ", "OO", "YX"}                  # premium_economy
LEGACY_CARRIERS    = {"AA", "DL", "UA", "HA", "9E"}            # business


def time_bucket(crs_dep: pd.Series) -> pd.Series:
    h = crs_dep.astype(int) // 100
    return pd.cut(
        h,
        bins=[-1, 4, 11, 17, 21, 24],
        labels=["redeye", "morning", "afternoon", "evening", "redeye_late"],
    ).replace("redeye_late", "redeye").astype(str)


def season_from_month(month: pd.Series) -> pd.Series:
    return month.map({
        12: "winter", 1: "winter",  2: "winter",
        3:  "spring", 4: "spring",  5: "spring",
        6:  "summer", 7: "summer",  8: "summer",
        9:  "fall",   10: "fall",   11: "fall",
    })


def distance_tier(dist: pd.Series) -> pd.Series:
    return pd.cut(
        dist,
        bins=[0, 500, 1500, 1e6],
        labels=["short", "medium", "long"],
    ).astype(str)


def budget_tier(carrier: pd.Series) -> pd.Series:
    def _map(c):
        if c in LCC_CARRIERS:
            return "economy"
        if c in HYBRID_CARRIERS:
            return "premium_economy"
        return "business"
    return carrier.map(_map)


def freq_tier(carrier: pd.Series, dow: pd.Series) -> pd.Series:
    """
    Proxy for trip-frequency using carrier type and day-of-week.
    Mon–Thu = likely business travel; Fri–Sun = likely leisure.
    """
    is_legacy = carrier.isin(LEGACY_CARRIERS)
    is_hybrid  = carrier.isin(HYBRID_CARRIERS)
    is_weekday = dow.isin([1, 2, 3, 4])   # BTS: 1=Mon … 7=Sun

    result = pd.Series("low", index=carrier.index)
    result[is_weekday & is_legacy]             = "frequent"
    result[is_weekday & is_hybrid]             = "high"
    result[(~is_weekday) & is_legacy]          = "high"
    result[is_weekday & (~is_legacy) & (~is_hybrid)] = "medium"
    # LCC + weekend stays "low"
    return result


FREQ_MAP   = {"low": 0, "medium": 1, "high": 2, "frequent": 3}
BUDGET_MAP = {"economy": 0, "premium_economy": 1, "business": 2}
TIME_MAP   = {"morning": 0, "afternoon": 1, "evening": 2, "redeye": 3}
SEASON_MAP = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}
DIST_MAP   = {"short": 0, "medium": 1, "long": 2}


def load_all() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("bts_*.csv"))
    print(f"Loading {len(files)} monthly files...")
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in NEEDED_COLS, low_memory=False)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} total rows")
    return combined


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Filter, clean, and engineer all features."""
    # Keep non-cancelled, non-diverted, valid delay
    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy()
    df = df.dropna(subset=["ArrDelay", "DepDelay", "Distance", "CRSDepTime"])

    df["route"]       = df["Origin"] + "-" + df["Dest"]
    df["time_bucket"] = time_bucket(df["CRSDepTime"])
    df["season"]      = season_from_month(df["Month"])
    df["dist_tier"]   = distance_tier(df["Distance"])
    df["budget_tier"] = budget_tier(df["Reporting_Airline"])
    df["freq_tier"]   = freq_tier(df["Reporting_Airline"], df["DayOfWeek"])

    # Delay label: 1 if arrival delay > 30 min
    df["delay_label"] = (df["ArrDelay"] > 30).astype(int)

    # Cohort ID = string key for (freq, budget, time_pref)
    # time_pref = time_bucket of THIS flight (not user's pref — that's the same thing)
    df["cohort_id"] = (
        df["freq_tier"] + "|" + df["budget_tier"] + "|" + df["time_bucket"]
    )

    # Integer-encoded user features
    df["freq_idx"]   = df["freq_tier"].map(FREQ_MAP)
    df["budget_idx"] = df["budget_tier"].map(BUDGET_MAP)
    df["time_idx"]   = df["time_bucket"].map(TIME_MAP)
    df["season_idx"] = df["season"].map(SEASON_MAP)
    df["dist_idx"]   = df["dist_tier"].map(DIST_MAP)

    # Flight item key: route × carrier × time_bucket
    df["item_key"] = df["route"] + "|" + df["Reporting_Airline"] + "|" + df["time_bucket"]

    return df


def filter_sparse(df: pd.DataFrame, min_item_flights: int = 100) -> pd.DataFrame:
    """Drop extremely rare items to keep the embedding table tractable."""
    item_counts = df["item_key"].value_counts()
    keep = item_counts[item_counts >= min_item_flights].index
    before = len(df)
    df = df[df["item_key"].isin(keep)]
    print(f"Item filter (>={min_item_flights} flights): {before:,} → {len(df):,} rows")
    print(f"Unique items after filter: {df['item_key'].nunique()}")
    return df


def build_interaction_matrix(df: pd.DataFrame):
    """
    Each row = one cohort.
    Each col = one flight item.
    Value = 1 if that cohort flew that item at least once.
    Returns list of (cohort_features, item_idx, delay_label) positive pairs.
    """
    cohort_item = df.groupby(["cohort_id", "item_key"]).agg(
        freq_idx   = ("freq_idx",   "first"),
        budget_idx = ("budget_idx", "first"),
        time_idx   = ("time_idx",   "first"),
        delay_label = ("delay_label", "mean"),  # fraction of those flights delayed
        n_flights  = ("ArrDelay",   "count"),
    ).reset_index()

    cohort_item["delay_label"] = (cohort_item["delay_label"] > 0.15).astype(int)
    print(f"Cohort-item pairs (positives): {len(cohort_item):,}")
    print(f"Unique cohorts: {cohort_item['cohort_id'].nunique()}")
    return cohort_item


def negative_sample(
    positives: pd.DataFrame,
    all_items: np.ndarray,
    item_counts: dict,
    n_neg: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Popularity-weighted negative sampling.
    Hard negatives are drawn proportional to flight volume — popular routes
    the cohort didn't take are more informative than random obscure ones.
    Pre-computes per-cohort candidate sets (32 cohorts) for efficiency.
    """
    rng = np.random.default_rng(seed)
    pos_set = positives.groupby("cohort_id")["item_key"].apply(set).to_dict()

    # Pre-compute per-cohort candidate arrays and popularity weights
    all_items_arr = np.array(all_items)
    raw_weights   = np.array([item_counts.get(i, 1) for i in all_items], dtype=float)

    cohort_candidates = {}
    for cid, exclude in pos_set.items():
        mask = np.array([item not in exclude for item in all_items])
        cand_items   = all_items_arr[mask]
        cand_weights = raw_weights[mask]
        cand_weights = cand_weights / cand_weights.sum()
        cohort_candidates[cid] = (cand_items, cand_weights)

    negs = []
    for _, row in positives.iterrows():
        cid = row["cohort_id"]
        cand_items, cand_weights = cohort_candidates[cid]
        n_sample = min(n_neg, len(cand_items))
        sampled_idx = rng.choice(len(cand_items), size=n_sample, replace=False, p=cand_weights)
        for item_k in cand_items[sampled_idx]:
            negs.append({
                "cohort_id":   cid,
                "item_key":    item_k,
                "freq_idx":    row["freq_idx"],
                "budget_idx":  row["budget_idx"],
                "time_idx":    row["time_idx"],
                "delay_label": 0,
                "label":       0,
                "n_flights":   0,
            })
    neg_df = pd.DataFrame(negs)
    positives = positives.copy()
    positives["label"] = 1
    combined = pd.concat([positives, neg_df], ignore_index=True)
    print(f"After negative sampling: {len(combined):,} rows "
          f"({len(positives):,} pos + {len(neg_df):,} neg)")
    return combined


def train_val_test_split(df: pd.DataFrame, seed: int = 42):
    """
    Random 80/10/10 split stratified by label.
    Cohort-item pairs are archetypes (not temporal sequences), so random
    split is correct here — temporal ordering is captured by season features.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    labels = df["label"].values

    train_idx, tmp_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    tmp_labels = labels[tmp_idx]
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=tmp_labels, random_state=seed)

    train = df.iloc[train_idx].reset_index(drop=True)
    val   = df.iloc[val_idx].reset_index(drop=True)
    test  = df.iloc[test_idx].reset_index(drop=True)

    print(f"\nSplit sizes (random 80/10/10):")
    print(f"  Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val):,}   ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test):,}   ({len(test)/len(df)*100:.1f}%)")
    return train, val, test


def encode_items(df_all: pd.DataFrame):
    """Fit a LabelEncoder on all items, encode item_key → item_idx."""
    le = LabelEncoder()
    le.fit(df_all["item_key"].unique())
    return le


def build_item_feature_lookup(df_all: pd.DataFrame, le: LabelEncoder) -> dict:
    """
    For each item (route×carrier×time_bucket), store:
      route_idx, carrier_idx, season_idx, dist_idx
    Used at inference time.
    """
    route_le   = LabelEncoder().fit(df_all["route"].unique())
    carrier_le = LabelEncoder().fit(df_all["Reporting_Airline"].unique())

    # Most common month for each item — captures seasonal patterns per route/carrier/time
    item_peak_month = (
        df_all.groupby("item_key")["Month"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .rename("peak_month")
    )

    item_feats = (
        df_all.groupby("item_key")[
            ["route", "Reporting_Airline", "time_bucket", "season", "dist_tier"]
        ]
        .first()
        .reset_index()
        .merge(item_peak_month, on="item_key")
    )
    item_feats["item_idx"]    = le.transform(item_feats["item_key"])
    item_feats["route_idx"]   = route_le.transform(item_feats["route"])
    item_feats["carrier_idx"] = carrier_le.transform(item_feats["Reporting_Airline"])
    item_feats["season_idx"]  = item_feats["season"].map(SEASON_MAP)
    item_feats["dist_idx"]    = item_feats["dist_tier"].map(DIST_MAP)
    item_feats["month_idx"]   = item_feats["peak_month"] - 1  # 0-11

    lookup = item_feats.set_index("item_idx")[
        ["item_key", "route", "Reporting_Airline", "time_bucket",
         "route_idx", "carrier_idx", "season_idx", "dist_idx", "month_idx"]
    ].to_dict("index")

    return lookup, route_le, carrier_le


def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    sep("LOADING & FEATURE ENGINEERING")
    raw = load_all()
    df  = build_features(raw)

    sep("ITEM FILTERING")
    df = filter_sparse(df, min_item_flights=100)

    sep("COHORT DISTRIBUTIONS")
    print("Freq tier distribution:")
    print(df["freq_tier"].value_counts().to_string())
    print("\nBudget tier distribution:")
    print(df["budget_tier"].value_counts().to_string())
    print("\nTime pref distribution:")
    print(df["time_bucket"].value_counts().to_string())
    print("\nCohort (combined) distribution (top 20):")
    print(df["cohort_id"].value_counts().head(20).to_string())

    sep("BUILDING INTERACTION MATRIX")
    positives = build_interaction_matrix(df)

    sep("NEGATIVE SAMPLING (4:1, popularity-weighted)")
    all_items   = df["item_key"].unique()
    item_counts = df["item_key"].value_counts().to_dict()
    interactions = negative_sample(positives, all_items, item_counts, n_neg=4)

    sep("ENCODING ITEMS")
    le = encode_items(df)
    interactions["item_idx"] = le.transform(interactions["item_key"])

    n_items    = len(le.classes_)
    n_routes   = df["route"].nunique()
    n_carriers = df["Reporting_Airline"].nunique()

    print(f"Vocabulary sizes:")
    print(f"  Items:    {n_items}")
    print(f"  Routes:   {n_routes}")
    print(f"  Carriers: {n_carriers}")

    sep("TRAIN/VAL/TEST SPLIT")
    train, val, test = train_val_test_split(interactions)

    sep("ITEM FEATURE LOOKUP")
    item_lookup, route_le, carrier_le = build_item_feature_lookup(df, le)

    # Encode route & carrier in train/val/test splits
    for split_df in [train, val, test]:
        split_df["route"]       = split_df["item_key"].str.split("|").str[0]
        split_df["carrier"]     = split_df["item_key"].str.split("|").str[1]
        split_df["route_idx"]   = route_le.transform(split_df["route"])
        split_df["carrier_idx"] = carrier_le.transform(split_df["carrier"])
        # Look up season_idx from the item_lookup for each item_idx
        split_df["season_idx"]  = split_df["item_idx"].map(
            lambda i: item_lookup.get(i, {}).get("season_idx", 1)
        )
        split_df["month_idx"]   = split_df["item_idx"].map(
            lambda i: item_lookup.get(i, {}).get("month_idx", 6)
        )

    sep("SAVING")
    train.to_parquet(PROC_DIR / "train.parquet", index=False)
    val.to_parquet(PROC_DIR / "val.parquet",     index=False)
    test.to_parquet(PROC_DIR / "test.parquet",   index=False)

    with open(PROC_DIR / "item_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(PROC_DIR / "route_encoder.pkl", "wb") as f:
        pickle.dump(route_le, f)
    with open(PROC_DIR / "carrier_encoder.pkl", "wb") as f:
        pickle.dump(carrier_le, f)
    with open(PROC_DIR / "item_lookup.pkl", "wb") as f:
        pickle.dump(item_lookup, f)

    meta = {
        "n_items":    n_items,
        "n_routes":   n_routes,
        "n_carriers": n_carriers,
        "n_freq":     4,
        "n_budget":   3,
        "n_time":     4,
        "n_season":   4,
        "n_month":    12,
        "freq_map":   FREQ_MAP,
        "budget_map": BUDGET_MAP,
        "time_map":   TIME_MAP,
        "season_map": SEASON_MAP,
        "train_size": len(train),
        "val_size":   len(val),
        "test_size":  len(test),
    }
    with open(PROC_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved to data/processed/:")
    for p in sorted(PROC_DIR.iterdir()):
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:30s}  {size_mb:.1f} MB")

    sep("FINAL SUMMARY")
    print(f"Train interactions:  {len(train):,}")
    print(f"Val interactions:    {len(val):,}")
    print(f"Test interactions:   {len(test):,}")
    print(f"Positive rate train: {train['label'].mean()*100:.1f}%")
    print(f"Delay rate train:    {train['delay_label'].mean()*100:.1f}%")
    print(f"Unique cohorts:      {interactions['cohort_id'].nunique()}")
    print(f"Unique items:        {n_items}")


if __name__ == "__main__":
    main()
