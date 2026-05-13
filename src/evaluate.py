"""Evaluation: HR@10, NDCG@10 for recommendation; AUC/precision/recall for delay."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pandas as pd
from pathlib import Path


def _dcg(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k]
    if rel.sum() == 0:
        return 0.0
    return float(np.sum(rel / np.log2(np.arange(1, len(rel) + 1) + 1)))


def _ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order  = np.argsort(scores)[::-1]
    ranked = labels[order]
    ideal  = np.sort(labels)[::-1]
    return _dcg(ranked, k) / (_dcg(ideal, k) + 1e-9)


def _hit_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].sum() > 0)


def _score_batch(model, device, freq, budget, time_p, route, carrier, season, month,
                 batch_size=4096):
    n = len(freq)
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s, _, _ = model.score(
            freq[start:end], budget[start:end], time_p[start:end],
            route[start:end], carrier[start:end], season[start:end], month[start:end],
        )
        scores[start:end] = s.cpu().numpy()
    return scores


@torch.no_grad()
def evaluate_recommendation(model, parquet_path: str, device: torch.device, k: int = 10):
    """
    Standard NCF evaluation protocol: for each positive test interaction,
    sample 99 negatives and ask if the positive ranks in the top-k out of 100.
    This is the correct methodology used in NCF literature (He et al. 2017).
    """
    model.eval()
    rng = np.random.default_rng(42)

    df      = pd.read_parquet(parquet_path)
    pos     = df[df["label"] == 1].reset_index(drop=True)
    neg_pool = df[df["label"] == 0].reset_index(drop=True)

    # Build per-cohort negative pools
    neg_by_cohort = neg_pool.groupby("cohort_id")

    hits, ndcgs = [], []

    for _, row in pos.iterrows():
        cid = row["cohort_id"]

        # Sample 99 negatives from this cohort's negative pool
        try:
            pool = neg_by_cohort.get_group(cid)
        except KeyError:
            continue
        sample_idx = rng.choice(len(pool), size=min(99, len(pool)), replace=False)
        neg_sample  = pool.iloc[sample_idx]

        # Combine 1 positive + 99 negatives
        combined = pd.concat([row.to_frame().T, neg_sample], ignore_index=True)
        int_cols = ["freq_idx", "budget_idx", "time_idx", "route_idx",
                    "carrier_idx", "season_idx", "month_idx", "label", "delay_label"]
        combined[int_cols] = combined[int_cols].astype(int)
        labels   = combined["label"].values.astype(float)

        freq    = torch.tensor(combined["freq_idx"].values,    dtype=torch.long).to(device)
        budget  = torch.tensor(combined["budget_idx"].values,  dtype=torch.long).to(device)
        time_p  = torch.tensor(combined["time_idx"].values,    dtype=torch.long).to(device)
        route   = torch.tensor(combined["route_idx"].values,   dtype=torch.long).to(device)
        carrier = torch.tensor(combined["carrier_idx"].values, dtype=torch.long).to(device)
        season  = torch.tensor(combined["season_idx"].values,  dtype=torch.long).to(device)
        month   = torch.tensor(combined["month_idx"].values,   dtype=torch.long).to(device)

        scores = _score_batch(model, device, freq, budget, time_p, route, carrier, season, month)

        hits.append(_hit_at_k(scores, labels, k))
        ndcgs.append(_ndcg_at_k(scores, labels, k))

    return {
        f"HR@{k}":   float(np.mean(hits)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "n_evaluated": len(hits),
    }


@torch.no_grad()
def evaluate_delay(model, parquet_path: str, device: torch.device):
    """AUC-ROC, precision, recall for delay prediction (positives only)."""
    model.eval()
    df  = pd.read_parquet(parquet_path)
    pos = df[df["label"] == 1].copy()

    if len(pos) == 0 or pos["delay_label"].nunique() < 2:
        return {"AUC": float("nan"), "Precision": float("nan"), "Recall": float("nan")}

    freq    = torch.tensor(pos["freq_idx"].values,    dtype=torch.long).to(device)
    budget  = torch.tensor(pos["budget_idx"].values,  dtype=torch.long).to(device)
    time_p  = torch.tensor(pos["time_idx"].values,    dtype=torch.long).to(device)
    route   = torch.tensor(pos["route_idx"].values,   dtype=torch.long).to(device)
    carrier = torch.tensor(pos["carrier_idx"].values, dtype=torch.long).to(device)
    season  = torch.tensor(pos["season_idx"].values,  dtype=torch.long).to(device)
    month   = torch.tensor(pos["month_idx"].values,   dtype=torch.long).to(device)

    n = len(pos)
    delay_probs = np.empty(n, dtype=np.float32)
    batch_size  = 4096
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        _, _, dp = model.score(
            freq[start:end], budget[start:end], time_p[start:end],
            route[start:end], carrier[start:end], season[start:end], month[start:end],
        )
        delay_probs[start:end] = dp.cpu().numpy()

    true      = pos["delay_label"].values
    threshold = np.percentile(delay_probs, 100 * (1 - true.mean()))
    pred      = (delay_probs >= threshold).astype(int)

    return {
        "AUC":       float(roc_auc_score(true, delay_probs)),
        "Precision": float(precision_score(true, pred, zero_division=0)),
        "Recall":    float(recall_score(true, pred, zero_division=0)),
    }


def popularity_baseline(parquet_path: str, k: int = 10) -> dict:
    """
    Standard 100-item protocol applied to popularity baseline.
    """
    rng        = np.random.default_rng(42)
    train_path = Path(parquet_path).parent / "train.parquet"
    train      = pd.read_parquet(train_path)
    pop_scores = (
        train[train["label"] == 1]
        .groupby("item_idx")["label"].count()
        .rename("pop")
    )

    df      = pd.read_parquet(parquet_path)
    df      = df.merge(pop_scores, on="item_idx", how="left").fillna({"pop": 0})
    pos     = df[df["label"] == 1].reset_index(drop=True)
    neg_pool = df[df["label"] == 0]
    neg_by_cohort = neg_pool.groupby("cohort_id")

    hits, ndcgs = [], []
    for _, row in pos.iterrows():
        cid = row["cohort_id"]
        try:
            pool = neg_by_cohort.get_group(cid)
        except KeyError:
            continue
        sample_idx = rng.choice(len(pool), size=min(99, len(pool)), replace=False)
        neg_sample  = pool.iloc[sample_idx]
        combined    = pd.concat([row.to_frame().T, neg_sample], ignore_index=True)
        scores      = combined["pop"].values.astype(float)
        labels      = combined["label"].values.astype(float)
        hits.append(_hit_at_k(scores, labels, k))
        ndcgs.append(_ndcg_at_k(scores, labels, k))

    return {
        f"Popularity HR@{k}":   float(np.mean(hits)),
        f"Popularity NDCG@{k}": float(np.mean(ndcgs)),
    }
