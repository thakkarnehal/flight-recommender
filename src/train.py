"""Train FlightNCF with early stopping on val NDCG@10."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FlightDataset
from model import FlightNCF
from evaluate import evaluate_recommendation, evaluate_delay, popularity_baseline

PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
LR           = 2e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 512
MAX_EPOCHS   = 10
PATIENCE     = 3
REC_WEIGHT   = 0.7   # weight for recommendation loss in multi-task objective
DELAY_WEIGHT = 0.3


def train_epoch(model, loader, optimizer, device):
    model.train()
    rec_criterion   = nn.BCEWithLogitsLoss()
    delay_criterion = nn.BCEWithLogitsLoss()
    total_loss = rec_loss_sum = delay_loss_sum = 0.0

    for batch in loader:
        freq, budget, time_p, route, carrier, season, month, label, delay_label = (
            t.to(device) for t in batch
        )
        optimizer.zero_grad()
        rec_logit, delay_logit = model(freq, budget, time_p, route, carrier, season, month)

        rec_loss   = rec_criterion(rec_logit, label)
        delay_loss = delay_criterion(delay_logit, delay_label)
        loss = REC_WEIGHT * rec_loss + DELAY_WEIGHT * delay_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss     += loss.item()
        rec_loss_sum   += rec_loss.item()
        delay_loss_sum += delay_loss.item()

    n = len(loader)
    return total_loss / n, rec_loss_sum / n, delay_loss_sum / n


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Load metadata
    with open(PROC_DIR / "meta.json") as f:
        meta = json.load(f)

    # Datasets & loaders
    train_ds = FlightDataset(str(PROC_DIR / "train.parquet"))
    val_ds   = FlightDataset(str(PROC_DIR / "val.parquet"))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Model
    model = FlightNCF(
        n_routes   = meta["n_routes"],
        n_carriers = meta["n_carriers"],
        n_freq     = meta["n_freq"],
        n_budget   = meta["n_budget"],
        n_time     = meta["n_time"],
        n_season   = meta["n_season"],
        n_month    = meta["n_month"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── Popularity baseline (computed once) ────────────────────────────────────
    print("\nComputing popularity baseline...")
    pop_metrics = popularity_baseline(str(PROC_DIR / "val.parquet"), k=10)
    for k, v in pop_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_ndcg    = -1.0
    patience_ctr = 0
    history      = []

    print("\nEpoch  Loss    RecLoss  DelayLoss  HR@10   NDCG@10  Time")
    print("-" * 65)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        loss, rec_l, delay_l = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_rec   = evaluate_recommendation(model, str(PROC_DIR / "val.parquet"), device, k=10)
        val_delay = evaluate_delay(model, str(PROC_DIR / "val.parquet"), device)

        hr   = val_rec["HR@10"]
        ndcg = val_rec["NDCG@10"]
        elapsed = time.time() - t0

        print(f"  {epoch:2d}   {loss:.4f}  {rec_l:.4f}   {delay_l:.4f}    "
              f"{hr:.4f}  {ndcg:.4f}   {elapsed:.0f}s")

        row = {"epoch": epoch, "loss": loss, "rec_loss": rec_l,
               "delay_loss": delay_l, "HR@10": hr, "NDCG@10": ndcg,
               "delay_AUC": val_delay.get("AUC", float("nan"))}
        history.append(row)

        if ndcg > best_ndcg:
            best_ndcg    = ndcg
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_ndcg": best_ndcg,
                "meta": meta,
            }, MODELS_DIR / "best_model.pt")
            print(f"         ✓ saved checkpoint (NDCG@10={best_ndcg:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_rec   = evaluate_recommendation(model, str(PROC_DIR / "test.parquet"), device, k=10)
    test_delay = evaluate_delay(model, str(PROC_DIR / "test.parquet"), device)
    pop_test   = popularity_baseline(str(PROC_DIR / "test.parquet"), k=10)

    print("\n── Test Results ──────────────────────────────────────────────")
    print("Recommendation:")
    for k, v in test_rec.items():
        pop_k = f"Popularity {k}"
        pop_v = pop_test.get(pop_k, float("nan"))
        print(f"  {k}: {v:.4f}  (popularity baseline: {pop_v:.4f})")
    print("Delay prediction:")
    for k, v in test_delay.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nBest val NDCG@10: {best_ndcg:.4f}  (epoch {ckpt['epoch']})")

    results = {
        "best_val_ndcg": best_ndcg,
        "best_epoch": ckpt["epoch"],
        "test_recommendation": test_rec,
        "test_delay": test_delay,
        "popularity_baseline_test": pop_test,
        "history": history,
    }
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to models/results.json")


if __name__ == "__main__":
    main()
