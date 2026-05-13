"""PyTorch Dataset for FlightNCF interactions."""

import pandas as pd
import torch
from torch.utils.data import Dataset


class FlightDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self.freq_idx    = torch.tensor(df["freq_idx"].values,    dtype=torch.long)
        self.budget_idx  = torch.tensor(df["budget_idx"].values,  dtype=torch.long)
        self.time_idx    = torch.tensor(df["time_idx"].values,    dtype=torch.long)
        self.route_idx   = torch.tensor(df["route_idx"].values,   dtype=torch.long)
        self.carrier_idx = torch.tensor(df["carrier_idx"].values, dtype=torch.long)
        self.season_idx  = torch.tensor(df["season_idx"].values,  dtype=torch.long)
        self.month_idx   = torch.tensor(df["month_idx"].values,   dtype=torch.long)
        self.label       = torch.tensor(df["label"].values,       dtype=torch.float32)
        self.delay_label = torch.tensor(df["delay_label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (
            self.freq_idx[idx],
            self.budget_idx[idx],
            self.time_idx[idx],
            self.route_idx[idx],
            self.carrier_idx[idx],
            self.season_idx[idx],
            self.month_idx[idx],
            self.label[idx],
            self.delay_label[idx],
        )
