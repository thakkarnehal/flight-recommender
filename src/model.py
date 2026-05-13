"""FlightNCF: Neural Collaborative Filtering for flight recommendation."""

import torch
import torch.nn as nn


class FlightNCF(nn.Module):
    def __init__(
        self,
        n_routes: int,
        n_carriers: int,
        n_freq: int = 4,
        n_budget: int = 3,
        n_time: int = 4,
        n_season: int = 4,
        n_month: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()

        # User feature embeddings
        self.freq_emb    = nn.Embedding(n_freq,   16)
        self.budget_emb  = nn.Embedding(n_budget, 16)
        self.time_emb    = nn.Embedding(n_time,   16)

        # Flight feature embeddings
        self.route_emb   = nn.Embedding(n_routes,   64)
        self.carrier_emb = nn.Embedding(n_carriers, 32)
        self.season_emb  = nn.Embedding(n_season,    8)
        self.month_emb   = nn.Embedding(n_month,     8)  # peak month per item

        # Input dim: 16+16+16+64+32+8+8 = 160
        in_dim = 160

        # Recommendation MLP
        self.rec_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Delay prediction head
        self.delay_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for emb in [self.freq_emb, self.budget_emb, self.time_emb,
                    self.route_emb, self.carrier_emb, self.season_emb]:
            nn.init.xavier_uniform_(emb.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, freq, budget, time_pref, route, carrier, season, month):
        u = torch.cat([
            self.freq_emb(freq),
            self.budget_emb(budget),
            self.time_emb(time_pref),
        ], dim=-1)  # (B, 48)

        f = torch.cat([
            self.route_emb(route),
            self.carrier_emb(carrier),
            self.season_emb(season),
            self.month_emb(month),
        ], dim=-1)  # (B, 112)

        x = torch.cat([u, f], dim=-1)  # (B, 152)

        rec_logit   = self.rec_net(x).squeeze(-1)
        delay_logit = self.delay_net(x).squeeze(-1)
        return rec_logit, delay_logit

    def score(self, freq, budget, time_pref, route, carrier, season, month):
        """Combined inference score: 0.7×preference + 0.3×reliability."""
        rec_logit, delay_logit = self.forward(freq, budget, time_pref, route, carrier, season, month)
        pref_score    = torch.sigmoid(rec_logit)
        delay_prob    = torch.sigmoid(delay_logit)
        reliability   = 1.0 - delay_prob
        return 0.7 * pref_score + 0.3 * reliability, pref_score, delay_prob
