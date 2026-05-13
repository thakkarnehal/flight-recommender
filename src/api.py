"""FastAPI backend for FlightNCF recommendation engine."""

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from model import FlightNCF

BASE_DIR   = Path(__file__).parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ── Load artifacts at startup ──────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

with open(PROC_DIR / "meta.json")         as f: META         = json.load(f)
with open(PROC_DIR / "item_lookup.pkl",  "rb") as f: ITEM_LOOKUP  = pickle.load(f)
with open(PROC_DIR / "route_encoder.pkl","rb") as f: ROUTE_LE     = pickle.load(f)
with open(PROC_DIR / "carrier_encoder.pkl","rb") as f: CARRIER_LE  = pickle.load(f)
with open(PROC_DIR / "item_encoder.pkl", "rb") as f: ITEM_LE      = pickle.load(f)

ckpt  = torch.load(MODELS_DIR / "best_model.pt", map_location=device, weights_only=False)
MODEL = FlightNCF(
    n_routes   = META["n_routes"],
    n_carriers = META["n_carriers"],
    n_freq     = META["n_freq"],
    n_budget   = META["n_budget"],
    n_time     = META["n_time"],
    n_season   = META["n_season"],
    n_month    = META["n_month"],
).to(device)
MODEL.load_state_dict(ckpt["model_state"])
MODEL.eval()

FREQ_MAP   = META["freq_map"]
BUDGET_MAP = META["budget_map"]
TIME_MAP   = META["time_map"]
SEASON_MAP = META["season_map"]

# ── Valid route/carrier sets ───────────────────────────────────────────────────
KNOWN_ROUTES   = set(ROUTE_LE.classes_)
KNOWN_CARRIERS = set(CARRIER_LE.classes_)

app = FastAPI(title="Flight Recommender", version="1.0")

# ── Schema ─────────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    freq_tier:   Literal["low", "medium", "high", "frequent"]
    budget_tier: Literal["economy", "premium_economy", "business"]
    time_pref:   Literal["morning", "afternoon", "evening", "redeye"]
    origin:      str
    destination: str


class FlightResult(BaseModel):
    rank:             int
    carrier:          str
    time_bucket:      str
    season:           str
    route:            str
    preference_score: float
    delay_risk:       float
    final_score:      float
    delay_badge:      Literal["low", "medium", "high"]


class RecommendResponse(BaseModel):
    top_flights: list[FlightResult]
    origin:      str
    destination: str


# ── Core inference ─────────────────────────────────────────────────────────────
def get_delay_badge(delay_prob: float) -> str:
    if delay_prob < 0.20:
        return "low"
    if delay_prob < 0.40:
        return "medium"
    return "high"


@torch.no_grad()
def recommend(req: RecommendRequest, top_k: int = 10) -> list[FlightResult]:
    route_str = f"{req.origin.upper()}-{req.destination.upper()}"

    # Find all items matching this route
    matching_items = [
        (idx, info)
        for idx, info in ITEM_LOOKUP.items()
        if info["route"] == route_str
    ]

    if not matching_items:
        raise HTTPException(
            status_code=404,
            detail=f"No flights found for route {route_str}. "
                   f"Check that both airport codes are valid US airports."
        )

    freq_idx   = FREQ_MAP[req.freq_tier]
    budget_idx = BUDGET_MAP[req.budget_tier]
    time_idx   = TIME_MAP[req.time_pref]

    n = len(matching_items)
    freq_t    = torch.full((n,), freq_idx,   dtype=torch.long, device=device)
    budget_t  = torch.full((n,), budget_idx, dtype=torch.long, device=device)
    time_t    = torch.full((n,), time_idx,   dtype=torch.long, device=device)
    route_t   = torch.tensor([info["route_idx"]   for _, info in matching_items], dtype=torch.long, device=device)
    carrier_t = torch.tensor([info["carrier_idx"] for _, info in matching_items], dtype=torch.long, device=device)
    season_t  = torch.tensor([info["season_idx"]  for _, info in matching_items], dtype=torch.long, device=device)
    month_t   = torch.tensor([info["month_idx"]   for _, info in matching_items], dtype=torch.long, device=device)

    final_scores, pref_scores, delay_probs = MODEL.score(
        freq_t, budget_t, time_t, route_t, carrier_t, season_t, month_t
    )
    final_scores = final_scores.cpu().numpy()
    pref_scores  = pref_scores.cpu().numpy()
    delay_probs  = delay_probs.cpu().numpy()

    order = np.argsort(final_scores)[::-1][:top_k]
    results = []
    for rank, i in enumerate(order, start=1):
        _, info = matching_items[i]
        results.append(FlightResult(
            rank             = rank,
            carrier          = info["Reporting_Airline"],
            time_bucket      = info["time_bucket"],
            season           = {v: k for k, v in SEASON_MAP.items()}[int(info["season_idx"])],
            route            = route_str,
            preference_score = round(float(pref_scores[i]),  4),
            delay_risk       = round(float(delay_probs[i]),  4),
            final_score      = round(float(final_scores[i]), 4),
            delay_badge      = get_delay_badge(float(delay_probs[i])),
        ))
    return results


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "FlightNCF", "device": str(device)}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest):
    results = recommend(req)
    return RecommendResponse(
        top_flights = results,
        origin      = req.origin.upper(),
        destination = req.destination.upper(),
    )


@app.get("/", response_class=HTMLResponse)
def frontend():
    return HTMLResponse(content=HTML_FRONTEND)


# ── HTML Frontend ──────────────────────────────────────────────────────────────
HTML_FRONTEND = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flight Recommender</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  .container { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
  h1 { font-size: 1.8rem; font-weight: 700; color: #60a5fa;
       margin-bottom: 0.4rem; }
  .subtitle { color: #94a3b8; margin-bottom: 2rem; font-size: 0.9rem; }
  .card { background: #1e2130; border: 1px solid #2d3748;
          border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .form-group { display: flex; flex-direction: column; gap: 0.4rem; }
  label { font-size: 0.8rem; font-weight: 600; color: #94a3b8;
          text-transform: uppercase; letter-spacing: 0.05em; }
  select, input { background: #0f1117; border: 1px solid #374151;
                  color: #e2e8f0; padding: 0.6rem 0.8rem;
                  border-radius: 8px; font-size: 0.95rem; width: 100%; }
  select:focus, input:focus { outline: none; border-color: #60a5fa; }
  .route-row { display: grid; grid-template-columns: 1fr auto 1fr; gap: 0.5rem;
               align-items: center; margin-bottom: 1rem; }
  .arrow { color: #60a5fa; font-size: 1.4rem; text-align: center; }
  button { width: 100%; background: #2563eb; color: white; border: none;
           padding: 0.75rem; border-radius: 8px; font-size: 1rem;
           font-weight: 600; cursor: pointer; margin-top: 1rem;
           transition: background 0.2s; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #374151; cursor: not-allowed; }
  .results-header { display: flex; justify-content: space-between;
                    align-items: center; margin-bottom: 1rem; }
  .results-title { font-size: 1.1rem; font-weight: 600; }
  .results-sub   { font-size: 0.8rem; color: #94a3b8; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { text-align: left; padding: 0.5rem 0.75rem; color: #94a3b8;
       font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
       border-bottom: 1px solid #2d3748; }
  td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #1a1f2e; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #252a3a; }
  .rank { color: #60a5fa; font-weight: 700; width: 2rem; }
  .score-bar { display: flex; align-items: center; gap: 0.5rem; }
  .bar { height: 6px; border-radius: 3px; background: #374151; width: 80px; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
           font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
  .badge-low    { background: #064e3b; color: #6ee7b7; }
  .badge-medium { background: #78350f; color: #fcd34d; }
  .badge-high   { background: #7f1d1d; color: #fca5a5; }
  .final-score  { font-weight: 700; color: #60a5fa; }
  .error { background: #450a0a; border: 1px solid #991b1b; color: #fca5a5;
           padding: 1rem; border-radius: 8px; margin-top: 1rem; }
  .loading { text-align: center; color: #94a3b8; padding: 2rem; }
  #results { display: none; }
</style>
</head>
<body>
<div class="container">
  <h1>✈ Flight Recommender</h1>
  <p class="subtitle">Neural Collaborative Filtering · Trained on 14M BTS flights (2024–2025)</p>

  <div class="card">
    <div class="route-row">
      <div class="form-group">
        <label>Origin Airport</label>
        <input id="origin" placeholder="e.g. JFK" maxlength="3" style="text-transform:uppercase">
      </div>
      <div class="arrow">→</div>
      <div class="form-group">
        <label>Destination Airport</label>
        <input id="dest" placeholder="e.g. LAX" maxlength="3" style="text-transform:uppercase">
      </div>
    </div>

    <div class="form-grid">
      <div class="form-group">
        <label>Travel Frequency</label>
        <select id="freq_tier">
          <option value="low">Low (1–3 flights/year)</option>
          <option value="medium">Medium (4–10 flights/year)</option>
          <option value="high" selected>High (11–20 flights/year)</option>
          <option value="frequent">Frequent (20+ flights/year)</option>
        </select>
      </div>
      <div class="form-group">
        <label>Budget Preference</label>
        <select id="budget_tier">
          <option value="economy">Economy</option>
          <option value="premium_economy" selected>Premium Economy</option>
          <option value="business">Business</option>
        </select>
      </div>
      <div class="form-group">
        <label>Preferred Departure Time</label>
        <select id="time_pref">
          <option value="morning" selected>Morning (5am–12pm)</option>
          <option value="afternoon">Afternoon (12pm–6pm)</option>
          <option value="evening">Evening (6pm–10pm)</option>
          <option value="redeye">Redeye (10pm–5am)</option>
        </select>
      </div>
    </div>

    <button id="search-btn" onclick="search()">Find Best Flights</button>
  </div>

  <div id="error-box" class="error" style="display:none"></div>

  <div id="results" class="card">
    <div class="results-header">
      <div>
        <div class="results-title" id="results-title"></div>
        <div class="results-sub">Ranked by combined score = 0.7 × preference + 0.3 × reliability</div>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Carrier</th>
          <th>Time</th>
          <th>Season</th>
          <th>Preference</th>
          <th>Delay Risk</th>
          <th>Final Score</th>
        </tr>
      </thead>
      <tbody id="results-body"></tbody>
    </table>
  </div>
</div>

<script>
const CARRIER_NAMES = {
  AA:"American", DL:"Delta", UA:"United", WN:"Southwest", B6:"JetBlue",
  AS:"Alaska", NK:"Spirit", F9:"Frontier", G4:"Allegiant", MQ:"Envoy Air",
  OO:"SkyWest", YX:"Republic", HA:"Hawaiian", OH:"PSA Airlines", "9E":"Endeavor"
};

function barColor(v) {
  if (v > 0.7) return "#22c55e";
  if (v > 0.4) return "#eab308";
  return "#ef4444";
}

async function search() {
  const btn = document.getElementById("search-btn");
  const errBox = document.getElementById("error-box");
  errBox.style.display = "none";
  document.getElementById("results").style.display = "none";

  const origin = document.getElementById("origin").value.trim().toUpperCase();
  const dest   = document.getElementById("dest").value.trim().toUpperCase();
  if (!origin || !dest || origin.length !== 3 || dest.length !== 3) {
    errBox.textContent = "Please enter valid 3-letter IATA airport codes (e.g. JFK, LAX).";
    errBox.style.display = "block"; return;
  }

  btn.disabled = true;
  btn.textContent = "Searching…";

  try {
    const resp = await fetch("/recommend", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        freq_tier:   document.getElementById("freq_tier").value,
        budget_tier: document.getElementById("budget_tier").value,
        time_pref:   document.getElementById("time_pref").value,
        origin, destination: dest,
      })
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Unknown error");
    }
    const data = await resp.json();
    renderResults(data);
  } catch(e) {
    errBox.textContent = "Error: " + e.message;
    errBox.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Find Best Flights";
  }
}

function renderResults(data) {
  document.getElementById("results-title").textContent =
    `Top ${data.top_flights.length} flights · ${data.origin} → ${data.destination}`;

  const tbody = document.getElementById("results-body");
  tbody.innerHTML = data.top_flights.map(f => `
    <tr>
      <td class="rank">${f.rank}</td>
      <td>${CARRIER_NAMES[f.carrier] || f.carrier} <span style="color:#64748b">(${f.carrier})</span></td>
      <td style="text-transform:capitalize">${f.time_bucket.replace("_"," ")}</td>
      <td style="text-transform:capitalize">${f.season}</td>
      <td>
        <div class="score-bar">
          <div class="bar"><div class="bar-fill" style="width:${f.preference_score*100}%;background:${barColor(f.preference_score)}"></div></div>
          <span>${(f.preference_score*100).toFixed(1)}%</span>
        </div>
      </td>
      <td><span class="badge badge-${f.delay_badge}">${f.delay_badge} (${(f.delay_risk*100).toFixed(0)}%)</span></td>
      <td class="final-score">${(f.final_score*100).toFixed(1)}%</td>
    </tr>
  `).join("");

  document.getElementById("results").style.display = "block";
}

document.getElementById("origin").addEventListener("input", e => {
  e.target.value = e.target.value.toUpperCase();
});
document.getElementById("dest").addEventListener("input", e => {
  e.target.value = e.target.value.toUpperCase();
});
document.addEventListener("keydown", e => {
  if (e.key === "Enter") search();
});
</script>
</body>
</html>
"""
