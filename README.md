# Flight Recommendation Engine

Neural Collaborative Filtering model trained on 14M domestic US flights (2024–2025) from the Bureau of Transportation Statistics. Recommends flights based on traveler profile and predicts delay risk — no user history required.

**Live demo:** http://3.86.83.209:8080

---

## Architecture

```
User Profile                    Flight Features
────────────                    ───────────────
freq_tier    →  Embedding(4,16) ┐
budget_tier  →  Embedding(3,16) ├→ concat → MLP (152→128→64→1) → preference_score
time_pref    →  Embedding(4,16) ┘                              → delay_score

route        →  Embedding(5793,64) ┐
carrier      →  Embedding(15,32)   ├→ concat ┘
season       →  Embedding(4,8)     ┘

final_score = 0.7 × preference_score + 0.3 × (1 − delay_prob)
```

**Key design decision:** Feature-based user representation (no user IDs) solves the cold-start problem — new users get recommendations immediately based on their travel profile.

---

## Results

### Recommendation (test set)

Evaluated using the **standard NCF protocol**: for each positive interaction, 99 negatives are sampled and the model must rank the positive in the top 10 out of 100. This is the correct methodology from the original NCF paper (He et al., 2017) and is significantly harder than ranking across all items.

| Metric | NCF Model | Popularity Baseline |
|---|---|---|
| HR@10 | **0.617** | 0.000 |
| NDCG@10 | **0.322** | 0.000 |

The popularity baseline scores 0.0 because popularity-weighted negatives are used during training — popular routes that a cohort didn't take appear as hard negatives, so a global popularity ranker fails.

### Delay Prediction (test set)

| Metric | Score |
|---|---|
| AUC-ROC | **0.818** |
| Precision | **0.694** |
| Recall | **0.694** |

### Training

- 6 epochs, early stopped at epoch 3 (best val NDCG@10 = 0.319)
- 410K parameters, ~16s/epoch on Apple M-series (MPS)
- Multi-task loss: 70% recommendation BCE + 30% delay BCE
- AdamW + cosine LR decay
- Popularity-weighted negative sampling (hard negatives)
- Month embedding captures seasonal route patterns

---

## Data

**Source:** [US Bureau of Transportation Statistics](https://transtats.bts.gov) On-Time Performance data — downloaded directly, no account required.

- **Period:** January 2024 – December 2025 (24 months)
- **Flights:** 14.08M total, 13.84M active (non-cancelled, non-diverted)
- **Carriers:** 15 (Southwest, Delta, American, United, SkyWest, ...)
- **Routes:** 7,305 unique origin-destination pairs
- **Delay rate (>30 min):** 13.9%

### Feature Engineering

| Feature | Source | Values |
|---|---|---|
| `freq_tier` | Carrier type × day of week | low / medium / high / frequent |
| `budget_tier` | Carrier category | economy / premium_economy / business |
| `time_pref` | Scheduled departure hour | morning / afternoon / evening / redeye |
| `route` | Origin-Dest pair | 5,793 unique routes |
| `carrier` | Reporting airline code | 15 carriers |
| `season` | Flight month | spring / summer / fall / winter |

### Cohort Construction

Flights are grouped into 32 synthetic user cohorts based on (freq_tier × budget_tier × time_pref). Each cohort represents a traveler archetype — e.g. "frequent business morning traveler" or "low-frequency economy evening traveler."

The interaction matrix (cohort × flight item) drives NCF training with 4:1 negative sampling.

---

## API

**Base URL:** `http://3.86.83.209:8080`

### `POST /recommend`

```bash
curl -X POST http://3.86.83.209:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "freq_tier": "frequent",
    "budget_tier": "business",
    "time_pref": "morning",
    "origin": "JFK",
    "destination": "LAX"
  }'
```

**Response:**
```json
{
  "origin": "JFK",
  "destination": "LAX",
  "top_flights": [
    {
      "rank": 1,
      "carrier": "DL",
      "time_bucket": "morning",
      "season": "summer",
      "route": "JFK-LAX",
      "preference_score": 0.620,
      "delay_risk": 0.029,
      "final_score": 0.725,
      "delay_badge": "low"
    }
  ]
}
```

**Parameters:**

| Field | Options |
|---|---|
| `freq_tier` | `low` / `medium` / `high` / `frequent` |
| `budget_tier` | `economy` / `premium_economy` / `business` |
| `time_pref` | `morning` / `afternoon` / `evening` / `redeye` |
| `origin` | 3-letter IATA airport code (e.g. `JFK`) |
| `destination` | 3-letter IATA airport code (e.g. `LAX`) |

### `GET /health`

```bash
curl http://3.86.83.209:8080/health
# {"status":"ok","model":"FlightNCF","device":"cpu"}
```

---

## Project Structure

```
flight_recommender/
├── src/
│   ├── download_data.py   # BTS data downloader (2024-2025)
│   ├── explore_data.py    # EDA and distributions
│   ├── preprocess.py      # Feature engineering, cohorts, interaction matrix
│   ├── model.py           # FlightNCF architecture
│   ├── dataset.py         # PyTorch Dataset
│   ├── train.py           # Training loop with early stopping
│   ├── evaluate.py        # HR@10, NDCG@10, AUC, popularity baseline
│   └── api.py             # FastAPI backend + HTML frontend
├── models/
│   ├── best_model.pt      # Trained weights (409K params)
│   └── results.json       # Evaluation metrics
├── data/
│   └── processed/         # Encoders, parquets, item lookup (baked into Docker)
├── Dockerfile
├── requirements.txt
└── deploy.sh
```

---

## Limitations

This project demonstrates the NCF architecture and ML pipeline end-to-end, but has known limitations worth being transparent about:

**Synthetic users, not real users.** The 32 cohorts are archetypes built from BTS aggregate data using heuristics (carrier type × day of week → traveler type). No one actually clicked, searched, or booked anything. A real recommender trains on actual user decisions.

**32 cohorts is too coarse.** Real traveler behavior has far more variance than 32 buckets can capture. Two "frequent business morning" travelers might have completely different carrier preferences based on hub location, loyalty status, or personal experience.

**No price signal.** Price is arguably the #1 booking factor and is completely absent. BTS data contains no fare information.

**Delay labels are route-level, not flight-level.** The delay head predicts whether a (cohort, route, carrier) combination historically had >15% of flights delayed — not whether a specific flight tomorrow will be delayed. A real delay predictor would use weather, connecting traffic, and real-time data.

**Not real-time.** Recommendations surface historical patterns, not live inventory. There is no pricing, seat availability, or live schedules.

---

## What the Recommendations Actually Are

The 22,780 "flight items" in the model are real historical (route × carrier × time_bucket) combinations extracted from BTS data. So **JFK-LAX on Delta in the morning is a real route that operated in 2024-2025** — not invented.

But the recommendations are **pattern-based, not real-time.** The model learned: *"travelers with a profile like yours historically flew these carrier/time combinations."* There is no pricing, no live schedule, no seat availability. If you ask for JFK-LAX, the model returns the carriers and time slots that cohort-similar travelers actually took — not what's departing tomorrow.

Think of it as: *"given who you are as a traveler, here's what people like you tend to book on this route."*

---

## Next Steps

### 1. Connect to a real flight search API
Right now the model ranks historical (route, carrier, time) patterns. To surface actual bookable flights, wire the output into a live flight data source:

- **[Amadeus Self-Service API](https://developers.amadeus.com)** — free tier, returns live fares and schedules. Use the NCF score to re-rank Amadeus results before showing them.
- **[Duffel API](https://duffel.com)** — modern REST API, returns real inventory from airlines directly.
- **[Skyscanner Affiliates](https://www.partners.skyscanner.net)** — redirect links to search results.

The integration pattern would be:
```
user profile + route
    → NCF scores all known (carrier, time) combos for that route
    → top-scored combos become search filters sent to Amadeus
    → Amadeus returns real flights matching those filters
    → display with price + NCF score combined
```

### 2. Replace synthetic cohorts with real user data
The current "users" are 32 synthetic archetypes built from BTS aggregate data. With real booking data you'd have actual user histories:

- Replace cohort embeddings with per-user embeddings trained on click/booking logs
- Use session data (searches, clicks, purchases) as implicit positive signals
- Add user account features (loyalty tier, home airport, past destinations)

### 3. Add price as a feature
BTS data has no fares. Price is one of the strongest booking signals:

- Add a `price_tier` embedding (budget / mid / premium) to the flight item features
- Source historical fare data from [Bureau of Transportation Statistics DB1B dataset](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD&DB_URL=) (quarterly origin-destination fare samples)
- At inference time, fetch live prices from Amadeus and bucket them into tiers

### 4. Retrain on more recent data
BTS publishes monthly data with a ~60 day lag. A cron job could:
```bash
# Monthly retrain pipeline
python3 src/download_data.py   # pulls new months automatically
python3 src/preprocess.py
python3 src/train.py
docker build + push to ECR
```

### 5. Add exact departure times
The model currently uses time-of-day buckets (morning/afternoon/evening/redeye). With a flight search API you'd get exact departure times and could show the user "Delta 8:15am" instead of "Delta morning."

---

## Running Locally

```bash
git clone <repo>
cd flight_recommender

python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

PYTHONPATH=src uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Then open `http://localhost:8080`.

### Retraining from scratch

```bash
# 1. Download BTS data (2024-2025, ~700MB compressed)
python3 src/download_data.py

# 2. Explore distributions
python3 src/explore_data.py

# 3. Feature engineering + train/val/test split
python3 src/preprocess.py

# 4. Train NCF model
python3 src/train.py
```

### Docker

```bash
docker build --platform linux/amd64 -t flight-recommender .
docker run -p 8080:8080 flight-recommender
```

---

## License

MIT
