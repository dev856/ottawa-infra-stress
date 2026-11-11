# Ottawa 311 Insights

What residents in each Ottawa ward report to 311, how long the City takes to close those requests, and what's unusual this week. Built entirely on **real, openly licensed City of Ottawa data**: about 640,000 service requests since January 2025, and the 24 ward boundaries.

> **Not an official City service.** "Closed" means the City marked a request closed, which is not always when the work was done. Not affiliated with or endorsed by the City of Ottawa.

---

## What it does

- **Ward report card:** pick a ward to see its most common requests over the last 12 months and how many days they typically take to close there compared with the whole city. These are counts and dates from City records, not predictions.
- **"How long will my request take?":** pick a request type and your ward to get *"Usually closed in 3 days; 9 in 10 within 18 days."*
- **"I already reported it, how much longer?":** enter the days you have waited to see how much longer past requests took once they had been open that long, for example *"Of 640 still open after 10 days, half were closed within another 108 days."* These are counts from history, not model output.
- **Explorable map:** colour the wards by number of requests, typical days to close, or unusual jumps, for all requests or for the one problem you picked.
- **Unusual this week:** request types that jumped in a ward, for example *"Street light out, Rideau-Vanier: 46 this week, usually about 4."* A cluster of leak or light-out reports can be the first sign of one bigger problem.
- **Honest caveats built in:** between 14 and 22 May 2026 the City closed about 130,000 old requests in one bulk clean-up, a quarter of everything closed in this data. For the roughly half of request types caught up in it (potholes, street lights, tree work), older closing dates say when records were tidied, not when work was done. Those types are marked ⚠️, and every closing time shown for them uses only requests reported after the clean-up.

## The model, and how well it does

Two LightGBM **quantile regression** models predict days to close: the median ("usually") and the 90th percentile ("9 in 10 within"). Features: request type, ward, submission channel, and weekday.

The model is trained only on request types whose closing dates can be trusted. Types caught up in the May 2026 bulk clean-up are left out, and the app answers for them from their requests since the clean-up instead (when at least 30 have had 120 days to close; otherwise it says there is not enough history yet).

Evaluated on 78,585 requests opened March–July 2026 that the model never saw:

| | Model | Per-type lookup baseline |
| :--- | ---: | ---: |
| Median: pinball loss (lower is better) | **10.1** | 11.0 |
| 90th percentile: requests actually closed within it (target 90%) | **90.4%** | 90.4% |
| 90th percentile: pinball loss | 6.8 | **6.6** |

- The model beats looking up each type's history by **8%** on the typical time, mainly because the same request type closes faster in some wards than others. Its typical error is about one day.
- The "9 in 10" estimate is **well calibrated (90.4%)**. On pinball loss it ties the simple lookup.
- **Avoiding a biased test set:** requests opened in the last 120 days are excluded, because many slow ones haven't closed yet and would make the times look shorter. Test requests still open after 120 days (2,253) count as "not closed within the estimate" rather than being dropped.
- **What didn't help:** adding the month made the median worse, so it's left out. That was measured before the bulk-closed types were removed and has not been re-tested since.
- **Known gap:** for ⚠️ types the report card covers every request since the clean-up, while the estimate uses only those with 120 days to close, so the two can differ (8 versus 20 days for potholes in one ward). They converge as more history builds up.

## Spike detection

For every ward and request type, this week's count is compared with the previous 8 weeks. It is flagged only if it has at least 5 requests, at least 3× the usual rate, and a Poisson probability of at most 1 in 10,000. The strict bar is because about 20,000 ward/type pairs are tested each week.

## How it works

```text
City of Ottawa open data ──> open_data/ snapshot ──┬──> report cards + spike alerts (pandas)
  311 service requests        (committed, ~4 MB)   └──> LightGBM quantile models
  ward boundaries                                            │
                                                             ▼
                                          FastAPI ──> MapLibre dashboard
```

1. **`src/ottawa_data.py`** downloads the 311 files and ward boundaries. It keeps only dates, category, type, ward, channel, and status (no addresses, coordinates, or request IDs). The snapshot is committed, so everything else runs offline.
2. **`src/insights.py`** builds the report cards, batch-closure flags, and spike alerts.
3. **`src/train_model.py`** trains and evaluates the closing-time models.
4. **`api/`** serves it all; **`frontend/`** draws the map with self-hosted MapLibre GL JS, with no third-party tiles or tracking.

**Stack:** Python 3.11, pandas, scikit-learn, LightGBM, FastAPI, MapLibre GL JS, pytest, Ruff, MyPy, Docker.

---

## Quickstart

```bash
git clone https://github.com/dev856/ottawa-infra-stress.git
cd ottawa-infra-stress
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

python run_pipeline.py             # train from the committed snapshot (offline)
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. To download fresh data from the City first, run `python run_pipeline.py --refresh`.

**Staying current:** `.github/workflows/refresh-data.yml` downloads fresh City data every Monday and commits the snapshot, so `git pull` followed by `python run_pipeline.py` brings a local copy up to date. The evaluation numbers above are from the snapshot they were measured on.

**Docker:** `docker compose up --build`, then open http://127.0.0.1:8000.

## API

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/wards` | Ward GeoJSON with 12-month request totals and spike counts |
| `GET` | `/wards/{ward}` | Report card: top requests and typical closing days, ward vs city |
| `GET` | `/request-types` | Request types with at least 100 requests in 12 months |
| `GET` | `/estimate?type=…&ward=…` | Usual and 9-in-10 days to close |
| `GET` | `/remaining?type=…&ward=…&waited=…` | How much longer requests took once already open `waited` days |
| `GET` | `/ward-metrics?type=…` | Per-ward requests, median closing days, and spikes for colouring the map (`type` optional) |
| `GET` | `/spikes` | This week's unusual ward/type jumps |
| `GET` | `/metrics` | Model evaluation from training |
| `GET` | `/health`, `/ready` | Liveness and readiness |

Interactive docs: http://127.0.0.1:8000/docs

## Testing

```bash
python -m pytest -q --cov=src --cov=api
python -m ruff check src tests api run_pipeline.py
python -m mypy src api run_pipeline.py
```

---

## Data and licence

- **City of Ottawa:** 311 service requests (previous and current year) and Wards 2022–2026. *Contains information licensed under the [Open Government Licence – City of Ottawa](https://open.ottawa.ca/pages/open-data-licence).* This project is not affiliated with or endorsed by the City of Ottawa.
- **Code:** [MIT License](LICENSE).
