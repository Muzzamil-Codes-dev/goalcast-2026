# GoalCast 2026 — Roadmap

I built this in phases so I could get each part running and checked before moving on. They're
all done now; this is roughly the order I worked in.

| Phase | Title | Status |
|-------|-------|--------|
| 0 | Scaffold (structure, config, tooling, Docker, CI) | ✅ |
| 1 | Data engineering (ingest -> validate -> Postgres/SQLite) | ✅ |
| 2 | Feature engineering (Elo, form, rankings, point-in-time) | ✅ |
| 3 | Models + MLflow (baselines, XGBoost, calibration, eval) | ✅ |
| 4 | FastAPI serving (predict, log every prediction) | ✅ |
| 5 | Streamlit dashboard (MVP: predictor + performance) | ✅ |
| 6 | Poisson goals model + Monte Carlo tournament sim | ✅ |
| 7 | Evidently monitoring (data/prediction drift, predicted-vs-actual) | ✅ |
| 8 | AI Match Analyst (RAG + Claude, offline fallback) | ✅ |
| 9 | Polish (tests, docs, architecture diagram, README) | ✅ |

## A few things I tried to stick to
- **Offline-first.** Every part still runs with no external services, so it's easy to clone
  and try without setting anything up.
- **No leakage.** Features are point-in-time and the train/test split is by date.
- **Baselines first.** I checked the models actually beat a simple Elo-only baseline before
  trusting them.
- **Judge the probabilities.** Log loss, Brier score and calibration, not just accuracy.
- **Keep it proportionate.** No Kubernetes or Kafka here. Docker and GitHub Actions were
  enough for a project this size.
