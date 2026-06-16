# Architecture

```mermaid
flowchart TB
    subgraph Ingestion["1 · Data Engineering"]
        A[Public dataset / synthetic fallback] -->|ingest| B[Raw parquet]
        B -->|Pandera validation + quality checks| C[Processed]
        C --> D[(SQLite / PostgreSQL)]
    end

    subgraph Features["2 · Feature Engineering"]
        D --> E[Elo · Form · Goals<br/>Rest days · Host advantage]
        E --> F[features.parquet]
    end

    subgraph Training["3 · ML + MLOps"]
        F --> G[Baselines -> XGBoost + Dixon-Coles Poisson]
        G -->|params/metrics/calibration| H[(MLflow)]
        G --> I[model_bundle.joblib]
        H --> J[Registry: staging -> production]
    end

    subgraph Serving["4 · Serving"]
        I --> K[FastAPI<br/>/predict /simulate /explain]
        K -->|log every prediction| D
        K --> L[AI Match Analyst<br/>RAG + Claude]
    end

    subgraph Product["5 · Product + Observability"]
        I --> M[Streamlit Dashboard]
        D --> N[Evidently / PSI drift<br/>predicted vs actual]
        N --> M
    end

    O[GitHub Actions: lint · test · scheduled retrain] -.-> Training
```

**Data flow:** `raw → validated → features → model → bundle/registry → API → dashboard`,
with every prediction logged back to the database so monitoring can compare predicted
probabilities against real results.
