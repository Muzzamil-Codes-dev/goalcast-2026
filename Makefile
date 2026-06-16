# GoalCast 2026 — task runner
.PHONY: help install lint test ingest features train evaluate promote serve dashboard pipeline demo-plots up down clean

help:
	@echo "install   - pip install requirements"
	@echo "lint      - ruff + mypy"
	@echo "test      - pytest"
	@echo "ingest    - download/generate raw data -> DB"
	@echo "features  - build point-in-time feature table"
	@echo "train     - train models, log to MLflow"
	@echo "evaluate  - evaluate vs baselines"
	@echo "promote   - promote best model staging->production"
	@echo "serve     - run FastAPI on :8000"
	@echo "dashboard - run Streamlit on :8501"
	@echo "pipeline  - ingest -> features -> train -> evaluate"
	@echo "demo-plots- regenerate README demo images (winner + scoreline)"
	@echo "up/down   - docker compose up/down"

install:
	pip install -r requirements.txt && pip install -e .

lint:
	ruff check src tests && mypy src

test:
	pytest

ingest:
	python -m goalcast.data.ingest

features:
	python -m goalcast.features.build

train:
	python -m goalcast.models.train

evaluate:
	python -m goalcast.models.evaluate

promote:
	python -m goalcast.models.registry --promote

serve:
	uvicorn goalcast.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

pipeline: ingest features train evaluate

demo-plots:
	python scripts/make_demo_plots.py

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf data/processed/* data/features/* models/artifacts/* mlruns
