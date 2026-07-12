# Convenience targets mirroring the pipeline run order.

.PHONY: install install-serving test lint eda events interval features labels train ablation explain assess batch-assess package serve ui all clean

install:
	pip install -e ".[dev]"

install-serving:
	pip install -e ".[dev,serving]"

test:
	pytest

lint:
	ruff check src tests

# --- pipeline ---
eda:
	python -m src.scripts.run_eda

events:
	python -m src.data.ingestion

interval:
	python -m src.features.interval

features:
	python -m src.features.build

labels:
	python -m src.features.labels

train:
	python -m src.models.train

ablation:
	python -m src.models.ablation

explain:
	python -m src.explainability.shap_utils

assess:
	python -m src.scripts.run_assessment

batch-assess:
	python -m src.scripts.run_batch_assess

batch-assess-recurring:
	python -m src.scripts.run_batch_assess --recurring-only

batch-assess-full:
	python -m src.scripts.run_batch_assess --full

package:
	python -m src.serving.package

# Production-style API (no reload). Streamlit UI talks to this.
serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

serve-dev:
	uvicorn src.serving.api:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py

all: events interval features labels train ablation explain eda assess package batch-assess

clean:
	rm -rf data/processed/* data/artifacts/*
	touch data/processed/.gitkeep data/artifacts/.gitkeep
