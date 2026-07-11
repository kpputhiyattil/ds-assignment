# Convenience targets mirroring the pipeline run order.
# Later steps fill in the currently-stubbed pipeline stages.

.PHONY: install test lint events interval features labels train ablation explain all clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

# --- pipeline (stages added incrementally, Steps 2-8) ---
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

all: events interval features labels train ablation explain

clean:
	rm -rf data/processed/* data/artifacts/*
	touch data/processed/.gitkeep data/artifacts/.gitkeep

package:
	python -m src.serving.package

serve:
	uvicorn src.serving.api:app --reload
