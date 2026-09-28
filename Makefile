PY := .venv/bin/python

.PHONY: install index app api eval test lint format lock

install:  ## create .venv and install everything
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[app,api,dev]"
	.venv/bin/pre-commit install

index:  ## (re)build the search index from data/raw
	$(PY) scripts/build_index.py

app:  ## Streamlit chat UI on http://localhost:8501
	.venv/bin/streamlit run app/streamlit_app.py

api:  ## REST API on http://localhost:8000 (docs at /docs)
	.venv/bin/uvicorn challansaathi.api:app --port 8000

eval:  ## retrieval evaluation -> eval/results.md
	$(PY) eval/run_eval.py

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

lock:  ## pin exact versions of all dependencies
	uv pip compile pyproject.toml --extra app --extra api --extra dev -o requirements.lock
