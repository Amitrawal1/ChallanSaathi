PY := .venv/bin/python

.PHONY: install index app test lint format

install:
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[app,dev]"

index:
	.venv/bin/challansaathi build-index

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests app
	.venv/bin/ruff format --check src tests app

format:
	.venv/bin/ruff check --fix src tests app
	.venv/bin/ruff format src tests app
