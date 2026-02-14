.PHONY: help venv install run run-ever list-mics check test clean-pyc

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PIP := $(if $(wildcard .venv/bin/pip),.venv/bin/pip,pip3)

help:
	@echo "Available targets:"
	@echo "  make venv       - Create virtual environment at .venv"
	@echo "  make install    - Install dependencies"
	@echo "  make run        - Run CLI in enter mode"
	@echo "  make run-ever   - Run CLI in ever-running mode"
	@echo "  make list-mics  - List microphone devices"
	@echo "  make check      - Syntax-check Python files"
	@echo "  make test       - Run integration tests"
	@echo "  make clean-pyc  - Remove __pycache__ directories"

venv:
	python3 -m venv .venv

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) main.py

run-ever:
	$(PYTHON) main.py --mode ever-running --trigger-threshold 0.06

list-mics:
	$(PYTHON) main.py --list-mics

check:
	$(PYTHON) -m py_compile src/whisper_cli/cli.py main.py

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

clean-pyc:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
