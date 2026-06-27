# ==========================================
# ENTERPRISE FORENSICS PLATFORM AUTOMATION MAKEFILE
# ==========================================

.PHONY: install format lint test docker-build docker-up clean

PYTHON = python
PIP = pip
DOCKER = docker
DOCKER_COMPOSE = docker-compose

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]

format:
	$(PYTHON) -m black app/ ai_engine/ tests/
	$(PYTHON) -m isort app/ ai_engine/ tests/

lint:
	$(PYTHON) -m flake8 app/ ai_engine/ tests/ --max-line-length=120 --ignore=E203,W503
	$(PYTHON) -m mypy app/

test:
	$(PYTHON) -m pytest -o addopts="" tests/

docker-build:
	$(DOCKER) build -t forensics-api:latest -f Dockerfile .

docker-up:
	$(DOCKER_COMPOSE) up -d

clean:
	rm -rf `find . -name __pycache__`
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf storage/uploads/*
	rm -rf storage/reports/*
	rm -rf logs/*
