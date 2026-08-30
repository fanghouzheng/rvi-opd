PYTHON ?= python3

.PHONY: env-check lint test smoke validate

env-check:
	$(PYTHON) scripts/check_environment.py --profile dev

lint:
	$(PYTHON) -m ruff check src tests scripts

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) -m rvi_opd smoke --output runs/smoke

validate:
	PYTHONPATH=src $(PYTHON) -m rvi_opd validate-config --config-dir configs
	PYTHONPATH=src $(PYTHON) -m rvi_opd validate-execution-policy
