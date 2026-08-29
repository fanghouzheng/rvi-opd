PYTHON ?= python3

.PHONY: lint test smoke validate

lint:
	$(PYTHON) -m ruff check src tests

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src $(PYTHON) -m rvi_opd smoke --output runs/smoke

validate:
	PYTHONPATH=src $(PYTHON) -m rvi_opd validate-config --config-dir configs
