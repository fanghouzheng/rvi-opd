.PHONY: lint test smoke validate

lint:
	python3 -m ruff check src tests

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

smoke:
	PYTHONPATH=src python3 -m rvi_opd smoke --output runs/smoke

validate:
	PYTHONPATH=src python3 -m rvi_opd validate-config --config-dir configs
