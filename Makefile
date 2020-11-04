CONFIG = "config.json"

.PHONY: type_check
type_check:
	mypy .

.PHONY: test
test:
	pytest .

.PHONY: start
start:
	python entrypoint.py --config $(CONFIG)
