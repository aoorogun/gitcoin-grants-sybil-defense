.PHONY: all test lint
all:
	gitcoin-integrity generate
	gitcoin-integrity train
	gitcoin-integrity allocate
	gitcoin-integrity scenario --file configs/scenario_stricter_policy.json
	gitcoin-integrity monitor
test:
	python -m unittest discover -s tests -v
lint:
	ruff check src tests

