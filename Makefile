.PHONY: install test smoke full paysim dashboard
install:
	python3 -m venv .venv && .venv/bin/pip install -q -e ".[test]"
test:
	.venv/bin/python -m pytest
smoke:
	.venv/bin/python -m vista.run_all --config configs/experiments/smoke.json && .venv/bin/python -m vista.dashboard smoke
full:
	.venv/bin/python -m vista.run_all --config configs/experiments/full.json && .venv/bin/python -m vista.dashboard full
dashboard:
	.venv/bin/python -m vista.dashboard full
paysim:
	.venv/bin/python -m vista.run_all --config configs/experiments/paysim.json
