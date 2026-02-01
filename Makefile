VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
NPM ?= npm
TSX ?= npx tsx
COOLDOWN_S ?= 5

.PHONY: help install-node install-python start-single start-double bench-matrix clean-results

help:
	@printf "%s\n" \
		"Targets:" \
		"  install-node    Install Node dev dependencies (tsx, execa, commander)" \
		"  install-python  Install Python deps via uv (requires uv)" \
		"  start-single    Start vLLM single-process server" \
		"  start-double    Start vLLM double-process server" \
		"  bench-matrix    Run limiter matrix benchmark" \
		"  clean-results   Remove results directory"

install-node:
	$(NPM) install

install-python:
	uv venv $(VENV)
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync --frozen

bench-matrix:
	VLLM_PYTHON=$(PYTHON) $(TSX) scripts/run_matrix.ts --python $(PYTHON) --cooldown-s $(COOLDOWN_S) $(ARGS)

clean-results:
	rm -rf results
