PYTHON ?= python
NPM ?= npm
TSX ?= npx tsx

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
	uv venv .venv
	. .venv/bin/activate && uv sync --frozen

start-single:
	$(TSX) scripts/start_single.ts

start-double:
	$(TSX) scripts/start_double.ts

bench-matrix:
	$(TSX) scripts/run_matrix.ts

clean-results:
	rm -rf results
