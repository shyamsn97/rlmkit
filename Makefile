.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 oolong-paper oolong-rlm oolong-rlm-tips oolong-standard oolong-real oolong-ablations oolong-aggregate
	{%- if cookiecutter.use_black == 'y' %} lint/black{% endif %}
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"


clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr coverage/
	rm -fr .pytest_cache

lint: ## check style with flake8
	isort --profile black rlmkit
	black rlmkit
	flake8 rlmkit

install: clean lint
	python -m pip install . --upgrade

doc:
	rm -r docs/reference/
	pdocs as_markdown rlmkit -o docs/reference
	rm docs/reference/rlmkit/index.md
	cp examples/*.ipynb docs/examples/
	cp README.md docs/index.md

serve-docs:
	mkdocs serve

commit: install test doc
	git add .
	git commit -a

test:
	python -m pytest --cov=rlmkit/ --cov-report html:tests/cov-report tests/

test-html: test
	$(BROWSER) tests/cov-report/index.html

OOLONG_N ?= 20
OOLONG_SPLIT ?= validation
OOLONG_WORKERS ?= 1
OOLONG_MODEL ?= gpt-5
OOLONG_MAX_DEPTH ?= 1
OOLONG_MAX_ITERATIONS ?= 20

oolong-paper: ## Paper-style OOLONG RLM run: synth validation, depth 1, 20 iterations.
	python benchmarks/oolong/run.py --mode rlm --subset synth \
		--split validation --limit $(OOLONG_N) --shuffle --seed 42 \
		--max-depth $(OOLONG_MAX_DEPTH) \
		--max-iterations $(OOLONG_MAX_ITERATIONS) \
		--workers $(OOLONG_WORKERS) --model $(OOLONG_MODEL)

oolong-rlm: ## OOLONG synth rlm-mode quick run (override OOLONG_MODEL/OOLONG_N).
	python benchmarks/oolong/run.py --mode rlm --subset synth \
		--split $(OOLONG_SPLIT) --limit $(OOLONG_N) --shuffle \
		--workers $(OOLONG_WORKERS) --model $(OOLONG_MODEL)

oolong-rlm-tips: ## OOLONG synth rlm+<env_tips> quick run.
	python benchmarks/oolong/run.py --mode rlm_tips --subset synth \
		--split $(OOLONG_SPLIT) --limit $(OOLONG_N) --shuffle \
		--workers $(OOLONG_WORKERS) --model $(OOLONG_MODEL)

oolong-standard: ## OOLONG synth standard (\boxed{}) baseline quick run.
	python benchmarks/oolong/run.py --mode standard --subset synth \
		--split $(OOLONG_SPLIT) --limit $(OOLONG_N) --shuffle \
		--workers $(OOLONG_WORKERS) --model $(OOLONG_MODEL)

oolong-real: ## OOLONG real (DnD) rlm-mode quick run.
	python benchmarks/oolong/run.py --mode rlm --subset real \
		--split $(OOLONG_SPLIT) --limit $(OOLONG_N) --shuffle \
		--workers $(OOLONG_WORKERS) --model $(OOLONG_MODEL)

oolong-ablations: ## Full mode × subset sweep — see benchmarks/oolong/run_ablations.sh.
	bash benchmarks/oolong/run_ablations.sh

oolong-aggregate: ## Aggregate everything under benchmarks/oolong/outputs/.
	python benchmarks/oolong/aggregate.py