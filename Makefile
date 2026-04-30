.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 oolong-paper oolong-rlm oolong-rlm-tips oolong-standard oolong-real oolong-ablations oolong-aggregate animation animation-preview animation-mp4 animation-gif animation-gif-small animation-clean
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
	isort --profile black rlmflow
	black rlmflow
	flake8 rlmflow

install: clean lint
	python -m pip install . --upgrade

doc:
	rm -r docs/reference/
	pdocs as_markdown rlmflow -o docs/reference
	rm docs/reference/rlmflow/index.md
	cp examples/*.ipynb docs/examples/
	cp README.md docs/index.md

serve-docs:
	mkdocs serve

commit: install test doc
	git add .
	git commit -a

test:
	python -m pytest --cov=rlmflow/ --cov-report html:tests/cov-report tests/

test-html: test
	$(BROWSER) tests/cov-report/index.html

OOLONG_N ?= 20
OOLONG_SPLIT ?= validation
OOLONG_WORKERS ?= 1
OOLONG_MODEL ?= gpt-5
OOLONG_MAX_DEPTH ?= 1
OOLONG_MAX_ITERATIONS ?= 20

build-docker-image:
	docker build -t rlmflow:local .

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

# ── Animation (manim) ────────────────────────────────────────────────
ANIMATION_SRC := docs/rlm_animation.py
ANIMATION_SCENE := RLMFlowHero
ANIMATION_OUT_DIR := media/videos/rlm_animation/1080p60

animation: animation-mp4 animation-gif-small ## Render the rlmflow animation: MP4 + share-friendly GIF.

animation-preview: ## Quick low-res manim preview of the rlmflow animation.
	manim -pql $(ANIMATION_SRC) $(ANIMATION_SCENE)

animation-mp4: ## Render docs/rlm_animation.mp4 (1080p60).
	manim -qh $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE).mp4 docs/rlm_animation.mp4

animation-gif: ## Raw 1080p60 GIF from manim — WARNING: ~1GB+ output.
	manim -qh --format=gif $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp "$$(ls -t $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE)*.gif | head -n 1)" docs/rlm_animation.gif

animation-gif-small: animation-mp4 ## Share-friendly GIF (ffmpeg, ~5MB) — replaces the README hero.
	ffmpeg -y -i docs/rlm_animation.mp4 \
		-vf "fps=20,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
		docs/rlm_animation.gif

animation-clean: ## Remove manim render artifacts (media/).
	rm -rf media/