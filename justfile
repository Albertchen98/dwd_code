default:
  just --list

# Setup the repository
setup:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

# Install the repository
install:
  uv sync

# Run pre-commit
pre-commit *args: setup
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

# Run pyrefly
pyrefly *args:
  uv run pyrefly check --output-format=min-text --remove-unused-ignores {{args}}

# Run pyrefly and whitelist all errors
pyrefly-ignore *args:
  just -f {{source_file()}} pyrefly --suppress-errors {{args}}

# Run linting and formatting
lint: pre-commit pyrefly

# https://spdx.org/licenses/
allow_licenses := "MIT BSD-2-CLAUSE BSD-3-CLAUSE APACHE-2.0 ISC"
ignore_package_licenses := "nvidia-* hf-xet certifi filelock matplotlib typing-extensions sentencepiece"

# Update the license
license: install
  uvx licensecheck --show-only-failing --only-licenses {{allow_licenses}} --ignore-packages {{ignore_package_licenses}} --zero
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md
  pre-commit run --files ATTRIBUTIONS.md || true

# Run the docker container
docker:
  # https://github.com/astral-sh/uv-docker-example/blob/main/run.sh
  docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it $(docker build -q .)
