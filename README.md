# energy_project

A small project workspace for energy-related code.

## Ruff (linting and formatting)

This repository includes a basic Ruff configuration and CI hook.

- Config: [pyproject.toml](pyproject.toml)
- Pre-commit config: [.pre-commit-config.yaml](.pre-commit-config.yaml)
- GitHub Actions workflow: [.github/workflows/ruff.yml](.github/workflows/ruff.yml)

Quick start:

```bash
# install ruff (recommended in a venv or via pipx)
pip install --upgrade pip
pip install ruff

# run a lint pass
ruff check .

# check formatting (fails if formatting needed)
ruff format --check .

# (optional) set up pre-commit hooks after installing pre-commit and ruff
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
# energy_project
