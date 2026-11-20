# Changelog

## Unreleased

### Tooling
- CI runs `ruff check`, `ruff format --check` and `ty check` (pinned to 0.0.107) in a separate `lint` job; `pre-commit` runs ruff locally.
- ruff enforces complexity (`C901`, `max-complexity = 15`) and now covers `tests/`.
- `pytest` fails below 87% coverage (`--cov-fail-under` in `pyproject.toml`).
- Dependabot opens weekly updates for pip/poetry and GitHub Actions.
- Bumped `certifi` and `pillow` in `poetry.lock` to clear advisories.
