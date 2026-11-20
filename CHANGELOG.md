# Changelog

## Unreleased

### cfspopcon plugin
- `extended_lengyel` is a cfspopcon 9 plugin. Importing it registers nothing; call `cfspopcon.register_plugin("extended_lengyel")` or list it under `plugins` in an input file (the command line interface registers it when `plugins` is omitted). Registering twice is a no-op.
- Default units are declared in `extended_lengyel/variables.yaml`, which replaces `extended_units.yaml`. Algorithms use `declare_algorithm` and `CompositeAlgorithm.declare`; look them up with `cfspopcon.registry["name"]` instead of `Algorithm.get_algorithm`.
- Requires cfspopcon `^9.0.1` and radas `^1!1.0.1`.

### Tooling
- CI runs `ruff check`, `ruff format --check` and `ty check` (pinned to 0.0.107) in a separate `lint` job; `pre-commit` runs ruff locally and rejects AI-attribution trailers in commit messages.
- ruff enforces complexity (`C901`, `max-complexity = 15`) and now covers `tests/`.
- `pytest` fails below 87% coverage (`--cov-fail-under` in `pyproject.toml`).
- Dependabot opens weekly updates for pip/poetry and GitHub Actions.
- Bumped `certifi` and `pillow` in `poetry.lock` to clear advisories.
