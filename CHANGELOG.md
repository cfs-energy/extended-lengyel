# Changelog

## Unreleased

### Iterative solver
- `extended_lengyel.iterative_solver` adds four registered algorithms that share one fixed-point loop: `run_forward_extended_lengyel_model` (impurity fraction in, target temperature out), `run_inverse_extended_lengyel_model` (the reverse), and the `_pdiv` pair, which solve for the separatrix density from a given divertor neutral pressure instead of taking it as input.
- Every solve returns `converged`. A non-converged or unphysical solve (inverse needing a negative impurity fraction, or separatrix Z_eff below 1) returns NaN for all physics outputs, so mask on the flag.
- Optional inputs, inert at their defaults: `pressure_broadening_factor`, `divertor_enrichment_factor`, `flux_expansion_Bt_over_Bu`, `flux_expansion_Bxpt_over_Bu` (Moulton flux-expansion term).
- The loop evaluates heat conductivity separately in the divertor and main chamber, so it differs by a few percent in target temperature from `run_extended_lengyel_model_with_S_Zeff_and_alphat_correction`; do not mix the two.
- Example: `bench/forward_inverse_example.py`.

### Golden regression test
- `tests/golden/extended_lengyel_golden.json` pins the four iterative models (forward, inverse and their `_pdiv` variants) on a grid of inputs; `tests/test_golden_regression.py` and `python tests/generate_golden.py --check` compare against it (relative tolerance 1e-5, `converged` exactly). Non-converged grid points are not pinned; they are listed in the file's `metadata.skipped_cases`.
- Regenerate with `--write` only after a deliberate, reviewed physics change, never to make a failing check pass.

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
