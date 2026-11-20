"""Check (or regenerate) the golden regression file for the extended Lengyel models.

The committed golden file pins the behavior of the original (pre-refactor) model
implementations; it was generated from them before they were retired. The current
models are verified against it by ``test_golden_regression.py`` and by this script.

Usage (from the repo root)::

    poetry run python tests/generate_golden.py --check  # run the current models, compare to the committed file
    poetry run python tests/generate_golden.py --write  # OVERWRITE the golden file from the current models

Only use ``--write`` after an intentional behavior change: it replaces the pinned
reference values. Non-converged cases are dropped (their outputs are iteration-capped
and trajectory-sensitive) and listed in ``metadata.skipped_cases``.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from golden_common import (
    GOLDEN_FILE,
    GOLDEN_RTOL,
    MODEL_RETURN_KEYS,
    MODELS,
    OUTPUT_UNITS,
    apply_overrides,
    build_atomic_interpolators,
    build_base_kwargs,
    compare_case,
    model_outputs,
)


def build_case_grids() -> dict[str, list[dict]]:
    """Per-model lists of input overrides (plain magnitudes, units per OVERRIDE_UNITS)."""
    grids: dict[str, list[dict]] = {model: [] for model in MODELS}

    for target_electron_temp in [1.5, 2.34, 5.0, 10.0, 20.0]:
        for separatrix_electron_density in [2.5, 3.3]:
            grids["inverse"].append(
                dict(target_electron_temp=target_electron_temp, separatrix_electron_density=separatrix_electron_density)
            )
    for power_crossing_separatrix in [4.0, 8.0]:
        grids["inverse"].append(
            dict(
                target_electron_temp=2.34,
                separatrix_electron_density=3.3,
                power_crossing_separatrix=power_crossing_separatrix,
            )
        )

    for impurity_fraction in [0.01, 0.038, 0.06, 0.10]:
        for separatrix_electron_density in [2.5, 3.3]:
            grids["forward"].append(dict(impurity_fraction=impurity_fraction, separatrix_electron_density=separatrix_electron_density))
    for power_crossing_separatrix in [4.0, 8.0]:
        grids["forward"].append(
            dict(
                impurity_fraction=0.038,
                separatrix_electron_density=3.3,
                power_crossing_separatrix=power_crossing_separatrix,
            )
        )

    for target_electron_temp in [2.34, 5.0, 10.0]:
        for neutral_pressure_in_divertor in [1.0, 2.0, 5.0]:
            grids["inverse_pdiv"].append(
                dict(target_electron_temp=target_electron_temp, neutral_pressure_in_divertor=neutral_pressure_in_divertor)
            )

    for impurity_fraction in [0.02, 0.038, 0.06]:
        for neutral_pressure_in_divertor in [1.0, 2.0, 5.0]:
            grids["forward_pdiv"].append(
                dict(impurity_fraction=impurity_fraction, neutral_pressure_in_divertor=neutral_pressure_in_divertor)
            )
    # Higher pressures only converge at low impurity fraction; add convergent points at 2 Pa
    for impurity_fraction in [0.005, 0.01, 0.015]:
        grids["forward_pdiv"].append(dict(impurity_fraction=impurity_fraction, neutral_pressure_in_divertor=2.0))

    return grids


def run_model_case(model_name: str, model_function, overrides: dict, base_kwargs: dict) -> dict:
    """Run one model at one grid point and return outputs as plain magnitudes."""
    kwargs = apply_overrides(base_kwargs, overrides)
    return model_outputs(model_name, model_function(**kwargs))


def generate() -> dict:
    """Run all current models over the grid and assemble the golden-file structure."""
    base_kwargs = dict(build_base_kwargs(), **build_atomic_interpolators())
    grids = build_case_grids()

    skipped_cases = []
    models = {}
    for model_name, model_function in MODELS.items():
        cases = []
        for overrides in grids[model_name]:
            outputs = run_model_case(model_name, model_function, overrides, base_kwargs)
            non_finite = [k for k, v in outputs.items() if k != "converged" and not np.isfinite(v)]
            if not outputs["converged"] or non_finite:
                reason = "non-converged" if not outputs["converged"] else f"non-finite outputs: {non_finite}"
                skipped_cases.append(dict(model=model_name, overrides=overrides, reason=reason))
                continue
            cases.append(dict(overrides=overrides, outputs=outputs))
        models[model_name] = dict(
            function=model_function.__name__,
            return_keys=list(MODEL_RETURN_KEYS[model_name]),
            output_units={k: str(u) for k, u in OUTPUT_UNITS.items() if k in MODEL_RETURN_KEYS[model_name]},
            cases=cases,
        )

    return dict(
        metadata=dict(
            generated_on=datetime.date.today().isoformat(),
            generator="tests/generate_golden.py",
            atomic_data="mavrin",
            source="extended Lengyel models (values pinned from the pre-refactor originals)",
            skipped_cases=skipped_cases,
        ),
        models=models,
    )


def compare_to_committed(golden: dict) -> list[str]:
    """Compare freshly generated values against the committed golden file with ``compare_case``."""
    committed = json.loads(GOLDEN_FILE.read_text())
    mismatches = []
    for model_name, committed_model in committed["models"].items():
        generated_model = golden["models"].get(model_name)
        if generated_model is None:
            mismatches.append(f"{model_name}: missing from generated results")
            continue
        if committed_model["return_keys"] != generated_model["return_keys"]:
            mismatches.append(f"{model_name}: return_keys differ")
        if len(committed_model["cases"]) != len(generated_model["cases"]):
            mismatches.append(f"{model_name}: case count {len(generated_model['cases'])} != committed {len(committed_model['cases'])}")
            continue
        for committed_case, generated_case in zip(committed_model["cases"], generated_model["cases"], strict=True):
            if committed_case["overrides"] != generated_case["overrides"]:
                mismatches.append(f"{model_name} {generated_case['overrides']}: overrides differ from committed grid")
                continue
            mismatches.extend(
                f"{model_name} {committed_case['overrides']}: {message}"
                for message in compare_case(generated_case["outputs"], committed_case["outputs"])
            )
    return mismatches


def main() -> int:
    """Check the current models against the golden file, or overwrite it with --write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the current models against the committed golden file")
    parser.add_argument("--write", action="store_true", help="OVERWRITE the committed golden file from the current models")
    args = parser.parse_args()
    if not (args.check or args.write):
        parser.error("specify --check (compare) or --write (overwrite the pinned reference values)")

    with np.errstate(over="raise", under="raise"):
        golden = generate()
    n_cases = sum(len(m["cases"]) for m in golden["models"].values())
    n_skipped = len(golden["metadata"]["skipped_cases"])
    print(f"Generated {n_cases} cases ({n_skipped} skipped)")
    for skip in golden["metadata"]["skipped_cases"]:
        print(f"  skipped: {skip}")

    if args.check:
        mismatches = compare_to_committed(golden)
        if not mismatches:
            print(f"OK: current models match {GOLDEN_FILE} (rtol={GOLDEN_RTOL})")
            return 0
        print(f"MISMATCH: current models differ from {GOLDEN_FILE}:")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        return 1

    GOLDEN_FILE.parent.mkdir(exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"Wrote {GOLDEN_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
