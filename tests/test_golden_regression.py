"""Golden-regression test for the iterative extended Lengyel models .

The models are compared against golden reference values pinned from the original
(pre-refactor) implementations (see tests/generate_golden.py). The comparison criterion is
``compare_case``, shared with ``generate_golden.py --check``; its guards are tested below.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from cfspopcon.unit_handling import magnitude_in_units, ureg

sys.path.insert(0, str(Path(__file__).parent))

import generate_golden
from golden_common import (
    GOLDEN_FILE,
    GOLDEN_RTOL,
    MODELS,
    apply_overrides,
    build_atomic_interpolators,
    build_base_kwargs,
    compare_case,
    model_outputs,
)


@pytest.fixture(autouse=True)
def _raise_on_float_overflow():
    """Fail on over/underflow in this module only; a module-level np.seterr would leak into later tests."""
    with np.errstate(over="raise", under="raise"):
        yield


GOLDEN = json.loads(GOLDEN_FILE.read_text())

GOLDEN_CASES = [
    pytest.param(model_name, case, id=f"{model_name}-{i}")
    for model_name, model_data in GOLDEN["models"].items()
    for i, case in enumerate(model_data["cases"])
]


@pytest.fixture(scope="module")
def base_kwargs() -> dict:
    """Base plasma parameters plus Mavrin atomic-data interpolators."""
    return dict(build_base_kwargs(), **build_atomic_interpolators())


@pytest.mark.parametrize(("model_name", "case"), GOLDEN_CASES)
def test_golden_regression(model_name: str, case: dict, base_kwargs: dict):
    """Each model must reproduce the golden outputs pinned from the original implementation."""
    kwargs = apply_overrides(base_kwargs, case["overrides"])
    actual = model_outputs(model_name, MODELS[model_name](**kwargs))

    assert compare_case(actual, case["outputs"]) == []


def test_forward_first_pdiv_round_trip(base_kwargs: dict):
    """Forward-pdiv -> inverse-pdiv round trip recovers the impurity fraction and density.

    The round trip is run forward-first: the opposite direction (inverse then forward) does
    not generally converge, in the original models as well as the refactored ones, because the
    forward-pdiv fixed-point iteration cannot reach the deeply-detached solutions that the
    inverse model constructs directly.
    """
    impurity_fraction_in = 0.038
    neutral_pressure = 1.0 * ureg.Pa

    (
        target_electron_temp_out,
        FWD_separatrix_electron_density,
        _FWD_parallel_ion_flux_to_target,
        _FWD_alpha_t,
        _FWD_q_parallel,
        _FWD_heat_flux_perp_to_target,
        _FWD_separatrix_z_effective,
        FWD_converged,
    ) = MODELS["forward_pdiv"](impurity_fraction=impurity_fraction_in, neutral_pressure_in_divertor=neutral_pressure, **base_kwargs)
    assert FWD_converged

    (
        c_z_out,
        INV_separatrix_electron_density,
        _INV_parallel_ion_flux_to_target,
        _INV_alpha_t,
        _INV_q_parallel,
        _INV_heat_flux_perp_to_target,
        _INV_separatrix_z_effective,
        _INV_separatrix_electron_temp,
        INV_converged,
    ) = MODELS["inverse_pdiv"](target_electron_temp=target_electron_temp_out, neutral_pressure_in_divertor=neutral_pressure, **base_kwargs)
    assert INV_converged

    assert np.isclose(magnitude_in_units(c_z_out, ureg.dimensionless), impurity_fraction_in, rtol=1e-4)
    assert np.isclose(
        magnitude_in_units(INV_separatrix_electron_density, ureg.m**-3),
        magnitude_in_units(FWD_separatrix_electron_density, ureg.m**-3),
        rtol=1e-4,
    )


EXPECTED = dict(target_electron_temp=2.34, impurity_fraction=0.04, converged=True)


@pytest.mark.parametrize(
    ("changes", "n_mismatches"),
    [
        ({}, 0),
        ({"target_electron_temp": 2.34 * (1 + 0.5 * GOLDEN_RTOL)}, 0),
        ({"target_electron_temp": 2.34 * (1 + 2 * GOLDEN_RTOL)}, 1),
        ({"target_electron_temp": float("nan")}, 1),
        ({"target_electron_temp": float("inf")}, 1),
        ({"converged": False}, 1),
        ({"impurity_fraction": 0.0}, 1),
    ],
    ids=["equal", "within_rtol", "beyond_rtol", "nan", "inf", "converged_flipped", "zero_vs_finite"],
)
def test_compare_case_criterion(changes: dict, n_mismatches: int):
    """Outputs match within GOLDEN_RTOL, ``converged`` matches exactly, and NaN or inf never matches a finite value."""
    assert len(compare_case(dict(EXPECTED, **changes), EXPECTED)) == n_mismatches


def test_compare_case_nan_never_matches_nan():
    """A NaN pinned value cannot be reproduced: NaN outputs are mismatches even against NaN."""
    nan = float("nan")
    assert len(compare_case(dict(EXPECTED, target_electron_temp=nan), dict(EXPECTED, target_electron_temp=nan))) == 1


def test_compare_case_missing_key():
    """An output missing from the actual results is a mismatch, not a KeyError."""
    actual = {k: v for k, v in EXPECTED.items() if k != "impurity_fraction"}
    assert compare_case(actual, EXPECTED) == ["impurity_fraction: missing from actual outputs"]


def perturbed(golden: dict, model_name: str, key: str, factor: float) -> dict:
    """Copy of ``golden`` with one output of the model's first case scaled."""
    golden = copy.deepcopy(golden)
    golden["models"][model_name]["cases"][0]["outputs"][key] *= factor
    return golden


@pytest.fixture
def committed_golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``generate_golden`` at a temporary copy of the golden file; yields the golden dict."""
    path = tmp_path / "golden.json"
    path.write_text(GOLDEN_FILE.read_text())
    monkeypatch.setattr(generate_golden, "GOLDEN_FILE", path)
    return json.loads(path.read_text())


def test_check_accepts_identical_results(committed_golden):
    """Results equal to the committed file give no mismatches."""
    assert generate_golden.compare_to_committed(committed_golden) == []


@pytest.mark.parametrize("model_name", sorted(GOLDEN["models"]))
def test_check_flags_a_perturbed_output(committed_golden, model_name: str):
    """Scaling any model's output beyond the tolerance is reported against that model."""
    golden = committed_golden
    key = next(k for k in golden["models"][model_name]["return_keys"] if k != "converged")
    mismatches = generate_golden.compare_to_committed(perturbed(golden, model_name, key, 1 + 10 * GOLDEN_RTOL))
    assert len(mismatches) == 1
    assert mismatches[0].startswith(model_name)
    assert key in mismatches[0]


def test_check_flags_structural_differences(committed_golden):
    """A missing model, a changed case count or a changed grid point is reported."""
    golden = committed_golden
    missing = copy.deepcopy(golden)
    del missing["models"]["forward"]
    assert generate_golden.compare_to_committed(missing) == ["forward: missing from generated results"]

    fewer = copy.deepcopy(golden)
    fewer["models"]["inverse"]["cases"].pop()
    assert "case count" in generate_golden.compare_to_committed(fewer)[0]

    moved = copy.deepcopy(golden)
    first_overrides = moved["models"]["inverse"]["cases"][0]["overrides"]
    first_overrides[next(iter(first_overrides))] *= 1.1
    assert "overrides differ" in generate_golden.compare_to_committed(moved)[0]


def test_check_exit_code(committed_golden: dict, monkeypatch: pytest.MonkeyPatch):
    """``--check`` exits 0 for matching results and 1 for a perturbed one."""
    golden = committed_golden
    monkeypatch.setattr(sys, "argv", ["generate_golden.py", "--check"])

    monkeypatch.setattr(generate_golden, "generate", lambda: golden)
    assert generate_golden.main() == 0

    monkeypatch.setattr(generate_golden, "generate", lambda: perturbed(golden, "forward", "alpha_t", 1.001))
    assert generate_golden.main() == 1
