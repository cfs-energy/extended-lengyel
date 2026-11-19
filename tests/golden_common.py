"""Shared setup for golden-file generation and regression tests.

This module is the single source of truth for the base plasma parameters and
atomic-data interpolators used by both ``generate_golden.py`` and
``test_golden_regression.py``. The committed golden file pins the behavior of
the original (pre-refactor) model implementations; keeping the setup in one
place ensures the generator and the tests cannot drift apart.
"""

from pathlib import Path

import numpy as np
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, ureg

from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
from extended_lengyel.iterative_solver import (
    run_forward_extended_lengyel_model,
    run_forward_extended_lengyel_model_pdiv,
    run_inverse_extended_lengyel_model,
    run_inverse_extended_lengyel_model_with_pdiv,
)
from extended_lengyel.mavrin_data import read_mavrin_data

GOLDEN_FILE = Path(__file__).parent / "golden" / "extended_lengyel_golden.json"
MODELS = {
    "forward": run_forward_extended_lengyel_model,
    "forward_pdiv": run_forward_extended_lengyel_model_pdiv,
    "inverse": run_inverse_extended_lengyel_model,
    "inverse_pdiv": run_inverse_extended_lengyel_model_with_pdiv,
}

# Units used to (de)serialize the per-case input overrides in the golden file.
OVERRIDE_UNITS = {
    "target_electron_temp": ureg.eV,
    "impurity_fraction": ureg.dimensionless,
    "separatrix_electron_density": ureg.n19,
    "neutral_pressure_in_divertor": ureg.Pa,
    "power_crossing_separatrix": ureg.MW,
}

# Units used to (de)serialize the model outputs in the golden file.
# converged is stored as a plain bool (unit None).
OUTPUT_UNITS = {
    "target_electron_temp": ureg.eV,
    "impurity_fraction": ureg.dimensionless,
    "separatrix_electron_density": ureg.m**-3,
    "parallel_ion_flux_to_target": ureg.m**-2 * ureg.s**-1,
    "neutral_pressure_in_divertor": ureg.Pa,
    "alpha_t": ureg.dimensionless,
    "q_parallel": ureg.W / ureg.m**2,
    "heat_flux_perp_to_target": ureg.W / ureg.m**2,
    "separatrix_z_effective": ureg.dimensionless,
    "separatrix_electron_temp": ureg.eV,
    "converged": None,
}

# Return-tuple keys for each model variant, in the order returned by the
# original (and refactored) functions.
MODEL_RETURN_KEYS = {
    "forward": (
        "target_electron_temp",
        "parallel_ion_flux_to_target",
        "neutral_pressure_in_divertor",
        "alpha_t",
        "q_parallel",
        "heat_flux_perp_to_target",
        "separatrix_z_effective",
        "converged",
    ),
    "forward_pdiv": (
        "target_electron_temp",
        "separatrix_electron_density",
        "parallel_ion_flux_to_target",
        "alpha_t",
        "q_parallel",
        "heat_flux_perp_to_target",
        "separatrix_z_effective",
        "converged",
    ),
    "inverse": (
        "impurity_fraction",
        "parallel_ion_flux_to_target",
        "neutral_pressure_in_divertor",
        "alpha_t",
        "q_parallel",
        "heat_flux_perp_to_target",
        "separatrix_z_effective",
        "converged",
    ),
    "inverse_pdiv": (
        "impurity_fraction",
        "separatrix_electron_density",
        "parallel_ion_flux_to_target",
        "alpha_t",
        "q_parallel",
        "heat_flux_perp_to_target",
        "separatrix_z_effective",
        "separatrix_electron_temp",
        "converged",
    ),
}


# Relative tolerance of the golden comparison; the current models reproduce the pinned values to ~1e-15.
GOLDEN_RTOL = 1e-5


def model_outputs(model_name: str, results: tuple) -> dict:
    """Map a model's return tuple to plain magnitudes (floats, and a bool for ``converged``) keyed by output name."""
    outputs = {}
    for key, value in zip(MODEL_RETURN_KEYS[model_name], results, strict=True):
        unit = OUTPUT_UNITS[key]
        outputs[key] = bool(value) if unit is None else float(magnitude_in_units(value, unit))
    return outputs


def compare_case(actual: dict, expected: dict) -> list[str]:
    """Compare one case's outputs against the pinned ones, returning one message per mismatch.

    This is the single criterion behind both ``generate_golden.py --check`` and the regression
    test: ``converged`` must match exactly, other outputs within GOLDEN_RTOL (NaN never matches).
    """
    mismatches = []
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"{key}: missing from actual outputs")
            continue
        actual_value = actual[key]
        if key == "converged":
            ok = actual_value == expected_value
        else:
            atol = 1e-30 if key == "impurity_fraction" else 0.0
            ok = np.isclose(actual_value, expected_value, rtol=GOLDEN_RTOL, atol=atol)
        if not ok:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    return mismatches


def apply_overrides(base_kwargs: dict, overrides: dict) -> dict:
    """Return base_kwargs updated with per-case overrides (magnitudes -> pint quantities)."""
    kwargs = dict(base_kwargs)
    for key, magnitude in overrides.items():
        kwargs[key] = magnitude * OVERRIDE_UNITS[key]
    return kwargs


def build_atomic_interpolators() -> dict:
    """Build the seed/fixed impurity interpolator kwargs used by the golden cases.

    Nitrogen + Argon seed impurities (weights 1.0, 0.05) and Helium fixed
    impurity (weight 1e-2), from the Mavrin data the golden file was generated with.
    """
    atomic_data = read_mavrin_data()
    seed, fixed = [AtomicSpecies.Nitrogen, AtomicSpecies.Argon], [AtomicSpecies.Helium]
    return dict(
        CzLINT_for_seed_impurities=CzLINT_integrator.from_list(seed, [1.0, 0.05], atomic_data),
        mean_charge_for_seed_impurities=Mean_charge_interpolator.from_list(seed, atomic_data),
        CzLINT_for_fixed_impurities=CzLINT_integrator.from_list(fixed, [1.0e-2], atomic_data),
        mean_charge_for_fixed_impurities=Mean_charge_interpolator.from_list(fixed, atomic_data),
    )


def build_base_kwargs() -> dict:
    """Base plasma parameters shared by all golden cases.

    Matches the kwargs in ``test_forward_lengyel_model.py``, minus the
    interpolators (see :func:`build_atomic_interpolators`) and minus the
    per-variant driving/constraint inputs (impurity_fraction or
    target_electron_temp; separatrix_electron_density or
    neutral_pressure_in_divertor), which are supplied per golden case.
    """
    return dict(
        power_crossing_separatrix=5.5 * ureg.MW,
        divertor_broadening_factor=3.0,
        magnetic_field_on_axis=2.5 * ureg.T,
        plasma_current=1.0 * ureg.MA,
        parallel_connection_length=20.0 * ureg.m,
        divertor_parallel_length=5.0 * ureg.m,
        major_radius=1.65 * ureg.m,
        minor_radius=0.5 * ureg.m,
        elongation_psi95=1.6,
        triangularity_psi95=0.3,
        target_angle_of_incidence=3.0,
        fraction_of_P_SOL_to_divertor=2.0 / 3.0,
        sheath_heat_transmission_factor=8.0,
        iterations=100,
    )
