"""Tests for pressure broadening (b_pres) and divertor enrichment (f_enrich).

At the defaults ``pressure_broadening_factor = divertor_enrichment_factor = 1`` the
factors are inert. These tests check the derived behaviour: b_pres enhances the
upstream-referenced momentum loss and lowers the divertor neutral pressure, and
f_enrich weakens main-chamber radiation so more seed impurity is required.
"""

import numpy as np
import pytest
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, ureg

from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
from extended_lengyel.iterative_solver import run_inverse_extended_lengyel_model
from extended_lengyel.mavrin_data import read_mavrin_data


@pytest.fixture(autouse=True)
def raise_on_float_errors():
    """Make silent over/underflow fail this module without leaking into other tests."""
    with np.errstate(over="raise", under="raise"):
        yield


def _base() -> dict:
    """Base plasma parameters plus Nitrogen + Argon seed and Helium fixed interpolators."""
    atomic_data = read_mavrin_data()
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
        CzLINT_for_seed_impurities=CzLINT_integrator.from_list(
            [AtomicSpecies.Nitrogen, AtomicSpecies.Argon], [1.0, 0.05], atomic_data
        ),
        mean_charge_for_seed_impurities=Mean_charge_interpolator.from_list(
            [AtomicSpecies.Nitrogen, AtomicSpecies.Argon], atomic_data
        ),
        CzLINT_for_fixed_impurities=CzLINT_integrator.from_list([AtomicSpecies.Helium], [1.0e-2], atomic_data),
        mean_charge_for_fixed_impurities=Mean_charge_interpolator.from_list([AtomicSpecies.Helium], atomic_data),
    )


def _run_inverse(base: dict, **overrides: float) -> dict:
    """Run the inverse model at one operating point, varying only the new factors."""
    kwargs = dict(base)
    kwargs.update(overrides)
    result = run_inverse_extended_lengyel_model(
        target_electron_temp=5.0 * ureg.eV,
        separatrix_electron_density=3.0e19 * ureg.m**-3,
        **kwargs,
    )
    return dict(
        impurity_fraction=float(magnitude_in_units(result[0], ureg.dimensionless)),
        neutral_pressure=float(magnitude_in_units(result[2], ureg.Pa)),
        converged=bool(result[-1]),
    )


def test_defaults_identity() -> None:
    """Setting b_pres = f_enrich = 1 reproduces the default-argument result."""
    base = _base()
    default = _run_inverse(base)
    explicit = _run_inverse(base, pressure_broadening_factor=1.0, divertor_enrichment_factor=1.0)

    assert default["converged"] and explicit["converged"]
    assert default["impurity_fraction"] == explicit["impurity_fraction"]
    assert default["neutral_pressure"] == explicit["neutral_pressure"]


def test_pressure_broadening_reduces_divertor_pressure() -> None:
    """Increasing b_pres enhances the total momentum loss and lowers the divertor pressure."""
    base = _base()
    results = [_run_inverse(base, pressure_broadening_factor=factor) for factor in (1.0, 1.5, 2.0)]

    assert all(result["converged"] for result in results)
    pressures = [result["neutral_pressure"] for result in results]
    assert pressures[0] > pressures[1] > pressures[2]


def test_enrichment_increases_required_impurity() -> None:
    """Increasing f_enrich weakens main-chamber radiation, so more seed impurity is required."""
    base = _base()
    results = [_run_inverse(base, divertor_enrichment_factor=factor) for factor in (1.0, 1.5, 2.0)]

    assert all(result["converged"] for result in results)
    concentrations = [result["impurity_fraction"] for result in results]
    assert concentrations[0] < concentrations[1] < concentrations[2]
