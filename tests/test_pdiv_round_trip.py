"""Inverse-to-forward round trip of the divertor-pressure model pair.

The pair is not generally round-trip consistent in the inverse-to-forward
direction. At large impurity fraction the forward fixed-point iteration diverges.
At small impurity fraction it recovers the target temperature.
"""

import numpy as np
import pytest
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, ureg

from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
from extended_lengyel.iterative_solver import (
    run_forward_extended_lengyel_model_pdiv,
    run_inverse_extended_lengyel_model_with_pdiv,
)
from extended_lengyel.mavrin_data import read_mavrin_data


@pytest.fixture(autouse=True)
def raise_on_float_errors():
    """Make silent over/underflow fail this module without leaking into other tests."""
    with np.errstate(over="raise", under="raise"):
        yield


@pytest.fixture(scope="module")
def base_kwargs() -> dict:
    """Base plasma parameters plus Mavrin atomic-data interpolators."""
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


def run_inverse_then_forward(target_electron_temp: float, neutral_pressure_pa: float, base_kwargs: dict) -> dict:
    """Run inverse-pdiv, then forward-pdiv at the impurity fraction it found."""
    inverse_results = run_inverse_extended_lengyel_model_with_pdiv(
        target_electron_temp=target_electron_temp * ureg.eV,
        neutral_pressure_in_divertor=neutral_pressure_pa * ureg.Pa,
        **base_kwargs,
    )
    impurity_fraction = inverse_results[0]
    assert bool(inverse_results[-1]), "inverse-pdiv model did not converge: bad test point"

    forward_results = run_forward_extended_lengyel_model_pdiv(
        impurity_fraction=impurity_fraction,
        neutral_pressure_in_divertor=neutral_pressure_pa * ureg.Pa,
        **base_kwargs,
    )
    return dict(
        impurity_fraction=float(magnitude_in_units(impurity_fraction, ureg.dimensionless)),
        forward_target_electron_temp=float(magnitude_in_units(forward_results[0], ureg.eV)),
        forward_converged=bool(forward_results[-1]),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_inverse_to_forward_round_trip_fails_at_high_impurity_fraction(base_kwargs: dict) -> None:
    """At 2.34 eV and 1 Pa the inverse model needs strong seeding and the forward model diverges."""
    results = run_inverse_then_forward(target_electron_temp=2.34, neutral_pressure_pa=1.0, base_kwargs=base_kwargs)

    assert results["impurity_fraction"] > 0.1
    assert not results["forward_converged"]
    assert np.isnan(results["forward_target_electron_temp"])


def test_inverse_to_forward_round_trip_succeeds_at_low_impurity_fraction(base_kwargs: dict) -> None:
    """At 10 eV and 2 Pa the required impurity fraction is small and the round trip recovers it."""
    results = run_inverse_then_forward(target_electron_temp=10.0, neutral_pressure_pa=2.0, base_kwargs=base_kwargs)

    assert 0.0 < results["impurity_fraction"] < 0.05
    assert results["forward_converged"]
    assert np.isclose(results["forward_target_electron_temp"], 10.0, rtol=1e-3)
