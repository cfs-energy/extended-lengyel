"""Edge cases of the iterative solver: failure reporting, NaN inputs and xarray broadcasting."""

import numpy as np
import pytest
import xarray as xr
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, ureg

from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
from extended_lengyel.iterative_solver import (
    run_forward_extended_lengyel_model,
    run_forward_extended_lengyel_model_pdiv,
    run_inverse_extended_lengyel_model,
    run_inverse_extended_lengyel_model_with_pdiv,
)
from extended_lengyel.iterative_solver.units import (
    FORWARD_PDIV_RETURN_KEYS,
    FORWARD_RETURN_KEYS,
    INVERSE_PDIV_RETURN_KEYS,
    INVERSE_RETURN_KEYS,
    RETURN_UNITS,
)
from extended_lengyel.mavrin_data import read_mavrin_data


@pytest.fixture(scope="module")
def kwargs() -> dict:
    """Plasma parameters with a nitrogen seed, Mavrin atomic data."""
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
        CzLINT_for_seed_impurities=CzLINT_integrator.from_list([AtomicSpecies.Nitrogen], [1.0], atomic_data),
        mean_charge_for_seed_impurities=Mean_charge_interpolator.from_list([AtomicSpecies.Nitrogen], atomic_data),
    )


def as_arrays(keys, results) -> dict:
    """Pair model outputs with their names and strip units, keeping any xarray dimensions as ndarray axes."""
    outputs = dict(zip(keys, results, strict=True))
    return {
        name: np.asarray(value if RETURN_UNITS[name] is None else magnitude_in_units(value, RETURN_UNITS[name]))
        for name, value in outputs.items()
    }


def forward(kwargs, impurity_fraction, **overrides):
    """Forward solve at a fixed separatrix density, returned as {output name: value}."""
    args = dict(impurity_fraction=impurity_fraction * ureg.dimensionless, separatrix_electron_density=3.3 * ureg.n19)
    return as_arrays(FORWARD_RETURN_KEYS, run_forward_extended_lengyel_model(**{**kwargs, **args, **overrides}))


def inverse(kwargs, target_electron_temp, **overrides):
    """Inverse solve at a fixed separatrix density, returned as {output name: value}."""
    args = dict(target_electron_temp=target_electron_temp * ureg.eV, separatrix_electron_density=3.3 * ureg.n19)
    return as_arrays(INVERSE_RETURN_KEYS, run_inverse_extended_lengyel_model(**{**kwargs, **args, **overrides}))


def assert_failed_solve_is_all_nan(outputs):
    """A failed solve reports converged=False and NaN in every other output."""
    assert not bool(outputs["converged"])
    for name, value in outputs.items():
        if name != "converged":
            assert np.all(np.isnan(value)), name


def test_array_input_isolates_the_failed_element(kwargs):
    """A NaN or non-convergent element of an array input fails alone; the others equal their scalar solves exactly."""
    fractions = [0.01, 0.02, np.nan, 0.5]
    batch = forward(kwargs, xr.DataArray(fractions, dims="case"))

    assert batch["converged"].tolist() == [True, True, False, False]
    for index, fraction in enumerate(fractions[:2]):
        single = forward(kwargs, fraction)
        for name, value in single.items():
            assert batch[name][index] == value, name
    for index in (2, 3):
        assert_failed_solve_is_all_nan({name: value[index] for name, value in batch.items()})


@pytest.mark.parametrize("call", [lambda k: forward(k, np.nan), lambda k: inverse(k, np.nan)], ids=["forward", "inverse"])
def test_nan_driving_input_fails_cleanly(kwargs, call):
    """A NaN impurity fraction or target temperature gives a non-converged all-NaN result, not an exception."""
    assert_failed_solve_is_all_nan(call(kwargs))


def test_iteration_cap_without_convergence_fails_cleanly(kwargs):
    """Hitting the iteration cap reports failure even for a case that converges given more iterations."""
    assert bool(forward(kwargs, 0.01)["converged"])
    assert_failed_solve_is_all_nan(forward(kwargs, 0.01, iterations=1))


def test_inverse_target_needing_negative_seeding_is_rejected_by_the_physicality_gate(kwargs):
    """A converged fixed point with negative impurity fraction is reported as failed."""
    # The fixed point converges at a negative impurity fraction (separatrix Z_eff < 1) at the denser
    # separatrix, so only the physicality gate turns it into a failure.
    assert bool(inverse(kwargs, 15.0, separatrix_electron_density=3.3 * ureg.n19)["converged"])
    assert_failed_solve_is_all_nan(inverse(kwargs, 15.0, separatrix_electron_density=6.0 * ureg.n19))


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_inverse_target_above_the_unseeded_two_point_model_fails_cleanly(kwargs):
    """A target above the reachable range fails without raising."""
    assert_failed_solve_is_all_nan(inverse(kwargs, 400.0))


def test_inverse_forward_round_trip_recovers_target_temperature(kwargs):
    """The forward solve at the inverse solve's impurity fraction recovers the target temperature."""
    seeded = inverse(kwargs, 5.0)
    assert bool(seeded["converged"])
    recovered = forward(kwargs, seeded["impurity_fraction"])
    assert bool(recovered["converged"])
    np.testing.assert_allclose(float(recovered["target_electron_temp"]), 5.0, rtol=1e-3)


def test_pdiv_round_trip_recovers_target_temperature_and_separatrix_density(kwargs):
    """The pdiv pair recovers both the target temperature and the separatrix density."""
    pressure = 2.0 * ureg.Pa
    seeded = as_arrays(
        INVERSE_PDIV_RETURN_KEYS,
        run_inverse_extended_lengyel_model_with_pdiv(target_electron_temp=10.0 * ureg.eV, neutral_pressure_in_divertor=pressure, **kwargs),
    )
    assert bool(seeded["converged"])

    recovered = as_arrays(
        FORWARD_PDIV_RETURN_KEYS,
        run_forward_extended_lengyel_model_pdiv(
            impurity_fraction=seeded["impurity_fraction"] * ureg.dimensionless, neutral_pressure_in_divertor=pressure, **kwargs
        ),
    )
    assert bool(recovered["converged"])
    np.testing.assert_allclose(recovered["target_electron_temp"], 10.0, rtol=1e-3)
    np.testing.assert_allclose(recovered["separatrix_electron_density"], seeded["separatrix_electron_density"], rtol=1e-3)
