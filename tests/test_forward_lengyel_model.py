"""Tests for the iterative forward extended Lengyel model.

The reference values below are a regression baseline: they were produced by the
original (pre-refactor) implementations of these models and are pinned in the
golden file of the extended-lengyel-validation project, which remains the
authoritative regression suite. They use the Mavrin polynomial atomic data, so
this test does not depend on a radas dataset being built.
"""

import numpy as np
import pytest
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, ureg

from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
from extended_lengyel.iterative_solver import run_forward_extended_lengyel_model, run_inverse_extended_lengyel_model
from extended_lengyel.mavrin_data import read_mavrin_data


@pytest.fixture(autouse=True)
def raise_on_float_errors():
    """Make silent over/underflow in the solver fail these tests (scoped, so it does not leak)."""
    with np.errstate(over="raise", under="raise"):
        yield


# Plasma parameters shared by every reference case.
BASE_KWARGS = dict(
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

# Units of the per-case inputs and of every model output.
OVERRIDE_UNITS = dict(
    impurity_fraction=ureg.dimensionless,
    target_electron_temp=ureg.eV,
    separatrix_electron_density=ureg.n19,
    power_crossing_separatrix=ureg.MW,
)
OUTPUT_UNITS = dict(
    target_electron_temp=ureg.eV,
    impurity_fraction=ureg.dimensionless,
    parallel_ion_flux_to_target=ureg.m**-2 / ureg.s,
    neutral_pressure_in_divertor=ureg.Pa,
    alpha_t=ureg.dimensionless,
    q_parallel=ureg.W / ureg.m**2,
    heat_flux_perp_to_target=ureg.W / ureg.m**2,
    separatrix_z_effective=ureg.dimensionless,
)

FORWARD_RETURN_KEYS = (
    "target_electron_temp",
    "parallel_ion_flux_to_target",
    "neutral_pressure_in_divertor",
    "alpha_t",
    "q_parallel",
    "heat_flux_perp_to_target",
    "separatrix_z_effective",
)

# (inputs, expected outputs) for the converged reference cases.
FORWARD_REFERENCE_CASES = [
    (
        dict(impurity_fraction=0.01, separatrix_electron_density=2.5),
        dict(
            target_electron_temp=65.49112461678739,
            parallel_ion_flux_to_target=1.4681064591614932e24,
            neutral_pressure_in_divertor=0.5050966227756978,
            alpha_t=0.17692094272990122,
            q_parallel=436134412.5994577,
            heat_flux_perp_to_target=6449715.261109519,
            separatrix_z_effective=1.2406625829525029,
        ),
    ),
    (
        dict(impurity_fraction=0.01, separatrix_electron_density=3.3),
        dict(
            target_electron_temp=35.94328406827099,
            parallel_ion_flux_to_target=2.4609526880213976e24,
            neutral_pressure_in_divertor=0.8466817128781872,
            alpha_t=0.260053003958726,
            q_parallel=411740890.595647,
            heat_flux_perp_to_target=5933645.0604971405,
            separatrix_z_effective=1.241869557819673,
        ),
    ),
    (
        dict(impurity_fraction=0.038, separatrix_electron_density=2.5),
        dict(
            target_electron_temp=53.977817609083225,
            parallel_ion_flux_to_target=1.6112141020396254e24,
            neutral_pressure_in_divertor=0.5543322804897981,
            alpha_t=0.24624239649550964,
            q_parallel=406424307.37085736,
            heat_flux_perp_to_target=5834036.209256478,
            separatrix_z_effective=1.8590885204144865,
        ),
    ),
    (
        dict(impurity_fraction=0.038, separatrix_electron_density=3.3),
        dict(
            target_electron_temp=2.4566047419660366,
            parallel_ion_flux_to_target=5.0708001889293e24,
            neutral_pressure_in_divertor=1.74459013800771,
            alpha_t=0.3598835042569608,
            q_parallel=364587978.86462486,
            heat_flux_perp_to_target=835626.7383516281,
            separatrix_z_effective=1.8640696385197995,
        ),
    ),
    (
        dict(impurity_fraction=0.06, separatrix_electron_density=2.5),
        dict(
            target_electron_temp=43.08974428274713,
            parallel_ion_flux_to_target=1.800246557385063e24,
            neutral_pressure_in_divertor=0.6193682008715609,
            alpha_t=0.2962506740650417,
            q_parallel=383753753.24930555,
            heat_flux_perp_to_target=5203630.496488394,
            separatrix_z_effective=2.344755836032303,
        ),
    ),
    (
        dict(impurity_fraction=0.038, separatrix_electron_density=3.3, power_crossing_separatrix=8.0),
        dict(
            target_electron_temp=45.91719174198923,
            parallel_ion_flux_to_target=2.4741026848392794e24,
            neutral_pressure_in_divertor=0.8512059208746677,
            alpha_t=0.28329227383734557,
            q_parallel=546823857.8257205,
            heat_flux_perp_to_target=7620677.517582321,
            separatrix_z_effective=1.879207470757456,
        ),
    ),
]

# Inputs for which the fixed-point iteration is known not to converge.
FORWARD_NON_CONVERGENT_CASES = [
    dict(impurity_fraction=0.06, separatrix_electron_density=3.3),
    dict(impurity_fraction=0.1, separatrix_electron_density=2.5),
    dict(impurity_fraction=0.1, separatrix_electron_density=3.3),
    dict(impurity_fraction=0.038, separatrix_electron_density=3.3, power_crossing_separatrix=4.0),
]


@pytest.fixture(scope="module")
def model_kwargs() -> dict:
    """Base plasma parameters plus the Nitrogen + Argon seed and Helium fixed impurity interpolators."""
    atomic_data = read_mavrin_data()
    seed_species = [AtomicSpecies.Nitrogen, AtomicSpecies.Argon]
    seed_weights = [1.0, 0.05]
    fixed_species = [AtomicSpecies.Helium]
    fixed_weights = [1.0e-2]

    return dict(
        BASE_KWARGS,
        CzLINT_for_seed_impurities=CzLINT_integrator.from_list(seed_species, seed_weights, atomic_data),
        mean_charge_for_seed_impurities=Mean_charge_interpolator.from_list(seed_species, atomic_data),
        CzLINT_for_fixed_impurities=CzLINT_integrator.from_list(fixed_species, fixed_weights, atomic_data),
        mean_charge_for_fixed_impurities=Mean_charge_interpolator.from_list(fixed_species, atomic_data),
    )


def apply_overrides(model_kwargs: dict, overrides: dict) -> dict:
    """Return model_kwargs with the per-case overrides applied, converting magnitudes to quantities."""
    return dict(model_kwargs, **{key: value * OVERRIDE_UNITS[key] for key, value in overrides.items()})


@pytest.mark.parametrize(
    ("overrides", "expected"), FORWARD_REFERENCE_CASES, ids=[str(case[0]) for case in FORWARD_REFERENCE_CASES]
)
def test_forward_model_against_reference_values(model_kwargs, overrides, expected):
    """The forward model reproduces the reference values of the original implementation."""
    *outputs, converged = run_forward_extended_lengyel_model(**apply_overrides(model_kwargs, overrides))
    assert converged

    for key, output in zip(FORWARD_RETURN_KEYS, outputs, strict=True):
        assert np.isclose(magnitude_in_units(output, OUTPUT_UNITS[key]), expected[key], rtol=1e-5, atol=0.0), key


@pytest.mark.parametrize("overrides", FORWARD_NON_CONVERGENT_CASES, ids=[str(case) for case in FORWARD_NON_CONVERGENT_CASES])
def test_forward_model_reports_non_convergence(model_kwargs, overrides):
    """A non-converged solve reports converged=False and returns NaN for every physics output."""
    *outputs, converged = run_forward_extended_lengyel_model(**apply_overrides(model_kwargs, overrides))
    assert not converged

    for key, output in zip(FORWARD_RETURN_KEYS, outputs, strict=True):
        assert np.isnan(magnitude_in_units(output, OUTPUT_UNITS[key])), key


def test_forward_model_inverts_the_inverse_model(model_kwargs):
    """Feeding the inverse model's impurity fraction back into the forward model recovers its inputs."""
    target_electron_temp = 2.34 * ureg.eV

    impurity_fraction, *inverse_outputs, inverse_converged = run_inverse_extended_lengyel_model(
        target_electron_temp=target_electron_temp, separatrix_electron_density=3.3 * ureg.n19, **model_kwargs
    )
    assert inverse_converged

    target_electron_temp_out, *forward_outputs, forward_converged = run_forward_extended_lengyel_model(
        impurity_fraction=impurity_fraction, separatrix_electron_density=3.3 * ureg.n19, **model_kwargs
    )
    assert forward_converged

    assert np.isclose(
        magnitude_in_units(target_electron_temp_out, ureg.eV), magnitude_in_units(target_electron_temp, ureg.eV), rtol=1e-3
    )
    # Both solves describe the same plasma state, so every derived output must agree.
    for key, inverse_output, forward_output in zip(FORWARD_RETURN_KEYS[1:], inverse_outputs, forward_outputs, strict=True):
        assert np.isclose(
            magnitude_in_units(inverse_output, OUTPUT_UNITS[key]),
            magnitude_in_units(forward_output, OUTPUT_UNITS[key]),
            rtol=1e-3,
        ), key


def test_forward_model_agrees_with_the_published_inverse_model():
    """The forward model inverts run_extended_lengyel_model_with_S_Zeff_and_alphat_correction.

    That model is an independent implementation of the same physics (Body, Kallenbach and Eich,
    NF 2025), built from the modular algorithms rather than the iteration loop of this subpackage,
    so agreement between the two is a cross-check of the port and not a self-consistency check.
    """
    import cfspopcon
    import xarray as xr

    from extended_lengyel.config import setup_impurities

    seed_impurity_species, seed_impurity_weights = setup_impurities(["Nitrogen", "Argon"], [1.0, 0.05])
    target_electron_temp = 2.0 * ureg.eV

    shared_inputs = dict(
        seed_impurity_species=seed_impurity_species,
        seed_impurity_weights=seed_impurity_weights,
        sheath_heat_transmission_factor=8.0,
        ratio_of_upstream_to_average_poloidal_field=4.0 / 3.0,
        separatrix_electron_density=3.3e19 * ureg.m**-3,
        power_crossing_separatrix=5.5 * ureg.MW,
        fraction_of_P_SOL_to_divertor=2.0 / 3.0,
        divertor_broadening_factor=3.0,
        plasma_current=1.0 * ureg.MA,
        magnetic_field_on_axis=2.5 * ureg.T,
        major_radius=1.65 * ureg.m,
        minor_radius=0.5 * ureg.m,
        parallel_connection_length=20.0 * ureg.m,
        divertor_parallel_length=5.0 * ureg.m,
        elongation_psi95=1.6,
        triangularity_psi95=0.3,
        target_angle_of_incidence=3.0 * ureg.degree,
    )

    inverse_algorithm = cfspopcon.CompositeAlgorithm.from_list(
        [
            "calc_magnetic_field_and_safety_factor",
            "calc_fieldline_pitch_at_omp",
            "set_radas_dir",
            "read_atomic_data",
            "build_CzLINT_for_seed_impurities",
            "calc_kappa_e0",
            "build_mean_charge_for_seed_impurities",
            "calc_momentum_loss_from_cc_fit",
            "calc_power_loss_from_cc_fit",
            "calc_electron_temp_from_cc_fit",
            "run_extended_lengyel_model_with_S_Zeff_and_alphat_correction",
        ]
    )
    inverse = inverse_algorithm.update_dataset(
        xr.Dataset(
            data_vars=dict(ion_mass=2.0 * ureg.amu, target_electron_temp=target_electron_temp, **shared_inputs)
        )
    )

    forward_algorithm = cfspopcon.CompositeAlgorithm.from_list(
        [
            "set_radas_dir",
            "read_atomic_data",
            "build_CzLINT_for_seed_impurities",
            "build_mean_charge_for_seed_impurities",
            "run_forward_extended_lengyel_model",
        ]
    )
    forward = forward_algorithm.update_dataset(
        xr.Dataset(
            data_vars=dict(
                average_ion_mass=2.0 * ureg.amu, impurity_fraction=inverse["impurity_fraction"], **shared_inputs
            )
        )
    )

    assert forward["converged"].item()
    assert np.isclose(
        magnitude_in_units(forward["target_electron_temp"], ureg.eV),
        magnitude_in_units(target_electron_temp, ureg.eV),
        rtol=1e-2,
    )
    for key in ("alpha_t", "q_parallel", "separatrix_z_effective"):
        assert np.isclose(
            magnitude_in_units(forward[key], OUTPUT_UNITS[key]), magnitude_in_units(inverse[key], OUTPUT_UNITS[key]), rtol=1e-3
        ), key


def test_forward_model_is_registered_as_an_algorithm():
    """Both models are usable from a CompositeAlgorithm, like the other models in this package."""
    import cfspopcon

    setup = [
        "calc_magnetic_field_and_safety_factor",
        "set_radas_dir",
        "read_atomic_data",
        "build_CzLINT_for_seed_impurities",
        "build_mean_charge_for_seed_impurities",
    ]

    forward = cfspopcon.CompositeAlgorithm.from_list([*setup, "run_forward_extended_lengyel_model"])
    assert "impurity_fraction" in forward.input_keys
    assert "target_electron_temp" in forward.return_keys
    assert "converged" in forward.return_keys

    inverse = cfspopcon.CompositeAlgorithm.from_list([*setup, "run_inverse_extended_lengyel_model"])
    assert "target_electron_temp" in inverse.input_keys
    assert "impurity_fraction" in inverse.return_keys
    assert "converged" in inverse.return_keys
