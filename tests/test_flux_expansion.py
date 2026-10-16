"""Tests for the magnetic-flux-expansion (t_B) treatment (Moulton 2021).

At the default field ratios the t_B term and the conduction-ladder factor are inert.
These tests check the new behaviour against the spatial Lengyel integration, which
captures t_B by integrating the conservative field-line energy equation.
"""

import numpy as np
import pytest
from cfspopcon.named_options import AtomicSpecies

from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
from extended_lengyel.iterative_solver import physics
from extended_lengyel.iterative_solver.constants import MOMENTUM_LOSS_FIT, amu_to_kg, elementary_charge
from extended_lengyel.iterative_solver.solver import ModelInputs, run_extended_lengyel_solver
from extended_lengyel.iterative_solver.spatial_lengyel import run_spatial_lengyel_model
from extended_lengyel.mavrin_data import read_mavrin_data

N_CC, T_CC, Q_CC, B_Q = 3.0e20, 6.0, 5.0e7, 3.0
S_CC, L_DIV, L_PAR = 0.0, 5.0, 20.0


@pytest.fixture(autouse=True)
def raise_on_float_errors():
    """Make silent over/underflow fail this module without leaking into other tests."""
    with np.errstate(over="raise", under="raise"):
        yield


def _xpt_ratio(field_ratio: float) -> float:
    """X-point field ratio for a globally constant field gradient."""
    return 1.0 + (field_ratio - 1.0) * (L_PAR - L_DIV) / (L_PAR - S_CC)


def _tB_rtol(field_ratio: float) -> float:
    """Tolerance for calc_tB versus the spatial reference, growing with flux expansion."""
    return 0.02 + 0.04 * (field_ratio - 1.0)


def build_model_inputs(**overrides: object) -> ModelInputs:
    """Build solver inputs from the shared base magnitudes plus overrides."""
    atomic_data = read_mavrin_data()
    kwargs = dict(
        power_crossing_separatrix=5.5,
        divertor_broadening_factor=3.0,
        magnetic_field_on_axis=2.5,
        plasma_current=1.0,
        parallel_connection_length=20.0,
        divertor_parallel_length=5.0,
        major_radius=1.65,
        minor_radius=0.5,
        elongation_psi95=1.6,
        triangularity_psi95=0.3,
        target_angle_of_incidence=3.0,
        fraction_of_P_SOL_to_divertor=2.0 / 3.0,
        sheath_heat_transmission_factor=8.0,
        iterations=100,
        average_ion_mass=2.0,
        ratio_of_upstream_to_average_poloidal_field=4.0 / 3.0,
        wall_temperature=300.0,
        SOL_conduction_fraction=1.0,
        ratio_of_molecular_to_ion_mass=2.0,
        separatrix_mach_number=0.0,
        separatrix_ratio_of_ion_to_electron_temp=1.0,
        separatrix_ratio_of_electron_to_ion_density=1.0,
        target_ratio_of_ion_to_electron_temp=1.0,
        target_ratio_of_electron_to_ion_density=1.0,
        target_mach_number=1.0,
        flux_expansion_Bt_over_Bu=1.0,
        CzLINT_for_seed_impurities=CzLINT_integrator.from_list(
            [AtomicSpecies.Nitrogen, AtomicSpecies.Argon], [1.0, 0.05], atomic_data
        ),
        mean_charge_for_seed_impurities=Mean_charge_interpolator.from_list(
            [AtomicSpecies.Nitrogen, AtomicSpecies.Argon], atomic_data
        ),
        CzLINT_for_fixed_impurities=CzLINT_integrator.from_list([AtomicSpecies.Helium], [1.0e-2], atomic_data),
        mean_charge_for_fixed_impurities=Mean_charge_interpolator.from_list([AtomicSpecies.Helium], atomic_data),
    )
    kwargs.update(overrides)
    return ModelInputs.from_wrapper_args(**kwargs)


def _spatial(field_ratio: float, c_z: float, broadening: float = B_Q):
    """Run the spatial reference at the fixed cc-state with single-nitrogen seeding."""
    atomic_data = read_mavrin_data()
    seed = CzLINT_integrator.from_list([AtomicSpecies.Nitrogen], [1.0], atomic_data)
    kappa = physics.calc_heat_conductivity_factor(1.0)
    return run_spatial_lengyel_model(
        electron_density_at_cc_interface=N_CC,
        electron_temp_at_cc_interface=T_CC,
        parallel_heat_flux_at_cc_interface=Q_CC,
        impurity_fraction=c_z,
        CzLINT_for_seed_impurities=seed,
        CzLINT_for_fixed_impurities=CzLINT_integrator.empty(),
        divertor_heat_conductivity_factor=kappa,
        upstream_heat_conductivity_factor=kappa,
        divertor_broadening_factor=broadening,
        s_parallel_at_cc_interface=S_CC,
        divertor_parallel_length=L_DIV,
        parallel_connection_length=L_PAR,
        flux_expansion_Bt_over_Bu=field_ratio,
        flux_expansion_Bxpt_over_Bu=_xpt_ratio(field_ratio),
    )


def test_calc_tB_inert_at_default() -> None:
    """t_B is exactly zero when there is no flux expansion."""
    assert physics.calc_tB(4.0e8, 2.0e4, 1.0, 1e-3, 1.0, 1.0, 1.0) == 0.0


def test_spatial_no_radiation_analytic_limit() -> None:
    """With no radiation, q_u/q_cc = B_u/B_t and t_B = 0.5 q_u^2 ((B_t/B_u)^2 - 1)."""
    for field_ratio in (1.25, 1.5, 2.0, 0.7):
        result = _spatial(field_ratio, c_z=0.0, broadening=1.0)
        assert result.converged
        assert result.q_parallel / Q_CC == pytest.approx(1.0 / field_ratio, rel=1e-6)
        assert result.tB_cc_div + result.tB_div_u == pytest.approx(
            0.5 * result.q_parallel**2 * (field_ratio**2 - 1.0), rel=1e-6
        )


def test_ladder_correction_matches_spatial() -> None:
    """The conduction-ladder flux-expansion factor reproduces the spatial temperatures with radiation off."""
    kappa = physics.calc_heat_conductivity_factor(1.0)
    for field_ratio in (1.25, 1.5, 2.0):
        result = _spatial(field_ratio, c_z=0.0)
        inputs = build_model_inputs(
            impurity_fraction=0.0,
            separatrix_electron_density=N_CC,
            flux_expansion_Bt_over_Bu=field_ratio,
            flux_expansion_Bxpt_over_Bu=_xpt_ratio(field_ratio),
            divertor_broadening_factor=B_Q,
            SOL_conduction_fraction=1.0,
        )
        divertor_temp, upstream_temp = physics.calc_temperature_ladder(T_CC, result.q_parallel, kappa, kappa, inputs)
        assert divertor_temp == pytest.approx(result.divertor_entrance_electron_temp, rel=1e-6)
        assert upstream_temp == pytest.approx(result.separatrix_electron_temp, rel=1e-6)


def test_calc_tB_matches_spatial_with_radiation() -> None:
    """calc_tB matches the spatial numerator t_B, with a tolerance that grows with B_t/B_u."""
    kappa = physics.calc_heat_conductivity_factor(1.0)
    inputs = build_model_inputs(impurity_fraction=0.03, separatrix_electron_density=N_CC)
    k_base = 2.0 * (N_CC / 1e20 * T_CC) ** 2
    for impurity_fraction in (0.02, 0.05):
        for field_ratio in (1.25, 1.5, 2.0):
            result = _spatial(field_ratio, c_z=impurity_fraction)
            integrals = physics.calc_lengyel_integrals(
                T_CC, result.divertor_entrance_electron_temp, result.separatrix_electron_temp, inputs
            )
            t_b = physics.calc_tB(
                result.q_parallel,
                k_base,
                kappa,
                impurity_fraction * integrals.Ls_div_u,
                1.0,
                field_ratio,
                _xpt_ratio(field_ratio),
            )
            assert t_b == pytest.approx(B_Q**2 * result.tB_cc_div + result.tB_div_u, rel=_tB_rtol(field_ratio))


def test_full_models_physical_flux_expansion_trend() -> None:
    """Forward target heat flux and inverse impurity fraction both rise with B_t/B_u."""
    target_heat_flux = [
        run_extended_lengyel_solver(
            build_model_inputs(
                impurity_fraction=0.03, separatrix_electron_density=3.0e19, flux_expansion_Bt_over_Bu=field_ratio
            ),
            mode="forward",
            pdiv=False,
        ).heat_flux_perp_to_target
        for field_ratio in (0.8, 1.0, 1.25, 1.5)
    ]
    impurity_fraction = [
        run_extended_lengyel_solver(
            build_model_inputs(
                target_electron_temp=5.0, separatrix_electron_density=3.0e19, flux_expansion_Bt_over_Bu=field_ratio
            ),
            mode="inverse",
            pdiv=False,
        ).impurity_fraction
        for field_ratio in (0.6, 0.8, 1.0, 1.25, 1.5)
    ]
    assert np.all(np.diff(target_heat_flux) > 0)
    assert np.all(np.diff(impurity_fraction) > 0)
    detached = run_extended_lengyel_solver(
        build_model_inputs(impurity_fraction=0.03, separatrix_electron_density=3.0e19, flux_expansion_Bt_over_Bu=0.5),
        mode="forward",
        pdiv=False,
    )
    assert not detached.converged


def test_stangeby_2pm_self_consistency_with_flux_expansion() -> None:
    """The target state satisfies Stangeby 2018 eqs 12 and 15-17 once flux expansion is included."""
    for impurity_fraction in (0.0, 0.03):
        for field_ratio in (1.0, 0.7, 1.5):
            inputs = build_model_inputs(
                impurity_fraction=impurity_fraction,
                separatrix_electron_density=3.0e19,
                flux_expansion_Bt_over_Bu=field_ratio,
            )
            result = run_extended_lengyel_solver(inputs, mode="forward", pdiv=False)
            assert result.converged
            geometry = physics.calc_geometry(inputs)
            sheath_gamma = inputs.sheath_heat_transmission_factor
            ion_mass = inputs.average_ion_mass * amu_to_kg
            q_upstream = result.q_parallel
            target_temp = result.target_electron_temp
            ion_flux = result.parallel_ion_flux_to_target
            total_pressure = physics.calc_separatrix_total_pressure(
                inputs.separatrix_electron_density, result.separatrix_electron_temp, inputs
            )
            q_target = result.heat_flux_perp_to_target / geometry.parallel_to_perp_factor
            cooling = 1.0 - q_target / field_ratio / q_upstream
            momentum_loss = physics.temperature_fit_function(target_temp, **MOMENTUM_LOSS_FIT)
            target_density = ion_flux / np.sqrt(2.0 * target_temp * elementary_charge / ion_mass)

            assert q_target == pytest.approx((1.0 - cooling) * q_upstream * field_ratio, rel=1e-9)
            assert target_temp == pytest.approx(
                8.0
                * ion_mass
                / (elementary_charge * sheath_gamma**2)
                * q_upstream**2
                / total_pressure**2
                * (1 - cooling) ** 2
                / (1 - momentum_loss) ** 2
                * field_ratio**2,
                rel=1e-6,
            )
            assert target_density == pytest.approx(
                sheath_gamma**2
                / (32.0 * ion_mass)
                * total_pressure**3
                / q_upstream**2
                * (1 - momentum_loss) ** 3
                / (1 - cooling) ** 2
                / field_ratio**2,
                rel=1e-6,
            )
            assert ion_flux == pytest.approx(
                sheath_gamma
                / (8.0 * ion_mass)
                * total_pressure**2
                / q_upstream
                * (1 - momentum_loss) ** 2
                / (1 - cooling)
                / field_ratio,
                rel=1e-6,
            )
