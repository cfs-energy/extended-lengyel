"""Run the extended Lengyel model with S and self-consistent Zeff."""

import numpy as np
import xarray as xr
from cfspopcon import Algorithm, CompositeAlgorithm
from typing import Optional

from cfspopcon.unit_handling import ureg
from cfspopcon.formulas.metrics import calc_alpha_t
from cfspopcon.formulas.scrape_off_layer.heat_flux_density import calc_parallel_heat_flux_density
from cfspopcon.formulas.separatrix_conditions.separatrix_operational_space.shared import calc_lambda_q_Eich2020H
from cfspopcon.formulas.metrics.larmor_radius import calc_larmor_radius

from .Lengyel_model_extended_S_Zeff import run_extended_lengyel_model_with_S_and_Zeff_correction
from .Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator, calc_z_effective
from ..xr_helpers import item


@Algorithm.register_algorithm(
    return_keys=[
        "impurity_fraction",
        "radiated_fraction_above_xpt",
        "divertor_z_effective",
        "separatrix_z_effective",
        "divertor_entrance_electron_temp",
        "separatrix_electron_temp",
        "separatrix_total_pressure",
        "SOL_power_loss_fraction",
        "parallel_heat_flux_at_target",
        "parallel_heat_flux_at_cc_interface",
        "alpha_t",
        "q_parallel",
        "lambda_q",
    ]
)
def run_extended_lengyel_model_with_S_Zeff_and_alphat_correction(
    target_electron_temp,
    separatrix_electron_density,
    divertor_broadening_factor,
    parallel_connection_length,
    divertor_parallel_length,
    kappa_e0,
    electron_temp_at_cc_interface,
    SOL_momentum_loss_fraction,
    SOL_power_loss_fraction_in_convection_layer,
    ion_mass,
    sheath_heat_transmission_factor,
    CzLINT_for_seed_impurities,
    mean_charge_for_seed_impurities,
    fraction_of_P_SOL_to_divertor,
    power_crossing_separatrix,
    major_radius,
    minor_radius,
    fieldline_pitch_at_omp,
    cylindrical_safety_factor,
    separatrix_average_poloidal_field,
    ratio_of_upstream_to_average_poloidal_field,
    CzLINT_for_fixed_impurities: Optional[CzLINT_integrator] = None,
    mean_charge_for_fixed_impurities: Optional[Mean_charge_interpolator] = None,
    iterations_for_Lengyel_model: int = 5,
    iterations_for_alphat: int = 5,
    mask_invalid_results: bool = True,
):
    """Calculate the impurity fraction required to radiate a given fraction of the power in the scrape-off-layer, iterating to find a consistent Zeff."""
    if CzLINT_for_fixed_impurities is None:
        CzLINT_for_fixed_impurities = CzLINT_integrator.empty()
    if mean_charge_for_fixed_impurities is None:
        mean_charge_for_fixed_impurities = Mean_charge_interpolator.empty()

    f_share = (1.0 - 1.0 / np.e) * fraction_of_P_SOL_to_divertor

    separatrix_electron_temp = 100.0 * ureg.eV
    alpha_t = 0.0

    for _ in range(item(iterations_for_alphat)):
        separatrix_average_poloidal_sound_larmor_radius = calc_larmor_radius(
            species_temperature=separatrix_electron_temp,
            magnetic_field_strength=separatrix_average_poloidal_field,
            species_mass=ion_mass,
        )
        separatrix_average_lambda_q = calc_lambda_q_Eich2020H(alpha_t, separatrix_average_poloidal_sound_larmor_radius)
        ratio_of_upstream_to_average_lambda_q = ratio_of_upstream_to_average_poloidal_field * (major_radius + minor_radius) / major_radius
        lambda_q_outboard_midplane = separatrix_average_lambda_q / ratio_of_upstream_to_average_lambda_q

        q_parallel = calc_parallel_heat_flux_density(
            power_crossing_separatrix=power_crossing_separatrix,
            fraction_of_P_SOL_to_divertor=f_share,
            major_radius=major_radius,
            minor_radius=minor_radius,
            lambda_q=lambda_q_outboard_midplane,
            fieldline_pitch_at_omp=fieldline_pitch_at_omp,
        )

        (
            c_z,
            radiated_fraction_above_xpt,
            divertor_z_effective,
            divertor_entrance_electron_temp,
            separatrix_electron_temp,
            separatrix_total_pressure,
            SOL_power_loss_fraction,
            parallel_heat_flux_at_target,
            parallel_heat_flux_at_cc_interface,
        ) = run_extended_lengyel_model_with_S_and_Zeff_correction(
            target_electron_temp=target_electron_temp,
            separatrix_electron_density=separatrix_electron_density,
            q_parallel=q_parallel,
            divertor_broadening_factor=divertor_broadening_factor,
            parallel_connection_length=parallel_connection_length,
            divertor_parallel_length=divertor_parallel_length,
            kappa_e0=kappa_e0,
            electron_temp_at_cc_interface=electron_temp_at_cc_interface,
            SOL_momentum_loss_fraction=SOL_momentum_loss_fraction,
            SOL_power_loss_fraction_in_convection_layer=SOL_power_loss_fraction_in_convection_layer,
            ion_mass=ion_mass,
            sheath_heat_transmission_factor=sheath_heat_transmission_factor,
            CzLINT_for_seed_impurities=CzLINT_for_seed_impurities,
            mean_charge_for_seed_impurities=mean_charge_for_seed_impurities,
            CzLINT_for_fixed_impurities=CzLINT_for_fixed_impurities,
            mean_charge_for_fixed_impurities=mean_charge_for_fixed_impurities,
            iterations_for_Lengyel_model=iterations_for_Lengyel_model,
            mask_invalid_results=False,
        )

        # Use the separatrix electron temperature to calculate Z-eff for alpha-t
        separatrix_z_effective = calc_z_effective(
            separatrix_electron_temp,
            c_z,
            mean_charge_for_seed_impurities,
            mean_charge_for_fixed_impurities,
            CzLINT_for_seed_impurities,
            CzLINT_for_fixed_impurities,
        )

        alpha_t = calc_alpha_t(
            separatrix_electron_density=separatrix_electron_density,
            separatrix_electron_temp=separatrix_electron_temp,
            cylindrical_safety_factor=cylindrical_safety_factor,
            major_radius=major_radius,
            average_ion_mass=ion_mass,
            z_effective=separatrix_z_effective,
            mean_ion_charge_state=1.0,
        )

        alpha_t = np.maximum(alpha_t, 0.0)

    if mask_invalid_results:
        mask = c_z > 0.0
        c_z = xr.where(mask, c_z, np.nan)
        divertor_z_effective = xr.where(mask, divertor_z_effective, np.nan)

    return (
        c_z,
        radiated_fraction_above_xpt,
        divertor_z_effective,
        separatrix_z_effective,
        divertor_entrance_electron_temp,
        separatrix_electron_temp,
        separatrix_total_pressure,
        SOL_power_loss_fraction,
        parallel_heat_flux_at_target,
        parallel_heat_flux_at_cc_interface,
        alpha_t,
        q_parallel,
        lambda_q_outboard_midplane,
    )


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "read_atomic_data",
            "build_CzLINT_for_seed_impurities",
            "build_mean_charge_for_seed_impurities",
            "calc_kappa_e0",
            "calc_momentum_loss_from_cc_fit",
            "calc_power_loss_from_cc_fit",
            "calc_electron_temp_from_cc_fit",
            "run_extended_lengyel_model_with_S_Zeff_and_alphat_correction",
        ]
    ],
    name="extended_lengyel_model_with_S_Zeff_and_alphat_correction",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "calc_sound_speed_at_target",
            "calc_target_density",
            "calc_flux_density_to_pascals_factor",
            "calc_parallel_to_perp_factor",
            "calc_ion_flux_to_target",
            "calc_divertor_neutral_pressure",
            "calc_radiative_efficiency",
            "calc_qdet_ext_7a",
            "calc_qdet_ext_7b",
            "calc_heat_flux_perp_to_target",
        ]
    ],
    name="compare_alphat_lengyel_model_to_kallenbach_scaling",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
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
            "calc_sound_speed_at_target",
            "calc_target_density",
            "calc_flux_density_to_pascals_factor",
            "calc_parallel_to_perp_factor",
            "calc_ion_flux_to_target",
            "calc_divertor_neutral_pressure",
            "calc_heat_flux_perp_to_target"
        ]
    ],
    name="extended_lengyel_for_experiment_inputs",
    register=True,
)
