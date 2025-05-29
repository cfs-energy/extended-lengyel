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
from .Lengyel_model_core import item, L_int_integrator, Mean_charge_interpolator


@Algorithm.register_algorithm(
    return_keys=[
        "impurity_fraction",
        "radiated_fraction_above_xpt",
        "z_effective",
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
    L_int_integrator,
    mean_charge_interpolator,
    fraction_of_P_SOL_to_divertor,
    power_crossing_separatrix,
    major_radius,
    minor_radius,
    fieldline_pitch_at_omp,
    cylindrical_safety_factor,
    separatrix_average_poloidal_field,
    ratio_of_upstream_to_average_poloidal_field,
    background_cz_L_int= L_int_integrator.empty(),
    background_cz_mean_charge= Mean_charge_interpolator.empty(),
    iterations_for_Lengyel_model: int = 5,
    iterations_for_alphat: int = 5,
    mask_invalid_results: bool = True,
):
    """Calculate the impurity fraction required to radiate a given fraction of the power in the scrape-off-layer, iterating to find a consistent Zeff."""
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
            z_effective,
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
            L_int_integrator=L_int_integrator,
            mean_charge_interpolator=mean_charge_interpolator,
            background_cz_L_int=background_cz_L_int,
            background_cz_mean_charge=background_cz_mean_charge,
            iterations_for_Lengyel_model=iterations_for_Lengyel_model,
            mask_invalid_results=False,
        )

        # Use the separatrix electron temperature to calculate Z-eff for alpha-t
        seed_mean_z = item(mean_charge_interpolator)(separatrix_electron_temp)
        fixed_mean_z = item(background_cz_mean_charge)(separatrix_electron_temp)
        z_effective_upstream = 1.0 + seed_mean_z * (seed_mean_z - 1.0) * c_z + fixed_mean_z * (fixed_mean_z - 1.0)
        z_effective_upstream = np.maximum(z_effective_upstream, 1.0)

        alpha_t = calc_alpha_t(
            separatrix_electron_density=separatrix_electron_density,
            separatrix_electron_temp=separatrix_electron_temp,
            cylindrical_safety_factor=cylindrical_safety_factor,
            major_radius=major_radius,
            average_ion_mass=ion_mass,
            z_effective=z_effective_upstream,
            mean_ion_charge_state=1.0,
        )

        alpha_t = np.maximum(alpha_t, 0.0)

    if mask_invalid_results:
        mask = c_z > 0.0
        c_z = xr.where(mask, c_z, np.nan)
        z_effective = xr.where(mask, z_effective, np.nan)

    return (
        c_z,
        radiated_fraction_above_xpt,
        z_effective,
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
            "build_L_int_integrator",
            "build_mean_charge_interpolator",
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
