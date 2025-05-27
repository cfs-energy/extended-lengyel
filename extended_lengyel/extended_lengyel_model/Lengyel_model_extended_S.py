"""Run the extended Lengyel model with S corrections only."""

import numpy as np
import xarray as xr
from cfspopcon import Algorithm, CompositeAlgorithm

from .Lengyel_model_core import item


@Algorithm.register_algorithm(return_keys=["impurity_fraction", "radiated_fraction_above_xpt", "radiated_fraction_below_xpt"])
def run_extended_lengyel_model_with_S_correction(
    q_parallel,
    divertor_broadening_factor,
    kappa_e0,
    kappa_z,
    parallel_heat_flux_at_cc_interface,
    separatrix_electron_density,
    separatrix_electron_temp,
    electron_temp_at_cc_interface,
    divertor_entrance_electron_temp,
    L_int_integrator,
    mask_invalid_results: bool = True,
):
    """Calculate the impurity fraction required to radiate a given fraction of the power in the scrape-off-layer."""
    LINT_cc_div = item(L_int_integrator)(electron_temp_at_cc_interface, divertor_entrance_electron_temp)
    LINT_div_u = item(L_int_integrator)(divertor_entrance_electron_temp, separatrix_electron_temp)
    LINT_cc_u = item(L_int_integrator)(electron_temp_at_cc_interface, separatrix_electron_temp)

    radiated_fraction_above_xpt = 1.0 - np.sqrt(
        (LINT_cc_div + LINT_div_u * (parallel_heat_flux_at_cc_interface / q_parallel) ** 2)
        / (LINT_cc_div + LINT_div_u / divertor_broadening_factor**2)
    )

    total_radiated_fraction = (1.0 - parallel_heat_flux_at_cc_interface / q_parallel) * divertor_broadening_factor
    radiated_fraction_above_xpt = np.minimum(radiated_fraction_above_xpt, total_radiated_fraction)
    radiated_fraction_above_xpt = np.maximum(radiated_fraction_above_xpt, 0.0)

    qdiv = (1 - radiated_fraction_above_xpt) * q_parallel
    radiated_fraction_below_xpt = 1.0 - parallel_heat_flux_at_cc_interface / qdiv

    kappa = kappa_e0 / kappa_z

    c_z = (
        q_parallel**2
        + ((1 / divertor_broadening_factor) ** 2 - 1) * (1 - radiated_fraction_above_xpt) ** 2 * q_parallel**2
        - parallel_heat_flux_at_cc_interface**2
    ) / (2.0 * kappa * separatrix_electron_density**2 * separatrix_electron_temp**2 * LINT_cc_u)

    if mask_invalid_results:
        c_z = xr.where(c_z < 0.0, np.nan, c_z)

    return c_z, radiated_fraction_above_xpt, radiated_fraction_below_xpt


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "set_radas_dir",
            "read_atomic_data",
            "set_single_impurity_species",
            "build_L_int_integrator",
            "calc_kappa_e0",
            "calc_Goldston_kappa_z",
            "calc_momentum_loss_from_cc_fit",
            "ignore_power_loss_in_convection_layer",
            "ignore_temp_ratio_in_convection_layer",
            "calc_separatrix_electron_temp_with_broadening",
            "calc_separatrix_total_pressure_LG",
            "calc_required_power_loss_fraction",
            "calc_parallel_heat_flux_at_target_from_power_loss_fraction",
            "calc_parallel_heat_flux_from_conv_loss",
            "run_extended_lengyel_model_with_S_correction",
        ]
    ],
    name="extended_lengyel_model_with_S_correction",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "set_radas_dir",
            "read_atomic_data",
            "set_single_impurity_species",
            "build_L_int_integrator",
            "calc_kappa_e0",
            "calc_Goldston_kappa_z",
            "calc_momentum_loss_from_cc_fit",
            "calc_power_loss_from_cc_fit",
            "calc_electron_temp_from_cc_fit",
            "calc_separatrix_electron_temp_with_broadening",
            "calc_separatrix_total_pressure_LG",
            "calc_required_power_loss_fraction",
            "calc_parallel_heat_flux_at_target_from_power_loss_fraction",
            "calc_parallel_heat_flux_from_conv_loss",
            "run_extended_lengyel_model_with_S_correction",
        ]
    ],
    name="extended_lengyel_model_with_S_fconv_correction",
    register=True,
)
