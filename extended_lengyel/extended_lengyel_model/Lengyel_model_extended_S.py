"""Run the extended Lengyel model with S corrections only."""

import numpy as np
import xarray as xr
from cfspopcon import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import ureg
from typing import Optional

from .Lengyel_model_core import CzLINT_integrator
from ..xr_helpers import item

@Algorithm.register_algorithm(return_keys=["impurity_fraction", "radiated_fraction_above_xpt"])
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
    CzLINT_for_seed_impurities: CzLINT_integrator,
    CzLINT_for_fixed_impurities: Optional[CzLINT_integrator] = None,
    mask_invalid_results: bool = True,
):
    """Calculate the impurity fraction required to radiate a given fraction of the power in the scrape-off-layer."""
    if CzLINT_for_fixed_impurities is None:
        CzLINT_for_fixed_impurities = CzLINT_integrator.empty()

    # Seed impurities
    Ls_cc_div = item(CzLINT_for_seed_impurities)(electron_temp_at_cc_interface, divertor_entrance_electron_temp)
    Ls_div_u = item(CzLINT_for_seed_impurities)(divertor_entrance_electron_temp, separatrix_electron_temp)
    Ls_cc_u = item(CzLINT_for_seed_impurities)(electron_temp_at_cc_interface, separatrix_electron_temp)

    # Fixed impurities
    Lf_cc_div = item(CzLINT_for_fixed_impurities)(electron_temp_at_cc_interface, divertor_entrance_electron_temp)
    Lf_div_u = item(CzLINT_for_fixed_impurities)(divertor_entrance_electron_temp, separatrix_electron_temp)
    Lf_cc_u = item(CzLINT_for_fixed_impurities)(electron_temp_at_cc_interface, separatrix_electron_temp)

    qu = q_parallel
    qcc = parallel_heat_flux_at_cc_interface
    b = divertor_broadening_factor
    k = 2.0 * (kappa_e0 / kappa_z) * separatrix_electron_density**2 * separatrix_electron_temp**2

    q_div_squared = (
        (Ls_div_u * (qcc**2 + k * Lf_cc_div) + Ls_cc_div * (qu**2 - k * Lf_div_u))
        / (Ls_div_u / b**2  + Ls_cc_div)
    )
    q_div_squared = np.maximum(q_div_squared, 0.0 * ureg.W**2 * ureg.m**-4)
    f_rad_main = 1.0 - np.sqrt(q_div_squared) / qu

    c_z = (
        (qu**2 + (1 / b**2 - 1) * q_div_squared - qcc**2) / (k * Ls_cc_u)
        - Lf_cc_u / Ls_cc_u
    )

    if mask_invalid_results:
        c_z = xr.where(c_z < 0.0, np.nan, c_z)

    return c_z, f_rad_main


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "set_radas_dir",
            "read_atomic_data",
            "set_single_impurity_species",
            "build_CzLINT_for_seed_impurities",
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
            "build_CzLINT_for_seed_impurities",
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
