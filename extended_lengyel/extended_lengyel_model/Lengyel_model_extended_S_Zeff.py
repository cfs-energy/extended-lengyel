"""Run the extended Lengyel model with S and self-consistent Zeff."""

import numpy as np
import xarray as xr
from cfspopcon import Algorithm, CompositeAlgorithm
from typing import Optional

from ..initialize import calc_Goldston_kappa_z
from .convective_loss_fits import calc_parallel_heat_flux_from_conv_loss
from .power_loss import calc_parallel_heat_flux_at_target_from_power_loss_fraction, calc_required_power_loss_fraction
from .upstream_temp import calc_separatrix_electron_temp_with_broadening, calc_separatrix_total_pressure_LG

from .Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator, calc_z_effective
from .Lengyel_model_extended_S import run_extended_lengyel_model_with_S_correction
from ..xr_helpers import item


@Algorithm.register_algorithm(
    return_keys=[
        "impurity_fraction",
        "radiated_fraction_above_xpt",
        "divertor_z_effective",
        "divertor_entrance_electron_temp",
        "separatrix_electron_temp",
        "separatrix_total_pressure",
        "SOL_power_loss_fraction",
        "parallel_heat_flux_at_target",
        "parallel_heat_flux_at_cc_interface",
    ]
)
def run_extended_lengyel_model_with_S_and_Zeff_correction(
    target_electron_temp,
    separatrix_electron_density,
    q_parallel,
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
    CzLINT_for_fixed_impurities: Optional[CzLINT_integrator] = None,
    mean_charge_for_fixed_impurities: Optional[Mean_charge_interpolator] = None,
    iterations_for_Lengyel_model: int = 5,
    mask_invalid_results: bool = True,
):
    """Calculate the impurity fraction required to radiate a given fraction of the power in the scrape-off-layer, iterating to find a consistent Zeff."""
    divertor_z_effective = 1.0
    if CzLINT_for_fixed_impurities is None:
        CzLINT_for_fixed_impurities = CzLINT_integrator.empty()
    if mean_charge_for_fixed_impurities is None:
        mean_charge_for_fixed_impurities = Mean_charge_interpolator.empty()

    for _ in range(item(iterations_for_Lengyel_model)):
        kappa_z = calc_Goldston_kappa_z(divertor_z_effective)

        divertor_entrance_electron_temp, separatrix_electron_temp = calc_separatrix_electron_temp_with_broadening(
            electron_temp_at_cc_interface=electron_temp_at_cc_interface,
            q_parallel=q_parallel,
            divertor_broadening_factor=divertor_broadening_factor,
            parallel_connection_length=parallel_connection_length,
            divertor_parallel_length=divertor_parallel_length,
            kappa_e0=kappa_e0,
            kappa_z=kappa_z,
        )

        separatrix_total_pressure = calc_separatrix_total_pressure_LG(
            separatrix_electron_density=separatrix_electron_density, separatrix_electron_temp=separatrix_electron_temp
        )
        SOL_power_loss_fraction = calc_required_power_loss_fraction(
            target_electron_temp=target_electron_temp,
            q_parallel=q_parallel,
            separatrix_total_pressure=separatrix_total_pressure,
            ion_mass=ion_mass,
            sheath_heat_transmission_factor=sheath_heat_transmission_factor,
            SOL_momentum_loss_fraction=SOL_momentum_loss_fraction,
        )

        parallel_heat_flux_at_target = calc_parallel_heat_flux_at_target_from_power_loss_fraction(SOL_power_loss_fraction, q_parallel)
        parallel_heat_flux_at_cc_interface = calc_parallel_heat_flux_from_conv_loss(
            parallel_heat_flux_at_target, SOL_power_loss_fraction_in_convection_layer
        )

        c_z, radiated_fraction_above_xpt = run_extended_lengyel_model_with_S_correction(
            q_parallel=q_parallel,
            divertor_broadening_factor=divertor_broadening_factor,
            kappa_e0=kappa_e0,
            kappa_z=kappa_z,
            parallel_heat_flux_at_cc_interface=parallel_heat_flux_at_cc_interface,
            separatrix_electron_density=separatrix_electron_density,
            separatrix_electron_temp=separatrix_electron_temp,
            electron_temp_at_cc_interface=electron_temp_at_cc_interface,
            divertor_entrance_electron_temp=divertor_entrance_electron_temp,
            CzLINT_for_seed_impurities=CzLINT_for_seed_impurities,
            CzLINT_for_fixed_impurities=CzLINT_for_fixed_impurities,
            mask_invalid_results=False,
        )

        divertor_z_effective = calc_z_effective(
            divertor_entrance_electron_temp,
            c_z,
            mean_charge_for_seed_impurities,
            mean_charge_for_fixed_impurities,
            CzLINT_for_seed_impurities,
            CzLINT_for_fixed_impurities,
        )

    if mask_invalid_results:
        mask = c_z > 0.0
        c_z = xr.where(mask, c_z, np.nan)
        divertor_z_effective = xr.where(mask, divertor_z_effective, np.nan)

    return (
        c_z,
        radiated_fraction_above_xpt,
        divertor_z_effective,
        divertor_entrance_electron_temp,
        separatrix_electron_temp,
        separatrix_total_pressure,
        SOL_power_loss_fraction,
        parallel_heat_flux_at_target,
        parallel_heat_flux_at_cc_interface,
    )


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "set_radas_dir",
            "read_atomic_data",
            "set_single_impurity_species",
            "build_CzLINT_for_seed_impurities",
            "build_mean_charge_for_seed_impurities",
            "calc_kappa_e0",
            "calc_momentum_loss_from_cc_fit",
            "calc_power_loss_from_cc_fit",
            "calc_electron_temp_from_cc_fit",
            "run_extended_lengyel_model_with_S_and_Zeff_correction",
        ]
    ],
    name="extended_lengyel_model_with_S_fconv_Zeff_correction",
    register=True,
)
