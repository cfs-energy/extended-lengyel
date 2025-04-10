"""Run the extended Lengyel model with S and self-consistent Zeff."""

import numpy as np
import xarray as xr
from cfspopcon import Algorithm, CompositeAlgorithm

from ..initialize import calc_Goldston_kappa_z
from .convective_loss_fits import calc_parallel_heat_flux_from_conv_loss
from .power_loss import calc_parallel_heat_flux_at_target_from_power_loss_fraction, calc_required_power_loss_fraction
from .upstream_temp import calc_separatrix_electron_temp_with_broadening, calc_separatrix_total_pressure_LG

from .Lengyel_model_core import item
from .Lengyel_model_extended_S import run_extended_lengyel_model_with_S_correction


@Algorithm.register_algorithm(
    return_keys=[
        "impurity_fraction",
        "radiated_fraction_above_xpt",
        "radiated_fraction_below_xpt",
        "z_effective",
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
    L_int_integrator,
    mean_charge_interpolator,
    iterations_for_Lengyel_model: int = 5,
    mask_invalid_results: bool = True,
):
    """Calculate the impurity fraction required to radiate a given fraction of the power in the scrape-off-layer, iterating to find a consistent Zeff."""
    z_effective = 1.0

    for _ in range(item(iterations_for_Lengyel_model)):
        kappa_z = calc_Goldston_kappa_z(z_effective)

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

        c_z, radiated_fraction_above_xpt, radiated_fraction_below_xpt = run_extended_lengyel_model_with_S_correction(
            q_parallel=q_parallel,
            divertor_broadening_factor=divertor_broadening_factor,
            kappa_e0=kappa_e0,
            kappa_z=kappa_z,
            parallel_heat_flux_at_cc_interface=parallel_heat_flux_at_cc_interface,
            separatrix_electron_density=separatrix_electron_density,
            separatrix_electron_temp=separatrix_electron_temp,
            electron_temp_at_cc_interface=electron_temp_at_cc_interface,
            divertor_entrance_electron_temp=divertor_entrance_electron_temp,
            L_int_integrator=L_int_integrator,
            mask_invalid_results=False,
        )

        mean_z = item(mean_charge_interpolator)(divertor_entrance_electron_temp)
        z_effective = 1.0 + mean_z * (mean_z - 1.0) * c_z

    if mask_invalid_results:
        mask = c_z > 0.0
        c_z = xr.where(mask, c_z, np.nan)
        z_effective = xr.where(mask, z_effective, np.nan)

    return (
        c_z,
        radiated_fraction_above_xpt,
        radiated_fraction_below_xpt,
        z_effective,
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
            "build_mixed_seeding_L_int_integrator",
            "build_mixed_seeding_mean_charge_interpolator",
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
