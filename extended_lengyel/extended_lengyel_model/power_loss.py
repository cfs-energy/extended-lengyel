"""Compute the power loss in the SOL."""

from cfspopcon import Algorithm
from cfspopcon.formulas.scrape_off_layer.two_point_model.required_power_loss_fraction import calc_required_SOL_power_loss_fraction
from cfspopcon.formulas.scrape_off_layer.two_point_model.target_electron_temp import (
    calc_f_other_target_electron_temp,
    calc_target_electron_temp_basic,
)


@Algorithm.register_algorithm(return_keys=["SOL_power_loss_fraction"])
def calc_required_power_loss_fraction(
    target_electron_temp,
    q_parallel,
    separatrix_total_pressure,
    ion_mass,
    sheath_heat_transmission_factor,
    SOL_momentum_loss_fraction,
    target_ratio_of_ion_to_electron_temp=1.0,
    target_ratio_of_electron_to_ion_density=1.0,
    target_mach_number=1.0,
    toroidal_flux_expansion=1.0,
):
    """Calculate the amount of power which must be radiated in the SOL, to achieve the desired target electron temperature."""
    target_electron_temp_basic = calc_target_electron_temp_basic(
        average_ion_mass=ion_mass,
        q_parallel=q_parallel,
        upstream_total_pressure=separatrix_total_pressure,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )
    f_other_target_electron_temp = calc_f_other_target_electron_temp(
        target_ratio_of_ion_to_electron_temp=target_ratio_of_ion_to_electron_temp,
        target_ratio_of_electron_to_ion_density=target_ratio_of_electron_to_ion_density,
        target_mach_number=target_mach_number,
        toroidal_flux_expansion=toroidal_flux_expansion,
    )
    SOL_power_loss_fraction = calc_required_SOL_power_loss_fraction(
        target_electron_temp_basic=target_electron_temp_basic,
        f_other_target_electron_temp=f_other_target_electron_temp,
        SOL_momentum_loss_fraction=SOL_momentum_loss_fraction,
        required_target_electron_temp=target_electron_temp,
    )
    return SOL_power_loss_fraction


@Algorithm.register_algorithm(return_keys=["parallel_heat_flux_at_target"])
def calc_parallel_heat_flux_at_target_from_power_loss_fraction(SOL_power_loss_fraction, q_parallel):
    """Compute the parallel heat flux reaching the target from the power loss."""
    return q_parallel * (1.0 - SOL_power_loss_fraction)
