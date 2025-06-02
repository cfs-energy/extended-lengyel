"""Post-process the results of the Kallenbach model, using the results of the main loop."""

import numpy as np
from cfspopcon.algorithm_class import Algorithm
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import magnitude_in_units, ureg
from .xr_helpers import item


@Algorithm.register_algorithm(return_keys=["separatrix_electron_density"])
def calc_upstream_density(electron_density):
    """Calculate the upstream electron density."""
    return electron_density.isel(dim_s_parallel=-1)


@Algorithm.register_algorithm(return_keys=["separatrix_electron_temp"])
def calc_upstream_temp(electron_temp):
    """Calculate the upstream electron temperature."""
    return electron_temp.isel(dim_s_parallel=-1)


@Algorithm.register_algorithm(return_keys=["parallel_heat_flux"])
def calc_q_total(parallel_conductive_heat_flux, parallel_convective_heat_flux):
    """Calculate the total parallel heat flux."""
    return parallel_conductive_heat_flux + parallel_convective_heat_flux


@Algorithm.register_algorithm(return_keys=["q_parallel"])
def calc_upstream_q_total(parallel_heat_flux):
    """Calculate the total parallel heat flux at the OMP."""
    return parallel_heat_flux.isel(dim_s_parallel=-1)


@Algorithm.register_algorithm(return_keys=["parallel_heat_flux_at_target"])
def calc_target_q_total(parallel_heat_flux):
    """Calculate the total parallel heat flux at the target."""
    return parallel_heat_flux.isel(dim_s_parallel=0)


@Algorithm.register_algorithm(return_keys=["SOL_momentum_loss_fraction"])
def calc_SOL_momentum_loss_fraction(
    electron_density_at_target, target_electron_temp, separatrix_electron_density, separatrix_electron_temp
):
    """Calculate the momentum loss factor, defined such that target_dynamic_pressure = (1 - fmom) * upstream_dynamic_pressure."""
    return 1.0 - (2.0 * electron_density_at_target * target_electron_temp) / (separatrix_electron_density * separatrix_electron_temp)


@Algorithm.register_algorithm(return_keys=["SOL_power_loss_fraction"])
def calc_SOL_power_loss_fraction(parallel_heat_flux_at_target, q_parallel):
    """Calculate the power loss factor, defined such that target_heat_flux = (1 - fpow) * upstream_heat_flux."""
    return 1.0 - parallel_heat_flux_at_target / q_parallel


@Algorithm.register_algorithm(return_keys=["power_crossing_separatrix"])
def calc_power_crossing_separatrix_from_heat_flux_in_flux_tube(
    q_parallel, flux_tube_cross_section_area_out_of_divertor, fraction_of_P_SOL_to_divertor
):
    """Calculate the power crossing the separatrix from the heat flux in the modelled flux tube."""
    power_towards_outer_divertor_in_flux_tube = q_parallel * flux_tube_cross_section_area_out_of_divertor
    fraction_of_power_in_first_lambda_q = 1.0 - 1.0 / np.e
    fraction_of_power_in_flux_tube = fraction_of_P_SOL_to_divertor * fraction_of_power_in_first_lambda_q

    power_crossing_separatrix = (1.0 / fraction_of_power_in_flux_tube) * power_towards_outer_divertor_in_flux_tube
    return power_crossing_separatrix


@Algorithm.register_algorithm(return_keys=["radiative_efficiency"])
def calc_radiative_efficiency(
    impurity_species: AtomicSpecies,
):
    """Calculate the "f_Z" radiative efficiency of the edge impurity species, from table 1 in Kallenbach et al., 2016."""
    radiative_efficiency = {
        AtomicSpecies.Hydrogen: 1.0,
        AtomicSpecies.Nitrogen: 18.0,
        AtomicSpecies.Neon: 45.0,
        AtomicSpecies.Argon: 90.0,
    }

    impurity_species = item(impurity_species)

    return radiative_efficiency[impurity_species]


@Algorithm.register_algorithm(return_keys=["Psep/R/lambda_int", "p0(1+fz*cz)/1.3"])
def calc_qdet_ext_7a(
    power_crossing_separatrix,
    major_radius,
    radiative_efficiency,
    neutral_pressure_in_divertor,
    impurity_fraction,
    lambda_int,
):
    """Calculate the x and y axes of figure 7a in Kallenbach et al., 2016., adding an additional correction for lambda-int variations."""
    Psep_over_R = magnitude_in_units(power_crossing_separatrix / major_radius, ureg.MW / ureg.m)
    p0 = magnitude_in_units(neutral_pressure_in_divertor, ureg.Pa)
    lambda_int_norm = magnitude_in_units(lambda_int / (5.0 * ureg.mm), ureg.dimensionless)

    term1 = Psep_over_R / lambda_int_norm
    term2 = p0 * (1.0 + radiative_efficiency * impurity_fraction) / 1.3
    return term1, term2


@Algorithm.register_algorithm(return_keys=["cz*fz", "1.3Psep/R/p0/lambda_int - 1"])
def calc_qdet_ext_7b(
    power_crossing_separatrix,
    major_radius,
    radiative_efficiency,
    neutral_pressure_in_divertor,
    impurity_fraction,
    lambda_int,
):
    """Calculate the x and y axes of figure 7b in Kallenbach et al., 2016., adding an additional correction for lambda-int variations."""
    Psep_over_R = magnitude_in_units(power_crossing_separatrix / major_radius, ureg.MW / ureg.m)
    p0 = magnitude_in_units(neutral_pressure_in_divertor, ureg.Pa)
    lambda_int_norm = magnitude_in_units(lambda_int / (5.0 * ureg.mm), ureg.dimensionless)

    term1 = radiative_efficiency * impurity_fraction
    term2 = 1.3 * Psep_over_R / p0 / lambda_int_norm - 1
    return term1, term2


@Algorithm.register_algorithm(return_keys=["ion_temp"])
def calc_ion_temp(electron_temp):
    """We are assuming Te = Ti."""
    return electron_temp


@Algorithm.register_algorithm(return_keys=["ion_density"])
def calc_ion_density(electron_density):
    """We are assuming ne = ni."""
    return electron_density


@Algorithm.register_algorithm(return_keys=["static_pressure"])
def calc_static_pressure(electron_density, electron_temp, ion_temp):
    """Calculate the total static pressure."""
    return electron_density * (electron_temp + ion_temp)


@Algorithm.register_algorithm(return_keys=["dynamic_pressure"])
def calc_dynamic_pressure(ion_mass, electron_density, ion_velocity):
    """Calculate the total dynamic pressure."""
    return ion_mass * electron_density * ion_velocity**2


@Algorithm.register_algorithm(return_keys=["total_pressure"])
def calc_total_pressure(static_pressure, dynamic_pressure):
    """Calculate the total static and dynamic pressure."""
    return static_pressure + dynamic_pressure


@Algorithm.register_algorithm(return_keys=["separatrix_total_pressure"])
def calc_separatrix_total_pressure(total_pressure):
    """Calculate the total pressure at the OMP."""
    return total_pressure.isel(dim_s_parallel=-1)


@Algorithm.register_algorithm(return_keys=["sound_speed"])
def calc_sound_speed(electron_temp, ion_temp, ion_mass):
    """Calculate the local sound speed."""
    return np.sqrt((electron_temp + ion_temp) / ion_mass)


@Algorithm.register_algorithm(return_keys=["mach_number"])
def calc_mach_number(ion_velocity, sound_speed):
    """Calculate the local Mach number."""
    return ion_velocity / sound_speed


@Algorithm.register_algorithm(return_keys=["charge_exchange_power_loss"])
def calc_charge_exchange_power_loss_density(deuterium_adas_data, electron_density, electron_temp, neutral_density, ion_density):
    """Calculate the charge_exchange_power_loss."""
    deuterium_adas_data = item(deuterium_adas_data)
    charge_exchange_rate = deuterium_adas_data.charge_exchange_rate.eval(electron_density, electron_temp) * neutral_density * ion_density
    return electron_temp * charge_exchange_rate


@Algorithm.register_algorithm(return_keys=["ionization_power_loss"])
def calc_ionization_power_loss_density(
    hydrogen_effective_ionization_energy, deuterium_adas_data, electron_density, electron_temp, neutral_density
):
    """Calculate the ionization_power_loss."""
    deuterium_adas_data = item(deuterium_adas_data)
    ionization_rate = deuterium_adas_data.ionization_rate.eval(electron_density, electron_temp) * electron_density * neutral_density
    return hydrogen_effective_ionization_energy * ionization_rate


@Algorithm.register_algorithm(return_keys=["hydrogen_radiated_power"])
def calc_hydrogen_radiated_power_density(deuterium_adas_data, electron_density, electron_temp, ion_density):
    """Calculate the hydrogen_radiated_power."""
    deuterium_adas_data = item(deuterium_adas_data)
    return deuterium_adas_data.radiative_power_coeff.eval(electron_density, electron_temp) * ion_density * electron_density


@Algorithm.register_algorithm(return_keys=["impurity_radiated_power"])
def calc_impurity_radiated_power_density(impurity_adas_data, electron_density, electron_temp, impurity_fraction):
    """Calculate the impurity_radiated_power."""
    impurity_adas_data = item(impurity_adas_data)
    return impurity_adas_data.radiative_power_coeff.eval(electron_density, electron_temp) * impurity_fraction * electron_density**2


@Algorithm.register_algorithm(return_keys=["parallel_convective_heat_flux", "parallel_conductive_heat_flux"])
def calc_convected_and_conducted_heat_flux(parallel_heat_flux, electron_density, electron_temp, ion_mass, ion_velocity):
    """Calculate the convected and conducted heat fluxes."""
    parallel_convective_heat_flux = (
        -(5.0 * electron_density * electron_temp + 0.5 * ion_mass * electron_density * ion_velocity**2) * ion_velocity
    )
    parallel_conductive_heat_flux = parallel_heat_flux - parallel_convective_heat_flux
    return parallel_convective_heat_flux, parallel_conductive_heat_flux


@Algorithm.register_algorithm(
    return_keys=[
        "s_parallel_at_cc_interface",
        "electron_density_at_cc_interface",
        "electron_temp_at_cc_interface",
        "parallel_heat_flux_at_cc_interface",
    ]
)
def calc_values_at_conduction_convection_interface(
    parallel_heat_flux,
    parallel_conductive_heat_flux,
    s_parallel,
    electron_density,
    electron_temp,
    model_success,
    conducted_fraction_at_cc_interface=0.99,
):
    """Calculate the density, temp and heat flux at the conducted/convection interface."""
    cc_index = np.abs((parallel_conductive_heat_flux / parallel_heat_flux).fillna(0.0) - conducted_fraction_at_cc_interface).argmin(
        dim="dim_s_parallel"
    )

    s_parallel_at_cc_interface = s_parallel.isel(dim_s_parallel=cc_index)
    electron_density_at_cc_interface = electron_density.isel(dim_s_parallel=cc_index)
    electron_temp_at_cc_interface = electron_temp.isel(dim_s_parallel=cc_index)
    parallel_heat_flux_at_cc_interface = parallel_heat_flux.isel(dim_s_parallel=cc_index)

    return (
        s_parallel_at_cc_interface.where(model_success),
        electron_density_at_cc_interface.where(model_success),
        electron_temp_at_cc_interface.where(model_success),
        parallel_heat_flux_at_cc_interface.where(model_success),
    )


@Algorithm.register_algorithm(return_keys=["SOL_power_loss_fraction_in_convection_layer"])
def calc_SOL_power_loss_fraction_in_convection_layer(parallel_heat_flux_at_target, parallel_heat_flux_at_cc_interface):
    """Calculate the power loss in the convection layer."""
    return 1.0 - parallel_heat_flux_at_target / parallel_heat_flux_at_cc_interface


@Algorithm.register_algorithm(return_keys=["SOL_momentum_loss_fraction_in_convection_layer"])
def calc_SOL_momentum_loss_fraction_in_convection_layer(
    electron_density_at_target, target_electron_temp, electron_density_at_cc_interface, electron_temp_at_cc_interface
):
    """Calculate the momentum loss in the convection layer."""
    return 1.0 - (2.0 * electron_density_at_target * target_electron_temp) / (
        electron_density_at_cc_interface * electron_temp_at_cc_interface
    )


@Algorithm.register_algorithm(return_keys=["electron_temp_ratio_in_convection_layer"])
def calc_electron_temp_ratio_in_convection_layer(electron_temp_at_cc_interface, target_electron_temp):
    """Calculate the ratio of the electron temp at the convection-conduction boundary and at the target."""
    return target_electron_temp / electron_temp_at_cc_interface


@Algorithm.register_algorithm(return_keys=["electron_density_ratio_in_convection_layer"])
def calc_electron_density_ratio_in_convection_layer(electron_density_at_cc_interface, electron_density_at_target):
    """Calculate the ratio of the electron density at the convection-conduction boundary and at the target."""
    return electron_density_at_target / electron_density_at_cc_interface


@Algorithm.register_algorithm(return_keys=["lambda_int"])
def calc_lambda_int_from_lambda_q(lambda_q, divertor_broadening_factor):
    """Calculate lambda-int from lambda-q, assuming a fixed ratio between the two."""
    return lambda_q * divertor_broadening_factor


@Algorithm.register_algorithm(return_keys=["heat_flux_perp_to_target"])
def calc_heat_flux_perp_to_target(parallel_heat_flux_at_target, parallel_to_perp_factor):
    """Calculate the perpendicular heat flux density at the target."""
    return parallel_heat_flux_at_target * parallel_to_perp_factor
