"""Common initialization routines (shared by several models)."""

import numpy as np
from cfspopcon.algorithm_class import Algorithm
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import convert_units, ureg

from .adas_data import AtomicSpeciesAdasData
from .directories import radas_dir

from cfspopcon.formulas.separatrix_conditions.separatrix_operational_space.shared import calc_lambda_q_Eich2020H
from cfspopcon.formulas.metrics.larmor_radius import calc_larmor_radius
from .xr_helpers import item

@Algorithm.register_algorithm(return_keys=["radas_dir"])
def set_radas_dir():
    """Sets the radas directory."""
    return radas_dir


@Algorithm.register_algorithm(return_keys=["deuterium_adas_data"])
def read_deuterium_adas_data(reference_ne_tau, radas_dir=None):
    """Read in the deuterium radas data."""
    radas_dir = item(radas_dir)
    return AtomicSpeciesAdasData(species_name="deuterium", reference_ne_tau=reference_ne_tau, radas_dir=radas_dir)


@Algorithm.register_algorithm(return_keys=["impurity_adas_data"])
def read_impurity_adas_data(impurity_species: AtomicSpecies, reference_ne_tau, radas_dir=None):
    """Read in the radas data corresponding to the specified edge impurity species."""
    impurity_species = item(impurity_species)
    radas_dir = item(radas_dir)

    return AtomicSpeciesAdasData(species_name=impurity_species, reference_ne_tau=reference_ne_tau, radas_dir=radas_dir)


@Algorithm.register_algorithm(return_keys=["parallel_to_perp_factor"])
def calc_parallel_to_perp_factor(target_angle_of_incidence):
    """Calculate the projection of parallel quantities into perpendicular-to-target quantities.

    i.e. q_perp = parallel_to_perp_factor * q_parallel
    """
    return np.sin(convert_units(target_angle_of_incidence, ureg.radian))


@Algorithm.register_algorithm(return_keys=["lambda_q"])
def calc_lambda_q_from_lambda_int(lambda_int, divertor_broadening_factor):
    """Calculate lambda-q from lambda-int, assuming a fixed ratio between the two."""
    return lambda_int / divertor_broadening_factor


@Algorithm.register_algorithm(return_keys=["parallel_heat_flux_at_target"])
def calc_parallel_heat_flux_at_target(heat_flux_perp_to_target, parallel_to_perp_factor):
    """Calculate the parallel heat flux density at the target."""
    return heat_flux_perp_to_target / parallel_to_perp_factor


@Algorithm.register_algorithm(return_keys=["flux_tube_cross_section_area_in_divertor"])
def calc_flux_tube_cross_section_area_in_divertor(major_radius, minor_radius, lambda_int, B_t_out_mid, B_pol_out_mid):
    """Calculate the cross-sectional flux_tube_cross_section_area of the flux tube in the divertor."""
    major_radius_at_OMP = major_radius + minor_radius
    circumference_at_OMP = 2.0 * np.pi * major_radius_at_OMP

    return circumference_at_OMP * lambda_int * np.sin(np.arctan(B_pol_out_mid / B_t_out_mid))


@Algorithm.register_algorithm(return_keys=["flux_tube_cross_section_area_out_of_divertor"])
def calc_flux_tube_cross_section_area_out_of_divertor(major_radius, minor_radius, lambda_q, B_t_out_mid, B_pol_out_mid):
    """Calculate the cross-sectional flux_tube_cross_section_area of the flux tube above the X-point."""
    major_radius_at_OMP = major_radius + minor_radius
    circumference_at_OMP = 2.0 * np.pi * major_radius_at_OMP

    return circumference_at_OMP * lambda_q * np.sin(np.arctan(B_pol_out_mid / B_t_out_mid))


@Algorithm.register_algorithm(return_keys=["sound_speed_at_target"])
def calc_sound_speed_at_target(target_electron_temp, ion_mass):
    """Calculate the sound speed at the target."""
    return np.sqrt(2.0 * target_electron_temp / ion_mass)


def mean_thermal_velocity(particle_temp, particle_mass):
    """Calculate the mean thermal velocity for thermal distribution of particles."""
    return np.sqrt(8.0 / np.pi * particle_temp / particle_mass)


@Algorithm.register_algorithm(return_keys=["franck_condon_neutral_velocity"])
def calc_franck_condon_neutral_velocity(ion_mass, Franck_condon_energy=5.0 * ureg.eV):
    """Calculate the mean thermal velocity of Franck-Condon neutrals."""
    return 0.25 * mean_thermal_velocity(Franck_condon_energy, ion_mass)


@Algorithm.register_algorithm(return_keys=["electron_density_at_target"])
def calc_target_density(parallel_heat_flux_at_target, target_electron_temp, sound_speed_at_target, sheath_heat_transmission_factor):
    """Calculate the density at the divertor target."""
    return parallel_heat_flux_at_target / (sheath_heat_transmission_factor * target_electron_temp * sound_speed_at_target)


@Algorithm.register_algorithm(return_keys=["parallel_ion_flux_to_target", "perp_ion_flux_to_target"])
def calc_ion_flux_to_target(electron_density_at_target, sound_speed_at_target, parallel_to_perp_factor):
    """Calculate the parallel and perpendicular ion fluxes to the target."""
    parallel_ion_flux_to_target = electron_density_at_target * sound_speed_at_target
    perp_ion_flux_to_target = parallel_ion_flux_to_target * parallel_to_perp_factor
    return parallel_ion_flux_to_target, perp_ion_flux_to_target


@Algorithm.register_algorithm(return_keys=["flux_density_to_pascals_factor"])
def calc_flux_density_to_pascals_factor(ion_mass, ratio_of_molecular_to_ion_mass=2.0, wall_temperature=300.0 * ureg.K):
    """Calculate a factor to convert from a flux density to a pressure."""
    if wall_temperature.check("[temperature]"):
        wall_temperature = ureg.k_B * wall_temperature

    test_molecular_density = 1e20 * ureg.m**-3
    test_molecular_pressure = test_molecular_density * wall_temperature
    neutral_density = 2.0 * test_molecular_density
    molecular_mass = ion_mass * ratio_of_molecular_to_ion_mass

    onesided_maxwellian_flux_density = 0.25 * mean_thermal_velocity(wall_temperature, molecular_mass)
    flux_density_to_pascals_factor = neutral_density * onesided_maxwellian_flux_density / test_molecular_pressure

    return flux_density_to_pascals_factor


@Algorithm.register_algorithm(return_keys=["neutral_pressure_in_divertor"])
def calc_divertor_neutral_pressure(parallel_ion_flux_to_target, parallel_to_perp_factor, flux_density_to_pascals_factor):
    """Calculate the divertor neutral pressure."""
    return parallel_ion_flux_to_target * parallel_to_perp_factor / flux_density_to_pascals_factor


@Algorithm.register_algorithm(return_keys=["fast_neutral_velocity"])
def calc_fast_neutral_velocity(franck_condon_neutral_velocity, fast_neutral_penetration_factor):
    """Calculate the velocity of fast neutrals.

    These neutrals represent those which travel outside of the flux tube, and therefore
    penetrate further separatrix.
    """
    return franck_condon_neutral_velocity * fast_neutral_penetration_factor


@Algorithm.register_algorithm(return_keys=["franck_condon_neutral_flux_at_target", "fast_neutral_flux_at_target"])
def calc_neutral_recycling_fluxes(parallel_ion_flux_to_target, fast_neutral_fraction):
    """Calculate the neutral fluxes coming from the target in the slow and fast populations.

    This is equal to the incident ion flux, assuming perfect recycling.
    """
    franck_condon_neutral_flux_at_target = parallel_ion_flux_to_target * (1 - fast_neutral_fraction)
    fast_neutral_flux_at_target = parallel_ion_flux_to_target * fast_neutral_fraction
    return franck_condon_neutral_flux_at_target, fast_neutral_flux_at_target


@Algorithm.register_algorithm(return_keys=["z_effective"])
def calc_z_effective(
    impurity_adas_data: AtomicSpeciesAdasData,
    impurity_fraction,
    reference_electron_temp=30.0 * ureg.eV,
    reference_electron_density=1e20 * ureg.m**-3,
):
    """Calculate the effective charge due to the seeding species."""
    impurity_adas_data = item(impurity_adas_data)

    mean_z = impurity_adas_data.mean_charge.eval(reference_electron_density, reference_electron_temp)

    return 1.0 + mean_z * (mean_z - 1.0) * impurity_fraction


@Algorithm.register_algorithm(return_keys=["convective_heat_flux_at_target"])
def calc_q_convective_tar(electron_density_at_target, target_electron_temp, ion_mass, sound_speed_at_target):
    """Calculate the convected flux at the target."""
    return (
        5.0 * target_electron_temp * electron_density_at_target + 0.5 * ion_mass * electron_density_at_target * sound_speed_at_target**2
    ) * sound_speed_at_target


@Algorithm.register_algorithm(return_keys=["conductive_heat_flux_at_target"])
def calc_q_conductive_tar(parallel_heat_flux_at_target, convective_heat_flux_at_target):
    """Calculate the conducted flux at the target."""
    return parallel_heat_flux_at_target - convective_heat_flux_at_target


@Algorithm.register_algorithm(return_keys=["kappa_e0"])
def calc_kappa_e0():
    """Return the electron heat conductivity.

    N.b. This does not include the Zeff dependence.
    """
    return 2390.0 * ureg.W / (ureg.m * ureg.eV**3.5)


@Algorithm.register_algorithm(return_keys=["neutral_density_at_target"])
def calc_neutral_density_at_target(electron_density_at_target, sound_speed_at_target, franck_condon_neutral_velocity):
    """Calculate the neutral density at the target."""
    return electron_density_at_target * sound_speed_at_target / franck_condon_neutral_velocity


@Algorithm.register_algorithm(return_keys=["kappa_z"])
def calc_Goldston_kappa_z(z_effective):
    """Calculate the correction to the electron heat conduction, due to z-effective.

    kappa = kappa_e0 / kappa_z

    Equation 10 from Brown and Goldston, 2021, NME 27 101002
    """
    return 0.672 + 0.076 * np.sqrt(z_effective) + 0.252 * z_effective


@Algorithm.register_algorithm(return_keys=["kappa_z"])
def calc_Kallenbach_kappa_z(z_effective):
    """Calculate the correction to the electron heat conduction, due to z-effective.

    kappa = kappa_e0 / kappa_z

    Correction from equation 5 from Kallenbach et al, 2016, PPCF 58 045013
    """
    return z_effective**0.3


@Algorithm.register_algorithm(return_keys=["B_t_out_mid", "B_pol_out_mid", "separatrix_average_poloidal_field", "cylindrical_safety_factor"])
def calc_magnetic_field_and_safety_factor(
    magnetic_field_on_axis, major_radius, minor_radius, elongation_psi95, triangularity_psi95, plasma_current, ratio_of_upstream_to_average_poloidal_field
):
    """Calculate upstream magnetic field."""
    shaping_factor = np.sqrt((1.0 + elongation_psi95**2 * (1.0 + 2.0 * triangularity_psi95**2 - 1.2 * triangularity_psi95**3)) / 2.0)
    poloidal_circumference = 2.0 * np.pi * minor_radius * shaping_factor

    upstream_toroidal_field = magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))
    separatrix_average_poloidal_field = ureg.mu_0 * plasma_current / poloidal_circumference
    upstream_poloidal_field = ratio_of_upstream_to_average_poloidal_field * separatrix_average_poloidal_field

    cylindrical_safety_factor = magnetic_field_on_axis / separatrix_average_poloidal_field * minor_radius / major_radius * shaping_factor

    return upstream_toroidal_field, upstream_poloidal_field, separatrix_average_poloidal_field, cylindrical_safety_factor


@Algorithm.register_algorithm(return_keys=["kappa_z"])
def ignore_kappa_z():
    """Ignore the z-effective correction to the electron heat conductivity."""
    return 1.0


@Algorithm.register_algorithm(return_keys=["lambda_q"])
def calc_lambda_q_HD(separatrix_average_poloidal_field, ion_mass, separatrix_electron_temp=100.0 * ureg.eV):
    """Return the lambda-q corresponding to Eich et al., 2020, neglecting the effect of broadening."""
    poloidal_sound_larmor_radius = calc_larmor_radius(
        species_temperature=separatrix_electron_temp,
        magnetic_field_strength=separatrix_average_poloidal_field,
        species_mass=ion_mass,
    )
    return calc_lambda_q_Eich2020H(alpha_t=0.0, poloidal_sound_larmor_radius=poloidal_sound_larmor_radius)
