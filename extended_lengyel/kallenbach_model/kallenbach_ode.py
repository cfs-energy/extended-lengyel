"""Defines the main ODEs solved by the Kallenbach model."""

from cfspopcon.unit_handling import magnitude_in_units as umag
from cfspopcon.unit_handling import ureg

from ..adas_data import AtomicSpeciesAdasData
from ..initialize import calc_Goldston_kappa_z

eV_to_J = umag(1.0 * ureg.eV, ureg.J)


def kallenbach_change_in_state_vector(
    _,
    state_vector,
    impurity_fraction,
    kappa_e0,
    z_effective,
    ion_mass,
    franck_condon_neutral_velocity,
    hydrogen_effective_ionization_energy,
    deuterium_adas_data: AtomicSpeciesAdasData,
    impurity_adas_data: AtomicSpeciesAdasData,
):
    """Given a state vector, return the derivative of each element of the state vector w.r.t. s_parallel."""
    electron_density, ion_velocity, neutral_density, electron_temp, parallel_heat_flux = state_vector

    parallel_convective_heat_flux = (
        -(5.0 * electron_density * (electron_temp * eV_to_J) + 0.5 * ion_mass * electron_density * ion_velocity**2) * ion_velocity
    )
    parallel_conductive_heat_flux = parallel_heat_flux - parallel_convective_heat_flux

    ion_temp, ion_density = electron_temp, electron_density

    impurity_radiated_power = (
        impurity_adas_data.radiative_power_coeff.unitless_eval(electron_density, electron_temp) * impurity_fraction * electron_density**2
    )
    ionization_rate = (
        deuterium_adas_data.ionization_rate.unitless_eval(electron_density, electron_temp) * electron_density * neutral_density
    )
    recombination_rate = (
        deuterium_adas_data.recombination_rate.unitless_eval(electron_density, electron_temp) * electron_density * ion_density
    )
    charge_exchange_rate = (
        deuterium_adas_data.charge_exchange_rate.unitless_eval(electron_density, electron_temp) * neutral_density * ion_density
    )

    change_in_parallel_heat_flux = (
        impurity_radiated_power
        + (ion_temp * eV_to_J) * charge_exchange_rate
        + (hydrogen_effective_ionization_energy * eV_to_J) * ionization_rate
    )

    change_in_neutral_density = (-ionization_rate + recombination_rate) / franck_condon_neutral_velocity

    kappa_z = calc_Goldston_kappa_z(z_effective)
    electron_heat_conductivity = kappa_e0 / kappa_z
    change_in_electron_temp = parallel_conductive_heat_flux / (electron_heat_conductivity * electron_temp**2.5)

    change_in_electron_density = (1 / (ion_mass * ion_velocity**2 - 2.0 * (electron_temp * eV_to_J))) * (
        ion_mass * ion_velocity * (2.0 * ionization_rate - recombination_rate + charge_exchange_rate)
        + 2 * electron_density * (change_in_electron_temp * eV_to_J)
    )

    change_in_ion_velocity = (1 / electron_density) * (ionization_rate - recombination_rate - ion_velocity * change_in_electron_density)

    return (
        change_in_electron_density,
        change_in_ion_velocity,
        change_in_neutral_density,
        change_in_electron_temp,
        change_in_parallel_heat_flux,
    )
