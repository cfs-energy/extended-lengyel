"""Defines the main ODEs solved by the spatial Lengyel model."""

from cfspopcon.unit_handling import magnitude_in_units as umag
from cfspopcon.unit_handling import ureg

from ..adas_data import AtomicSpeciesAdasData
from ..initialize import calc_Goldston_kappa_z

eV_to_J = umag(1.0 * ureg.eV, ureg.J)


def spatial_lengyel_change_in_state_vector(
    _,
    state_vector,
    impurity_fraction,
    kappa_e0,
    z_effective,
    impurity_adas_data: AtomicSpeciesAdasData,
):
    """Given a state vector, return the derivative of each element of the state vector w.r.t. s_parallel."""
    parallel_heat_flux, electron_density, electron_temp = state_vector

    impurity_radiated_power = (
        impurity_adas_data.radiative_power_coeff.unitless_eval(electron_density, electron_temp) * impurity_fraction * electron_density**2
    )

    kappa_z = calc_Goldston_kappa_z(z_effective)
    electron_heat_conductivity = kappa_e0 / kappa_z
    change_in_electron_temp = parallel_heat_flux / (electron_heat_conductivity * electron_temp**2.5)

    change_in_parallel_heat_flux = impurity_radiated_power
    change_in_electron_density = -electron_density / electron_temp * change_in_electron_temp

    return (change_in_parallel_heat_flux, change_in_electron_density, change_in_electron_temp)
