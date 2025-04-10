"""Core functions for evaluating the Lengyel model."""

import xarray as xr
from cfspopcon import Algorithm
from cfspopcon.formulas.impurities.edge_radiator_conc import build_L_int_integrator
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.formulas.atomic_data import AtomicData
from cfspopcon.unit_handling import ureg, Unitfull

from .mean_charge_interpolator import build_mean_charge_interpolator


def item(val):
    """Extract items from xr.DataArray wrappers."""
    if isinstance(val, xr.DataArray):
        return val.item()
    else:
        return val


class L_int_integrator:
    """Class to hold a mixed-seeding L-int integrator."""

    def __init__(
        self,
        impurity_species_list: list[AtomicSpecies],
        impurity_weights_list: list[float],
        atomic_data: AtomicData,
        reference_ne_tau: Unitfull = 0.5 * ureg.ms * ureg.n20,
        reference_electron_density: Unitfull = 1.0 * ureg.n20,
    ):
        """Initializes a L_int_integrator from linked lists of impurity species and weights."""
        self.integrators = []
        self.weights_list = impurity_weights_list

        for impurity_species in impurity_species_list:
            integrator = build_L_int_integrator(
                atomic_data=item(atomic_data),
                impurity_species=item(impurity_species),
                reference_electron_density=reference_electron_density,
                reference_ne_tau=reference_ne_tau,
            )

            self.integrators.append(integrator)

    def __call__(self, start_temp: Unitfull, stop_temp: Unitfull) -> Unitfull:
        """Return sum_z(Lint(start_temp, stop_temp) * wz), handling input and output units."""
        weighted_L_INT = 0.0

        for weight, integrator in zip(self.weights_list, self.integrators):
            weighted_L_INT += weight * integrator(start_temp, stop_temp)

        return weighted_L_INT

    def unitless_eval(self, start_temp: Unitfull, stop_temp: Unitfull) -> Unitfull:
        """Return sum_z(Lint(start_temp, stop_temp) * wz), without handling input and output units."""
        weighted_L_INT = 0.0

        for weight, integrator in zip(self.weights_list, self.integrators):
            weighted_L_INT += weight * integrator.unitless_func(start_temp, stop_temp)

        return weighted_L_INT


@Algorithm.register_algorithm(return_keys=["L_int_integrator"])
def build_mixed_seeding_L_int_integrator(
    impurity_species_list,
    impurity_weights_list,
    atomic_data,
    reference_ne_tau=0.5 * ureg.ms * ureg.n20,
    reference_electron_density=1.0 * ureg.n20,
):
    """Build an L_int integrator which returns sum_z(w_z L_int_z) for multiple species.

    N.b. the weighted sum is not normalized.
    i.e. If you define w_z = [1, 2] for species = [N, Ne], if the
    Lengyel model computes a 1% cz then that should be interpreted as 1% N, 2% Ne
    """
    return L_int_integrator(
        impurity_species_list=impurity_species_list,
        impurity_weights_list=impurity_weights_list,
        atomic_data=atomic_data,
        reference_ne_tau=reference_ne_tau,
        reference_electron_density=reference_electron_density,
    )


class Mean_charge_interpolator:
    """Class to hold a mixed-seeding mean charge interpolator."""

    def __init__(
        self,
        impurity_species_list: list[AtomicSpecies],
        impurity_weights_list: list[float],
        atomic_data: AtomicData,
        reference_ne_tau: Unitfull = 0.5 * ureg.ms * ureg.n20,
        reference_electron_density: Unitfull = 1.0 * ureg.n20,
    ):
        """Initializes a Mean_charge_interpolator from linked lists of impurity species and weights."""
        self.interpolators = []
        self.weights_list = impurity_weights_list

        for impurity_species in impurity_species_list:
            interpolator = build_mean_charge_interpolator(
                atomic_data=item(atomic_data),
                impurity_species=item(impurity_species),
                reference_electron_density=reference_electron_density,
                reference_ne_tau=reference_ne_tau,
            )

            self.interpolators.append(interpolator)

    def __call__(self, electron_temp: Unitfull) -> Unitfull:
        """Return sum_z(<Z>(electron_temp) * wz) / sum_z(wz), handling input and output units."""
        weighted_mean_charge = 0.0
        total_weights = 0.0

        for weight, interpolator in zip(self.weights_list, self.interpolators):
            weighted_mean_charge += weight * interpolator(electron_temp)
            total_weights += weight

        return weighted_mean_charge / total_weights

    def unitless_eval(self, electron_temp: Unitfull) -> Unitfull:
        """Return sum_z(<Z>(electron_temp) * wz) / sum_z(wz), without handling input and output units."""
        weighted_mean_charge = 0.0
        total_weights = 0.0

        for weight, interpolator in zip(self.weights_list, self.interpolators):
            weighted_mean_charge += weight * interpolator.unitless_func(electron_temp)
            total_weights += weight

        return weighted_mean_charge / total_weights


@Algorithm.register_algorithm(return_keys=["mean_charge_interpolator"])
def build_mixed_seeding_mean_charge_interpolator(
    impurity_species_list,
    impurity_weights_list,
    atomic_data,
    reference_ne_tau=0.5 * ureg.ms * ureg.n20,
    reference_electron_density=1.0 * ureg.n20,
):
    """Build a mean charge state interpolator which returns sum(w_z <Z>)/sum(w_z)."""
    return Mean_charge_interpolator(
        impurity_species_list=impurity_species_list,
        impurity_weights_list=impurity_weights_list,
        atomic_data=atomic_data,
        reference_ne_tau=reference_ne_tau,
        reference_electron_density=reference_electron_density,
    )
