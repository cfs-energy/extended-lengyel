"""Core functions for evaluating the Lengyel model with mixed impurities.

See https://github.com/cfs-energy/extended-lengyel/wiki/Background-impurities.
"""

import xarray as xr
import numpy as np
from cfspopcon import Algorithm
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.formulas.atomic_data import AtomicData
from cfspopcon.unit_handling import magnitude, ureg, wraps_ufunc, Unitfull, magnitude_in_units
from scipy.interpolate import InterpolatedUnivariateSpline  # type:ignore[import-untyped]
from typing import Self, Callable
from ..xr_helpers import item

def get_species_array(impurity_species_list) -> list[AtomicSpecies]:
    """Get a list of impurity species."""
    if isinstance(impurity_species_list, xr.DataArray):
        impurity_species_list = impurity_species_list.values
    return [s if isinstance(s, AtomicSpecies) else AtomicSpecies[s] for s in impurity_species_list]

class CzLINT_integrator:
    """Class to hold an L-int integrator."""

    def __init__(
        self,
        impurity_species_list: list[AtomicSpecies],
        impurity_weights_list: list[float],
        atomic_data: AtomicData,
        ne_tau: Unitfull = 0.5 * ureg.ms * ureg.n20,
        electron_density: Unitfull = 1.0 * ureg.n20,
        rtol_nearest: float=1e-6,
    ) -> None:
        """Initializes a CzLINT_for_seed_impurities from linked lists of impurity species and weights."""
        assert (np.ndim(impurity_species_list) == 1) and (np.ndim(impurity_weights_list) == 1)
        assert len(impurity_species_list) == len(impurity_weights_list)

        self.species = get_species_array(impurity_species_list)
        self.weights = xr.DataArray(impurity_weights_list, coords=dict(dim_species = self.species))
        self.is_empty = len(self.species) == 0

        self.integrators = dict()
        for species in self.species:
            self.integrators[species] = self.build_L_int_integrator(
                species_atomic_data = item(atomic_data).get_dataset(item(species)),
                electron_density=electron_density,
                ne_tau=ne_tau,
                rtol_nearest=rtol_nearest
            )

    def __call__(self, start_temp: Unitfull, stop_temp: Unitfull, **kwargs) -> Unitfull:
        """Return the weighted L_INT, handling input and output units.

        N.b. this is equivalent to sum_z (c_z L_INT).
        """
        return self._inner(start_temp, stop_temp, integrator_method="__call__", **kwargs)

    def unitless_eval(self, start_temp: Unitfull, stop_temp: Unitfull, **kwargs) -> Unitfull:
        """Return the weighted L_INT, without handling input and output units.

        N.b. this is equivalent to sum_z (c_z L_INT).
        """
        return self._inner(start_temp, stop_temp, integrator_method="unitless_eval", **kwargs)

    def _inner(self, start_temp: Unitfull, stop_temp: Unitfull, integrator_method: str, allow_negative: bool=False) -> Unitfull:
        """Common function for unitless and unit-aware eval."""
        if not(allow_negative):
            stop_temp = np.maximum(stop_temp, start_temp)

        weighted_L_INT = 0.0 * ureg.W * ureg.m**3 * ureg.eV**1.5

        for species in self.species:
            weight = self.weights.sel(dim_species = species)
            integrator = self.integrators[species].__getattribute__(integrator_method)

            weighted_L_INT += weight * integrator(start_temp, stop_temp)

        return weighted_L_INT

    @staticmethod
    def build_L_int_integrator(
        species_atomic_data: xr.Dataset,
        electron_density: Unitfull,
        ne_tau: Unitfull,
        rtol_nearest: float=1e-6,
    ) -> Callable[[Unitfull, Unitfull], Unitfull]:
        r"""Build an interpolator to calculate the integral of L_{int}$ between arbitrary temperature points.

        $L_int = \\int_a^b L_z(T_e) sqrt(T_e) dT_e$ where $L_z$ is a cooling curve for an impurity species.
        This is used in the calculation of the radiated power associated with a given impurity.
        """
        electron_density_ref = magnitude_in_units(electron_density, ureg.m**-3)
        ne_tau_ref = magnitude_in_units(ne_tau, ureg.m**-3 * ureg.s)

        Lz_curve = (
            species_atomic_data.equilibrium_Lz
            .sel(dim_electron_density=electron_density_ref, method="nearest", tolerance=rtol_nearest * electron_density_ref)
            .sel(dim_ne_tau=ne_tau_ref, method="nearest", tolerance=rtol_nearest * ne_tau_ref)
        )

        electron_temp = Lz_curve.dim_electron_temp
        Lz_sqrt_Te = Lz_curve * np.sqrt(electron_temp)

        interpolator = InterpolatedUnivariateSpline(electron_temp, magnitude(Lz_sqrt_Te), ext=3)

        def L_int(start_temp: float, stop_temp: float) -> float:
            integrated_Lz: float = interpolator.integral(start_temp, stop_temp)
            return integrated_Lz

        CzLINT_for_seed_impurities: Callable[[Unitfull, Unitfull], Unitfull] = wraps_ufunc(
            input_units=dict(start_temp=ureg.eV, stop_temp=ureg.eV), return_units=dict(L_int=ureg.W * ureg.m**3 * ureg.eV**1.5)
        )(L_int)
        return CzLINT_for_seed_impurities

    @classmethod
    def empty(cls) -> Self:
        """Returns an empty CzLINT_for_seed_impurities which always returns 0.0."""
        return cls(impurity_species_list=[], impurity_weights_list=[], atomic_data=None)


@Algorithm.register_algorithm(return_keys=["CzLINT_for_seed_impurities"])
def build_CzLINT_for_seed_impurities(
        seed_impurity_species,
        seed_impurity_weights,
        atomic_data,
        reference_ne_tau = 0.5 * ureg.ms * ureg.n20,
        reference_electron_density = 1.0 * ureg.n20,
        rtol_nearest_for_atomic_data = 1e-6,
    ) -> CzLINT_integrator:
    """Build a CzLINT_integrator for the seed impurities."""
    return CzLINT_integrator(seed_impurity_species, seed_impurity_weights, atomic_data, reference_ne_tau, reference_electron_density, rtol_nearest_for_atomic_data)


@Algorithm.register_algorithm(return_keys=["CzLINT_for_fixed_impurities"])
def build_CzLINT_for_fixed_impurities(
        fixed_impurity_species,
        fixed_impurity_weights,
        atomic_data,
        reference_ne_tau = 0.5 * ureg.ms * ureg.n20,
        reference_electron_density = 1.0 * ureg.n20,
        rtol_nearest_for_atomic_data = 1e-6,
    ) -> CzLINT_integrator:
    """Build a CzLINT_integrator for the fixed impurities."""
    return CzLINT_integrator(fixed_impurity_species, fixed_impurity_weights, atomic_data, reference_ne_tau, reference_electron_density, rtol_nearest_for_atomic_data)


class Mean_charge_interpolator:
    """Class to hold a mixed-seeding Zeff interpolator."""

    def __init__(
        self,
        impurity_species_list: list[AtomicSpecies],
        atomic_data: AtomicData,
        ne_tau: Unitfull = 0.5 * ureg.ms * ureg.n20,
        electron_density: Unitfull = 1.0 * ureg.n20,
        rtol_nearest: float=1e-6,
    ) -> None:
        """Initializes a Mean_charge_interpolator from a list of impurity species."""
        assert np.ndim(impurity_species_list) == 1

        self.species = get_species_array(impurity_species_list)
        self.is_empty = len(self.species) == 0

        self.interpolators = dict()
        for species in self.species:
            self.interpolators[species] = self.build_mean_charge_interpolator(
                species_atomic_data = item(atomic_data).get_dataset(item(species)),
                electron_density=electron_density,
                ne_tau=ne_tau,
                rtol_nearest=rtol_nearest
            )

    def __call__(self, electron_temp: Unitfull) -> Unitfull:
        """Return the mean charge of each impurity species, handling input and output units."""
        return self._inner(electron_temp, interpolator_method="__call__")

    def unitless_eval(self, electron_temp: Unitfull) -> Unitfull:
        """Return the mean charge of each impurity species, without handling input and output units."""
        return self._inner(electron_temp, interpolator_method="unitless_eval")

    def _inner(self, electron_temp: Unitfull, interpolator_method: str) -> Unitfull:
        """Common function for unitless and unit-aware eval."""
        if self.is_empty:
            return xr.DataArray([], dims="dim_species").broadcast_like(xr.DataArray(electron_temp))

        mean_charge = [
            xr.DataArray(self.interpolators[species_obj].__getattribute__(interpolator_method)(electron_temp))
            for species_obj in self.species
        ]

        return xr.concat(mean_charge, dim=xr.DataArray(self.species, dims="dim_species"))

    @staticmethod
    def build_mean_charge_interpolator(
        species_atomic_data: xr.Dataset,
        electron_density,
        ne_tau,
        rtol_nearest = 1e-6
    ) -> Callable[[Unitfull], Unitfull]:
        """Build an interpolator to calculate the mean charge."""
        electron_density_ref = magnitude_in_units(electron_density, ureg.m**-3)
        reference_ne_tau_ref = magnitude_in_units(ne_tau, ureg.m**-3 * ureg.s)

        mean_z_curve = (
            species_atomic_data.equilibrium_mean_charge_state
            .sel(dim_electron_density=electron_density_ref, method="nearest", tolerance=rtol_nearest * electron_density_ref)
            .sel(dim_ne_tau=reference_ne_tau_ref, method="nearest", tolerance=rtol_nearest * reference_ne_tau_ref)
        )

        electron_temp = mean_z_curve.dim_electron_temp
        interpolator = InterpolatedUnivariateSpline(electron_temp, magnitude(mean_z_curve), ext=3)

        def mean_charge_state(electron_temp: float) -> float:
            integrated_mean_z: float = interpolator(electron_temp)
            return integrated_mean_z

        mean_charge_state_integrator = wraps_ufunc(
            input_units=dict(electron_temp=ureg.eV), return_units=dict(mean_charge_state=ureg.dimensionless)
        )(mean_charge_state)
        return mean_charge_state_integrator

    @classmethod
    def empty(cls) -> Self:
        """Returns an empty Mean_charge_interpolator which always returns 0.0."""
        return cls(impurity_species_list=[], atomic_data=None)

@Algorithm.register_algorithm(return_keys=["mean_charge_for_seed_impurities"])
def build_mean_charge_for_seed_impurities(
        seed_impurity_species,
        atomic_data,
        reference_ne_tau = 0.5 * ureg.ms * ureg.n20,
        reference_electron_density = 1.0 * ureg.n20,
        rtol_nearest_for_atomic_data = 1e-6,
    ) -> Mean_charge_interpolator:
    """Build a Mean_charge_interpolator for the seed impurities."""
    return Mean_charge_interpolator(seed_impurity_species, atomic_data, reference_ne_tau, reference_electron_density, rtol_nearest_for_atomic_data)

@Algorithm.register_algorithm(return_keys=["mean_charge_for_fixed_impurities"])
def build_mean_charge_for_fixed_impurities(
        fixed_impurity_species,
        atomic_data,
        reference_ne_tau = 0.5 * ureg.ms * ureg.n20,
        reference_electron_density = 1.0 * ureg.n20,
        rtol_nearest_for_atomic_data = 1e-6,
    ) -> Mean_charge_interpolator:
    """Build a Mean_charge_interpolator for the fixed impurities."""
    return Mean_charge_interpolator(fixed_impurity_species, atomic_data, reference_ne_tau, reference_electron_density, rtol_nearest_for_atomic_data)

def calc_z_effective(
        electron_temp,
        c_z,
        mean_charge_for_seed_impurities: Mean_charge_interpolator,
        mean_charge_for_fixed_impurities: Mean_charge_interpolator,
        CzLINT_for_seed_impurities: CzLINT_integrator,
        CzLINT_for_fixed_impurities: CzLINT_integrator,
        starting_z_effective: float = 1.0,
        ) -> Unitfull:
    """Calculate the effective charge due to seed and fixed impurities."""
    seed_mean_z = item(mean_charge_for_seed_impurities)(electron_temp)
    fixed_mean_z = item(mean_charge_for_fixed_impurities)(electron_temp)
    seed_c_z = c_z * item(CzLINT_for_seed_impurities).weights
    fixed_c_z = item(CzLINT_for_fixed_impurities).weights
    z_effective = (
        starting_z_effective
        + (seed_mean_z * (seed_mean_z - 1.0) * seed_c_z).sum(dim="dim_species")
        + (fixed_mean_z * (fixed_mean_z - 1.0) * fixed_c_z).sum(dim="dim_species")
    )
    """Calculate the effective charge (z_effective) from the seed and fixed impurities at the given electron temp.

    The notation used here is explained in https://github.com/cfs-energy/extended-lengyel/wiki/Background-impurities.

    The seed impurities have a concentration of c_z * w_s, where w_s are the "weights" of the seed-impurity CzLINT_integrator.
    The fixed impurities have a concentration of w_f, where w_f are the "weights" of the fixed-impurity CzLINT_integrator.

    We then calculate Zeff as
    Zeff = sum[ c_s <Z_s> ] = sum[ w_f <Z_f> ] + c_z * sum[ w_s <Z_s> ]
    """
    return z_effective
