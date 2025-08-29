"""Interface to atomic data, using the Mavrin polynomials.

This is intended to function as a drop-in replacement for "AtomicData".
"""
from cfspopcon.named_options import AtomicSpecies
from radas.mavrin_reference.read_mavrin_data import read_mavrin_data as _read_mavrin_data, compute_Mavrin_polynomial_fit_single
from cfspopcon.unit_handling import Unitfull, ureg, wraps_ufunc
import numpy as np
import xarray as xr
from cfspopcon import Algorithm

class SpeciesMavrinData:
    """An interface to the Mavrin polynomials, for a single atomic species."""

    def __init__(self, species: AtomicSpecies, Lz_coeffs: dict[str, list[float | int]], mean_charge_coeffs: dict[str, list[float | int]]) -> None:
        """Builds an object to provide atomic data from the Mavrin polynomials, for a single atomic species."""
        self.species: AtomicSpecies = species
        self.Lz_coeffs: dict[str, list[float | int]] = Lz_coeffs
        self.mean_charge_coeffs: dict[str, list[float | int]] = mean_charge_coeffs

    @staticmethod
    def _return_values_for_curve(coeffs: dict[str, list[float | int]], ne_tau: Unitfull, resolution: int=100) -> xr.DataArray:
        Tmin = np.min(coeffs["Tmin_eV"])
        Tmax = np.max(coeffs["Tmax_eV"])

        electron_temp = xr.DataArray(np.logspace(np.log10(Tmin), np.log10(Tmax), num = resolution), dims="dim_electron_temp")
        electron_temp = np.minimum(np.maximum(electron_temp, Tmin), Tmax)

        curve_function = wraps_ufunc(
            input_units=dict(Te_eV=ureg.eV, ne_tau_s_per_m3=ureg.m**-3 * ureg.s, coeff=None, warn=None),
            return_units=dict(result = ureg.W * ureg.m**3),
            pass_as_kwargs=("coeff", "warn")
        )(compute_Mavrin_polynomial_fit_single)

        return curve_function(electron_temp * ureg.eV, ne_tau, coeff=coeffs).assign_coords(dim_electron_temp=electron_temp)

    def get_Lz_curve(self, ne_tau: Unitfull, resolution: int=100) -> xr.DataArray:
        """Returns the Lz curve from the Mavrin polynomial for the given species."""
        return self._return_values_for_curve(self.Lz_coeffs, ne_tau, resolution)

    def get_mean_charge_curve(self, ne_tau: Unitfull, resolution: int=100) -> xr.DataArray:
        """Returns the Lz curve from the Mavrin polynomials."""
        return self._return_values_for_curve(self.mean_charge_coeffs, ne_tau, resolution)

class MavrinData:
    """An interface to the Mavrin polynomials, for all available atomic species."""

    def __init__(self) -> None:
        """Builds an object to provide atomic data from the Mavrin polynomials."""
        mavrin_data = _read_mavrin_data()

        species_available = {key.removesuffix("_Lz").removesuffix("_mean_charge") for key in mavrin_data.keys()}

        self.datasets: dict[AtomicSpecies, SpeciesMavrinData] = dict()
        for key in species_available:
            species = AtomicSpecies[key.capitalize()]
            self.datasets[species] = SpeciesMavrinData(species, mavrin_data[f"{key}_Lz"], mavrin_data[f"{key}_mean_charge"])

@Algorithm.register_algorithm(return_keys=["atomic_data"])
def read_mavrin_data() -> MavrinData:
    """Construct a MavrinData interface using the Mavrin polynomials."""
    return MavrinData()
