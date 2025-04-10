"""Classes to access atomic data from ADAS, reading in the data files from radas."""

from pathlib import Path
from typing import Optional

import cfspopcon  # type:ignore[import-untyped]
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type:ignore[import-untyped]

from .directories import library_directory


class AdasRate2DInterp:
    """An object for performing 2D interpolations in log-space for ADAS rate curves."""

    def __init__(self, dataarray: xr.DataArray) -> None:
        """Build an AdasRate2DInterp from a data-array read from a radas file."""
        self.dataarray = dataarray

        # Take the log of the the electron density, the electron temp and the 2D curve values
        # when building the interpolator
        self.interpolator = RectBivariateSpline(
            np.log10(dataarray.dim_electron_density),
            np.log10(dataarray.dim_electron_temp),
            np.log10(dataarray.values.T),
        )

        # Store the units of the curve, so that we can annotate the interpolation result with this
        if hasattr(self.dataarray, "units"):
            self.units = cfspopcon.unit_handling.Quantity(1.0, self.dataarray.units)
        else:
            self.units = 1.0

    def unitless_eval(self, electron_density: float | NDArray, electron_temp: float | NDArray) -> float | NDArray:
        """Interpolate the curve, leaving input and output unit handling to the user."""
        if (electron_density <= 0.0) or (electron_temp <= 0.0):
            return np.nan

        log_electron_density = np.log10(electron_density)
        log_electron_temp = np.log10(electron_temp)

        # Provide log-values to the interpolator, and then return 10^interpolation_result
        result = np.power(10, self.interpolator(log_electron_density, log_electron_temp))

        if np.size(result) == 1:
            # Return a scalar if only a single value was returned
            return float(np.squeeze(result))
        else:
            # Return an array if more than a single value was returned
            return np.array(result)

    @cfspopcon.unit_handling.wraps_ufunc(  # type:ignore[misc]
        input_units=dict(
            self=None,
            electron_density=cfspopcon.unit_handling.ureg.m**-3,
            electron_temp=cfspopcon.unit_handling.ureg.eV,
        ),
        return_units=dict(output=""),
    )
    def call_ufunc(self, electron_density: float, electron_temp: float) -> float:
        """Inner function for performing element-wise interpolation. Use 'eval' instead of this function."""
        return float(self.unitless_eval(electron_density=electron_density, electron_temp=electron_temp))

    def eval(
        self, electron_density: cfspopcon.unit_handling.Unitfull, electron_temp: cfspopcon.unit_handling.Unitfull
    ) -> cfspopcon.unit_handling.Unitfull:
        """Interpolate the curve, handling input and output units."""
        return (
            self.call_ufunc(
                electron_density=electron_density,
                electron_temp=electron_temp,
            )
            * self.units
        )


class AtomicSpeciesAdasData:
    """An object for accessing and interpolating rates from radas for a specific atomic species."""

    def __init__(
        self,
        species_name: str,
        reference_ne_tau: cfspopcon.unit_handling.Quantity = 1e20
        * cfspopcon.unit_handling.ureg.m**-3
        * 0.5
        * cfspopcon.unit_handling.ureg.ms,
        radas_dir: Optional[Path] = None,
    ) -> None:
        """Read in ADAS data for an atomic species."""
        ds = self.get_dataset(species_name, radas_dir)

        self.ionization_rate = AdasRate2DInterp(ds["effective_ionisation"].sel(dim_charge_state=0))
        self.recombination_rate = AdasRate2DInterp(ds["effective_recombination"].sel(dim_charge_state=1))
        self.charge_exchange_rate = AdasRate2DInterp(ds["charge_exchange_cross_coupling"].sel(dim_charge_state=1))

        self.radiative_power_coeff = AdasRate2DInterp(
            ds["equilibrium_Lz"].sel(
                dim_ne_tau=cfspopcon.unit_handling.magnitude_in_units(
                    reference_ne_tau,
                    cfspopcon.unit_handling.ureg.s * cfspopcon.unit_handling.ureg.m**-3,
                )
            )
        )
        self.mean_charge = AdasRate2DInterp(
            ds["equilibrium_mean_charge_state"].sel(
                dim_ne_tau=cfspopcon.unit_handling.magnitude_in_units(
                    reference_ne_tau,
                    cfspopcon.unit_handling.ureg.s * cfspopcon.unit_handling.ureg.m**-3,
                )
            )
        )

    @staticmethod
    def get_dataset(
        species_name: str | cfspopcon.named_options.AtomicSpecies,
        radas_dir: Optional[Path] = None,
    ) -> xr.Dataset:
        """Open a NetCDF dataset from radas_dir."""
        if radas_dir is None:
            radas_dir = library_directory / "radas_dir"

        if isinstance(species_name, cfspopcon.named_options.AtomicSpecies):
            species_name: str = species_name.name  # type:ignore[no-redef]

        filepath = radas_dir / "output" / f"{species_name.lower()}.nc"

        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist. Make sure that you have run radas to build the atomic data files.")
        else:
            return xr.load_dataset(filepath)
