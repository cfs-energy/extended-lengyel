"""Build an interpolator to calculate the mean charge."""

import xarray as xr
from cfspopcon.unit_handling import convert_units, magnitude, ureg, wraps_ufunc
from scipy.interpolate import InterpolatedUnivariateSpline  # type:ignore[import-untyped]


def build_mean_charge_interpolator(
    atomic_data,
    impurity_species,
    reference_electron_density,
    reference_ne_tau,
):
    """Build an interpolator to calculate the mean charge."""
    if isinstance(impurity_species, xr.DataArray):
        impurity_species = impurity_species.item()

    electron_density_ref = magnitude(convert_units(reference_electron_density, ureg.m**-3))
    reference_ne_tau_ref = magnitude(convert_units(reference_ne_tau, ureg.m**-3 * ureg.s))

    mean_z_curve = (
        atomic_data.get_dataset(impurity_species)
        .equilibrium_mean_charge_state.sel(
            dim_electron_density=electron_density_ref, method="nearest", tolerance=1e-6 * electron_density_ref
        )
        .sel(dim_ne_tau=reference_ne_tau_ref, method="nearest", tolerance=1e-6 * reference_ne_tau_ref)
    )

    electron_temp = mean_z_curve.dim_electron_temp
    interpolator = InterpolatedUnivariateSpline(electron_temp, magnitude(mean_z_curve))

    def mean_charge_state(electron_temp: float) -> float:
        integrated_mean_z: float = interpolator(electron_temp)
        return integrated_mean_z

    mean_charge_state_integrator = wraps_ufunc(
        input_units=dict(electron_temp=ureg.eV), return_units=dict(mean_charge_state=ureg.dimensionless)
    )(mean_charge_state)
    return mean_charge_state_integrator
