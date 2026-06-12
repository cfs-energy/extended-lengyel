"""Analysis drivers used by the validation notebooks.

Each function wraps one of the extended Lengyel models, propagating the
experimental uncertainties through the model by running it at error factors
of -1, 0 and +1 and collapsing the results into center values with
asymmetric error bars.
"""

from typing import Any

import numpy as np
import xarray as xr
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import Unitfull, ureg

from .impurities import (
    ImpurityInterpolators,
    build_seed_impurities_from_dataset,
    model_interpolator_kwargs,
)
from .models import (
    run_forward_extended_lengyel_model_pdiv,
    run_inverse_extended_lengyel_model,
    run_inverse_extended_lengyel_model_with_pdiv,
)

EF_VALUES = [-1, 0, 1]


def error_factor_coords() -> xr.DataArray:
    """Error-factor coordinate (-1, 0, +1) used to vectorize the uncertainty scan."""
    return xr.DataArray(EF_VALUES, coords=dict(dim_ef=EF_VALUES))


def _collapse_error_factor(data_vars: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Collapse the dim_ef dimension into center values plus asymmetric errors.

    Each variable becomes its dim_ef=0 slice, with ``<name>_err_neg`` / ``<name>_err_pos``
    from the dim_ef=-1 / +1 slices; ``converged`` keeps only the center slice.
    """
    new_data_vars = {}
    for key, val in data_vars.items():
        if key == "converged":
            new_data_vars[key] = val.sel(dim_ef=0)
        else:
            low, center, high = val.sel(dim_ef=-1), val.sel(dim_ef=0), val.sel(dim_ef=1)
            new_data_vars[key] = center
            new_data_vars[f"{key}_err_neg"] = np.abs(center - low)
            new_data_vars[f"{key}_err_pos"] = np.abs(center - high)
    return new_data_vars


def calc_cz(
    experimental_data: xr.Dataset,
    seed: ImpurityInterpolators,
    fixed: ImpurityInterpolators,
    *,
    l_star: float = 1.0,
    l_div_fraction: float = 0.25,
    target_electron_temp: Unitfull = 2.34 * ureg.eV,
    divertor_broadening_factor: float = 1.8,
    triangularity_psi95: float = 0.3,
    target_angle_of_incidence: Unitfull = 3.0 * ureg.degree,
    fraction_of_P_SOL_to_divertor: float = 2.0 / 3.0,
    sheath_heat_transmission_factor: float = 7.5,
    iterations: int = 30,
) -> xr.Dataset:
    """Calculate the seed-impurity concentration required to detach, with uncertainties.

    Runs the inverse extended Lengyel model over the experimental dataset, propagating
    the power and density uncertainties via the error-factor scan.

    Args:
        experimental_data: dataset with the per-discharge plasma parameters and errors.
        seed: seed-impurity interpolators (the concentration solved for).
        fixed: fixed-impurity interpolators (background species).
        l_star: factor increase of parallel connection length, over the usual pi * q * R.
        l_div_fraction: fraction of connection length below the x-point.
        target_electron_temp: sheath entrance electron temperature, 2.34 eV for fmom = 0.5.
        divertor_broadening_factor: ratio of lambda_INT / lambda_q, usually between 2 and 3.
        triangularity_psi95: delta-95.
        target_angle_of_incidence: angle of incidence between magnetic field and divertor target.
        fraction_of_P_SOL_to_divertor: fraction of power to the divertor we're trying to detach.
        sheath_heat_transmission_factor: gamma-sheath, usually around 7.5.
        iterations: number of fixed point iterations to perform. Increase to ensure convergence.
    """
    major_radius = experimental_data["major_radius"]
    connection_length = experimental_data["safety_factor_q95"] * np.pi * major_radius * l_star
    divertor_length = connection_length * l_div_fraction

    error_factor = error_factor_coords()

    # Power and density act in opposite directions, so add one and subtract the other to get the uncertainty.
    power_crossing_separatrix = experimental_data["power_crossing_separatrix"] + error_factor * experimental_data["power_crossing_separatrix_err"]
    separatrix_electron_density = experimental_data["separatrix_electron_density"] - error_factor * experimental_data["separatrix_electron_density_err"]

    (
        c_z,
        _parallel_ion_flux_to_target,
        _neutral_pressure_in_divertor,
        _alpha_t,
        _q_parallel,
        _heat_flux_perp_to_target,
        _separatrix_z_effective,
        converged,
    ) = run_inverse_extended_lengyel_model(
        target_electron_temp=target_electron_temp,
        divertor_broadening_factor=divertor_broadening_factor,
        power_crossing_separatrix=power_crossing_separatrix,
        separatrix_electron_density=separatrix_electron_density,
        magnetic_field_on_axis=experimental_data["magnetic_field_on_axis"],
        plasma_current=experimental_data["plasma_current"],
        parallel_connection_length=connection_length,
        divertor_parallel_length=divertor_length,
        major_radius=major_radius,
        minor_radius=experimental_data["minor_radius"],
        elongation_psi95=experimental_data["elongation"],
        triangularity_psi95=triangularity_psi95,
        target_angle_of_incidence=target_angle_of_incidence,
        fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
        iterations=iterations,
        **model_interpolator_kwargs(seed, fixed),
    )

    if not np.all(converged):
        print("Not all points have converged. Increase the number of iterations.")

    data_vars = dict(
        nitrogen_concentration=c_z,
        converged=converged,
    )

    return xr.Dataset(data_vars=_collapse_error_factor(data_vars))


def calc_impurity_concentration(
    ds: xr.Dataset,
    atomic_data: Any,
    fixed: ImpurityInterpolators,
    *,
    divertor_broadening_factor: float | None = 1.8,
    target_angle_of_incidence: Unitfull = 3.0 * ureg.degree,
    fraction_of_P_SOL_to_divertor: float = 2.0 / 3.0,
    sheath_heat_transmission_factor: float = 7.5,
    iterations: int = 20,
    calc_uncertainty: bool = False,
) -> xr.Dataset:
    """Calculate the per-species impurity concentrations required to detach one discharge.

    Runs the inverse (pdiv-constrained) extended Lengyel model for a single discharge,
    with the seed-impurity mix taken from the measured concentrations. Intended for use
    via ``expt_data.groupby("index").apply(calc_impurity_concentration, ...)``.

    Args:
        ds: single-discharge dataset with plasma parameters, measured concentrations and errors.
        atomic_data: dataset from :func:`extended_lengyel_validation.impurities.load_atomic_data`.
        fixed: fixed-impurity interpolators (background species).
        divertor_broadening_factor: ratio of lambda_INT / lambda_q; None reads it from ``ds``.
        target_angle_of_incidence: angle of incidence between magnetic field and divertor target.
        fraction_of_P_SOL_to_divertor: fraction of power to the divertor we're trying to detach.
        sheath_heat_transmission_factor: gamma-sheath, usually around 7.5.
        iterations: number of fixed point iterations to perform. Increase to ensure convergence.
        calc_uncertainty: propagate the experimental uncertainties via the error-factor scan.
    """
    seed = build_seed_impurities_from_dataset(ds, atomic_data, zero_weight_fallback=True)

    error_factor = error_factor_coords() if calc_uncertainty else 0.0

    if divertor_broadening_factor is None:
        # Allow for setting bdiv via the input ds
        divertor_broadening_factor = ds["divertor_broadening_factor"]

    # Power and density act in opposite directions, so add one and subtract the other to get the uncertainty.
    power_crossing_separatrix = ds["power_crossing_separatrix"] + error_factor * ds["power_crossing_separatrix_err"]
    divertor_neutral_pressure = ds["divertor_neutral_pressure"] - error_factor * ds["divertor_neutral_pressure_err"]

    # To reach a lower target electron temperature, need lower Psep or higher nsep, so error factor should be negative
    target_electron_temp = ds["target_electron_temp"] - error_factor * ds["target_electron_temp_err"]

    (
        c_z,
        separatrix_electron_density,
        _parallel_ion_flux_to_target,
        _alpha_t,
        _q_parallel,
        _heat_flux_perp_to_target,
        _separatrix_z_effective,
        separatrix_electron_temp,
        converged,
    ) = run_inverse_extended_lengyel_model_with_pdiv(
        target_electron_temp=target_electron_temp,
        divertor_broadening_factor=divertor_broadening_factor,
        power_crossing_separatrix=power_crossing_separatrix,
        neutral_pressure_in_divertor=divertor_neutral_pressure,
        magnetic_field_on_axis=ds["magnetic_field_on_axis"],
        plasma_current=ds["plasma_current"],
        parallel_connection_length=ds["parallel_connection_length"],
        divertor_parallel_length=ds["divertor_length"],
        major_radius=ds["major_radius"],
        minor_radius=ds["minor_radius"],
        elongation_psi95=ds["elongation"],
        triangularity_psi95=ds["triangularity"],
        target_angle_of_incidence=target_angle_of_incidence,
        fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
        iterations=iterations,
        **model_interpolator_kwargs(seed, fixed),
    )

    impurity_fraction = seed.weights * c_z

    data_vars = dict(
        c_z_multiplier=c_z,
        impurity_fraction=impurity_fraction,
        separatrix_electron_density=separatrix_electron_density,
        nitrogen_concentration=impurity_fraction.sel(dim_species=AtomicSpecies.Nitrogen),
        neon_concentration=impurity_fraction.sel(dim_species=AtomicSpecies.Neon),
        argon_concentration=impurity_fraction.sel(dim_species=AtomicSpecies.Argon),
        separatrix_electron_temp=separatrix_electron_temp,
        converged=converged,
    )

    data_vars["summed_impurity_concentration"] = (
        data_vars["nitrogen_concentration"] + data_vars["neon_concentration"] + data_vars["argon_concentration"]
    )

    if calc_uncertainty:
        data_vars = _collapse_error_factor(data_vars)

    return xr.Dataset(data_vars=data_vars)


def calc_target_temperature(
    ds: xr.Dataset,
    atomic_data: Any,
    fixed: ImpurityInterpolators,
    *,
    divertor_broadening_factor: float = 1.8,
    target_angle_of_incidence: Unitfull = 3.0 * ureg.degree,
    fraction_of_P_SOL_to_divertor: float = 2.0 / 3.0,
    sheath_heat_transmission_factor: float = 7.5,
    iterations: int = 20,
    calc_uncertainty: bool = False,
) -> xr.Dataset:
    """Calculate the target electron temperature from the measured impurity concentrations.

    Runs the forward (pdiv-constrained) extended Lengyel model for a single discharge.
    Because the seed-impurity weights themselves carry the measured uncertainty, the
    error-factor scan rebuilds the interpolators per error factor rather than
    vectorizing over dim_ef. Intended for use via
    ``expt_data.groupby("index").apply(calc_target_temperature, ...)``.

    Args:
        ds: single-discharge dataset with plasma parameters, measured concentrations and errors.
        atomic_data: dataset from :func:`extended_lengyel_validation.impurities.load_atomic_data`.
        fixed: fixed-impurity interpolators (background species).
        divertor_broadening_factor: ratio of lambda_INT / lambda_q, usually between 2 and 3.
        target_angle_of_incidence: angle of incidence between magnetic field and divertor target.
        fraction_of_P_SOL_to_divertor: fraction of power to the divertor we're trying to detach.
        sheath_heat_transmission_factor: gamma-sheath, usually around 7.5.
        iterations: number of fixed point iterations to perform. Increase to ensure convergence.
        calc_uncertainty: propagate the experimental uncertainties via the error-factor scan.
    """
    ef_values = EF_VALUES if calc_uncertainty else [0]

    target_electron_temp = []
    separatrix_electron_density = []
    converged = []

    # Loop through each error scenario individually
    for ef in ef_values:
        # Adjust inputs based on current error factor
        power_crossing_separatrix = ds["power_crossing_separatrix"] + ef * ds["power_crossing_separatrix_err"]
        divertor_neutral_pressure = ds["divertor_neutral_pressure"] - ef * ds["divertor_neutral_pressure_err"]

        # Seed weights from the measured concentrations, shifted by the current error factor
        seed = build_seed_impurities_from_dataset(ds, atomic_data, error_factor=ef)

        (
            _target_electron_temp,
            _separatrix_electron_density,
            _parallel_ion_flux_to_target,
            _alpha_t,
            _q_parallel,
            _heat_flux_perp_to_target,
            _separatrix_z_effective,
            _converged,
        ) = run_forward_extended_lengyel_model_pdiv(
            impurity_fraction=1.0,
            divertor_broadening_factor=divertor_broadening_factor,
            power_crossing_separatrix=power_crossing_separatrix,
            neutral_pressure_in_divertor=divertor_neutral_pressure,
            magnetic_field_on_axis=ds["magnetic_field_on_axis"],
            plasma_current=ds["plasma_current"],
            parallel_connection_length=ds["parallel_connection_length"],
            divertor_parallel_length=ds["divertor_length"],
            major_radius=ds["major_radius"],
            minor_radius=ds["minor_radius"],
            elongation_psi95=ds["elongation"],
            triangularity_psi95=ds["triangularity"],
            target_angle_of_incidence=target_angle_of_incidence,
            fraction_of_P_SOL_to_divertor=fraction_of_P_SOL_to_divertor,
            sheath_heat_transmission_factor=sheath_heat_transmission_factor,
            iterations=iterations,
            **model_interpolator_kwargs(seed, fixed),
        )

        target_electron_temp.append(_target_electron_temp)
        separatrix_electron_density.append(_separatrix_electron_density)
        converged.append(_converged)

    # Concatenate results back into a single DataArray with the dim_ef dimension
    ef_dim = xr.DataArray(ef_values, name="dim_ef", coords={"dim_ef": ef_values})
    data_vars = dict(
        target_electron_temp=xr.concat(target_electron_temp, dim=ef_dim),
        separatrix_electron_density=xr.concat(separatrix_electron_density, dim=ef_dim),
        converged=xr.concat(converged, dim=ef_dim),
    )

    if calc_uncertainty:
        data_vars = _collapse_error_factor(data_vars)

    return xr.Dataset(data_vars=data_vars)


def calculate_factor_with_uncertainty(expt_val, sim_val, expt_err, sim_err_pos, sim_err_neg):
    """Ratio of experimental to simulated values, with propagated uncertainties.

    If z = a / b, then dz = z * (da / a + db / b). Returns the factor and its
    positive/negative errors, plus the inverse factor and its errors.
    """
    factor = expt_val / sim_val
    err_pos = factor * (expt_err / expt_val + sim_err_pos / sim_val)
    err_neg = factor * (expt_err / expt_val + sim_err_neg / sim_val)

    inverse_factor = 1 / factor
    inverse_err_pos = err_pos / factor**2
    inverse_err_neg = err_neg / factor**2
    return factor, err_pos, err_neg, inverse_factor, inverse_err_pos, inverse_err_neg
