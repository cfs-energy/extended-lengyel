"""Fit the convective-layer loss curves, and (re)write ``curve_fit.yml``.

The fits are curves of the form given in Equation 33 from Stangeby, 2018, PPCF 60 044022

    f = 1.0 - A [1.0 - exp(-T_e,t / w)]^S

and are determined from a scan of the Kallenbach model over the target electron temperature.
``curve_fit.yml`` is only rewritten by running this module's command line interface

    xlengyel-fit-convective-layer

so running the notebooks does not change the shipped coefficients.
"""

import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
import cfspopcon
from cfspopcon.unit_handling import magnitude_in_units, ureg
from scipy.optimize import curve_fit

from . import config
from .directories import notebook_dir
from .extended_lengyel_model.convective_loss_fits import CURVE_FIT_FILE, FIT_NAMES, fit_coefficients_key, temperature_fit_function

DEFAULT_OUTPUT_FILE = CURVE_FIT_FILE
# The dataset variable which each fit (see FIT_NAMES) is fitted to.
FITTED_VARIABLES = dict(
    fmom="SOL_momentum_loss_fraction_in_convection_layer",
    fpow="SOL_power_loss_fraction_in_convection_layer",
    fdens="electron_density_ratio_in_convection_layer",
)
# The notebooks are only present in a source checkout, so the default config is looked up at run time.
DEFAULT_CONFIG_FILE = notebook_dir / "config.yml"


def determine_temperature_fit_coefficients(x, y, info: bool = False) -> tuple[dict[str, float], Callable]:
    """Fit a temperature-fit function to the data."""
    popt, pcov = curve_fit(
        temperature_fit_function,
        xdata=x.where(~np.isnan(y), drop=True),
        ydata=y.where(~np.isnan(y), drop=True),
    )

    p_sigma = np.sqrt(np.diag(pcov))

    if info:
        click.echo(f"Covariance matrix condition number: {np.linalg.cond(pcov)}, diagonal elements: {np.diag(pcov)}")
        click.echo(
            f"amplitude: {popt[0]:.3g} \\pm {p_sigma[0]:.1g}\n"
            f"width: {popt[1]:.3g} \\pm {p_sigma[1]:.1g}\n"
            f"shape: {popt[2]:.3g} \\pm {p_sigma[2]:.1g}"
        )

    coeffs = dict(
        amplitude=float(popt[0]),
        width=float(popt[1]),
        shape=float(popt[2]),
        amplitude_uncertainty=float(p_sigma[0]),
        width_uncertainty=float(p_sigma[1]),
        shape_uncertainty=float(p_sigma[2]),
    )

    return coeffs, lambda x: temperature_fit_function(x, *popt)


def determine_fmomc_and_fpowc_fit_functions(ds: xr.Dataset, info: bool = False) -> tuple[dict, dict[str, Callable]]:
    """Determine fit functions for the momentum, power and density loss in the convective layer near the target.

    Returns the fit coefficients (in the layout of ``curve_fit.yml``) and a dictionary of the fitted functions.
    """
    x = magnitude_in_units(ds["target_electron_temp"], ureg.eV)

    fits = {}
    coeffs = {}
    for name in FIT_NAMES:
        if info:
            click.echo(f"Fitting {name}")
        coeffs[fit_coefficients_key(name)], fits[name] = determine_temperature_fit_coefficients(
            x, magnitude_in_units(ds[FITTED_VARIABLES[name]], ureg.dimensionless), info=info
        )

    return coeffs, fits


def build_reference_scan_dataset(config_file: Path = DEFAULT_CONFIG_FILE) -> xr.Dataset:
    """Run the Kallenbach model to the convection-conduction interface, scanning over the target electron temperature."""
    cfspopcon.register_plugin("extended_lengyel")
    algorithm = cfspopcon.registry["kallenbach_model_to_cc"]

    ds = xr.Dataset(
        data_vars=config.read_config(
            filepath=Path(config_file),
            elements=["base", "target_constraints"],
            keys=algorithm.input_keys,
            allowed_missing=algorithm.default_keys,
            overrides=dict(
                heat_flux_perp_to_target=1.0 * ureg.MW / ureg.m**2,
                target_electron_temp=config.promote_to_coordinate(
                    np.logspace(np.log10(1.0), np.log10(50), num=100), ureg.eV, dims="target_electron_temp"
                ),
                impurity_fraction=1.0 * ureg.percent,
            ),
        )
    )

    algorithm.validate_inputs(ds)
    return algorithm.update_dataset(ds)


def count_failed_scan_points(ds: xr.Dataset) -> tuple[int, int]:
    """Count the scan points where the Kallenbach model failed (and so gave NaN), and the total number of points."""
    failed = ds[FITTED_VARIABLES["fmom"]].isnull()
    return int(failed.sum()), int(failed.size)


def find_non_finite_coefficients(coefficients: dict[str, dict[str, float]]) -> list[str]:
    """List the entries of the fit coefficients (with their uncertainties) which are not finite numbers."""
    return [f"{group}.{name}" for group, values in coefficients.items() for name, value in values.items() if not math.isfinite(value)]


def write_fit_coefficients(coefficients: dict, output_file: Path = DEFAULT_OUTPUT_FILE) -> None:
    """Write the fit coefficients to a YAML file.

    The file is written in full to a temporary file next to the output, and then moved into place, so an error
    part way through never leaves a truncated or half-written output file.
    """
    output_file = Path(output_file)
    with tempfile.NamedTemporaryFile("w", dir=output_file.parent, suffix=".tmp", delete=False) as outfile:
        temporary_file = Path(outfile.name)
        try:
            yaml.dump(coefficients, outfile)
        except BaseException:
            temporary_file.unlink(missing_ok=True)
            raise
    # NamedTemporaryFile is private to the owner, so keep the permissions of any file being replaced.
    temporary_file.chmod(output_file.stat().st_mode & 0o777 if output_file.exists() else 0o644)
    os.replace(temporary_file, output_file)


@click.command()
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=f"Config file providing the 'base' and 'target_constraints' elements for the Kallenbach scan. Defaults to {DEFAULT_CONFIG_FILE} (source checkouts only).",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT_FILE,
    show_default=True,
    help="YAML file to write the fit coefficients to.",
)
@click.option("--info", is_flag=True, help="Print the fit covariance and the fitted coefficients.")
def fit_convective_layer_cli(config_file: Path | None, output_file: Path, info: bool) -> None:
    """Fit the convective-layer loss curves and write them to curve_fit.yml."""
    if config_file is None:
        if not DEFAULT_CONFIG_FILE.exists():
            raise click.UsageError(f"{DEFAULT_CONFIG_FILE} not found. Pass --config-file (the notebooks are only in a source checkout).")
        config_file = DEFAULT_CONFIG_FILE
    ds = build_reference_scan_dataset(config_file)

    failed_points, total_points = count_failed_scan_points(ds)
    if failed_points:
        click.echo(
            f"Warning: the Kallenbach model failed at {failed_points} of {total_points} scan points, which are left out of the fit.",
            err=True,
        )

    coefficients, _ = determine_fmomc_and_fpowc_fit_functions(ds, info=info)

    if non_finite := find_non_finite_coefficients(coefficients):
        raise click.ClickException(f"The fit gave non-finite values for {', '.join(non_finite)}. Nothing was written to {output_file}.")

    write_fit_coefficients(coefficients, output_file)
    click.echo(f"Wrote convective-layer fit coefficients to {output_file}.")


if __name__ == "__main__":
    fit_convective_layer_cli()
