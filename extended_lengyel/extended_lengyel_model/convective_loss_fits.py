"""Apply correction factors for the convective region losses, based on fits to the Kallenbach model."""

from pathlib import Path

import numpy as np
import yaml
from cfspopcon import declare_algorithm
from cfspopcon.unit_handling import ureg

from ..directories import module_directory

CURVE_FIT_FILE = module_directory / "curve_fit.yml"
# The fitted quantities: the momentum and power loss in the convective layer, and the density ratio across it.
FIT_NAMES = ("fmom", "fpow", "fdens")


def temperature_fit_function(target_electron_temp, amplitude, width, shape):
    """A general form for functions in terms of the electron temperature at the target.

    Equation 33 from Stangeby, 2018, PPCF 60 044022
    """
    return 1.0 - amplitude * np.power(1.0 - np.exp(-target_electron_temp / width), shape)


def fit_coefficients_key(name: str) -> str:
    """The key under which the fit called ``name`` (one of ``FIT_NAMES``) is stored in ``curve_fit.yml``."""
    return f"{name}_coeffs"


def read_fit_coefficients(fit_file: Path = CURVE_FIT_FILE) -> dict[str, dict[str, float]]:
    """Read the stored fit coefficients, keyed by ``fit_coefficients_key(name)`` for each name in ``FIT_NAMES``.

    The file is read on every call, so a regenerated ``curve_fit.yml`` is picked up without restarting.
    """
    with open(fit_file) as file:
        return yaml.safe_load(file)


def fit_parameters(name: str, fit_file: Path = CURVE_FIT_FILE) -> dict:
    """The arguments of ``temperature_fit_function`` (besides the temperature) for the stored fit called ``name``."""
    coefficients = read_fit_coefficients(fit_file)[fit_coefficients_key(name)]
    return dict(amplitude=coefficients["amplitude"], width=coefficients["width"] * ureg.eV, shape=coefficients["shape"])


@declare_algorithm(return_keys=["SOL_momentum_loss_fraction"])
def calc_momentum_loss_from_cc_fit(target_electron_temp):
    """Calculate the momentum loss function, from a fit to the Kallenbach model."""
    return temperature_fit_function(target_electron_temp=target_electron_temp, **fit_parameters("fmom"))


@declare_algorithm(return_keys=["SOL_power_loss_fraction_in_convection_layer"])
def calc_power_loss_from_cc_fit(target_electron_temp):
    """Calculate the fraction of power lost in the convection layer, from a fit to the Kallenbach model."""
    return temperature_fit_function(target_electron_temp=target_electron_temp, **fit_parameters("fpow"))


@declare_algorithm(return_keys=["electron_temp_at_cc_interface"])
def calc_electron_temp_from_cc_fit(target_electron_temp):
    """Calculate the electron_temp at the convection-conduction boundary, from fits to the Kallenbach model results."""
    fdens = temperature_fit_function(target_electron_temp=target_electron_temp, **fit_parameters("fdens"))
    fmom = temperature_fit_function(target_electron_temp=target_electron_temp, **fit_parameters("fmom"))

    electron_temp_at_cc_interface = target_electron_temp / ((1 - fmom) / (2.0 * fdens))

    return electron_temp_at_cc_interface


@declare_algorithm(return_keys=["electron_density_at_cc_interface"])
def calc_electron_density_from_cc_fit(target_electron_temp, electron_density_at_target):
    """Calculate the electron_density at the convection-conduction boundary, from fits to the Kallenbach model results."""
    fdens = temperature_fit_function(target_electron_temp=target_electron_temp, **fit_parameters("fdens"))

    electron_density_at_cc_interface = electron_density_at_target / fdens

    return electron_density_at_cc_interface


@declare_algorithm(return_keys=["parallel_heat_flux_at_cc_interface"])
def calc_parallel_heat_flux_from_conv_loss(parallel_heat_flux_at_target, SOL_power_loss_fraction_in_convection_layer):
    """Calculate the parallel_heat_flux at the convection-conduction boundary, from fits to the Kallenbach model results."""
    parallel_heat_flux_at_cc_interface = parallel_heat_flux_at_target / (1 - SOL_power_loss_fraction_in_convection_layer)

    return parallel_heat_flux_at_cc_interface


@declare_algorithm(return_keys=["s_parallel_at_cc_interface"])
def ignore_s_parallel_width_for_cc_interface():
    """Ignore the width of the convective layer."""
    s_parallel_at_cc_interface = 0.0 * ureg.m
    return s_parallel_at_cc_interface


@declare_algorithm(return_keys=["SOL_power_loss_fraction_in_convection_layer"])
def ignore_power_loss_in_convection_layer():
    """Ignore the change in heat flux density across the convection layer."""
    return 0.0


@declare_algorithm(return_keys=["electron_temp_at_cc_interface"])
def ignore_temp_ratio_in_convection_layer(target_electron_temp):
    """Ignore the change in electron temp across the convection layer."""
    return target_electron_temp


@declare_algorithm(return_keys=["electron_density_at_cc_interface"])
def ignore_density_ratio_in_convection_layer(electron_density_at_target):
    """Ignore the change in electron density across the convection layer."""
    return electron_density_at_target


@declare_algorithm(return_keys=["target_electron_temp"])
def calc_target_electron_temp_from_cc_fit(SOL_momentum_loss_fraction):
    """Compute the electron temperature required to reach a given momentum loss fraction."""
    fmom = fit_parameters("fmom")
    amplitude, width, shape = fmom["amplitude"], fmom["width"], fmom["shape"]

    target_electron_temp = -width * np.log(1 - np.power((1 - SOL_momentum_loss_fraction) / amplitude, 1 / shape))

    return target_electron_temp
