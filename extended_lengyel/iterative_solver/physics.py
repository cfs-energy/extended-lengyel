"""Pure physics step functions shared by the extended Lengyel model variants.

All functions operate on unitless magnitudes with the same conventions as the
original model implementations: SI units except electron temperatures in
electron-volts, electron densities in 10^20 m^-3 (n20) and ion masses in amu
(unless noted otherwise in the docstring).
"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .constants import (
    DENSITY_LOSS_FIT,
    MOMENTUM_LOSS_FIT,
    POWER_LOSS_FIT,
    amu_to_kg,
    boltzmann_constant,
    elementary_charge,
    eV_to_J,
    kappa_e0,
    mu_0,
    n20_to_m3,
)

if TYPE_CHECKING:
    from .solver import ModelInputs


def temperature_fit_function(target_electron_temp: float, amplitude: float, width: float, shape: float) -> float:
    """A general form for functions in terms of the electron temperature at the target.

    Equation 33 from Stangeby, 2018, PPCF 60 044022
    """
    return 1.0 - amplitude * np.power(1.0 - np.exp(-target_electron_temp / width), shape)


def calc_alpha_t(
    separatrix_electron_density: float,
    separatrix_electron_temp: float,
    cylindrical_safety_factor: float,
    major_radius: float,
    average_ion_mass: float,
    z_effective: float,
    mean_ion_charge_state: float,
    ion_to_electron_temp_ratio: float = 1.0,
) -> float:
    """Calculate the turbulence parameter alpha_t.

    Equation 9 from :cite:`Eich_2020`. Compared to this equation, the factor of the
    ion_to_electron_temp_ratio is added following a discussion with T. Eich.

    For separatrix electron density in per-cubic-metre, separatrix electron temp in electron-volts,
    major radius in metres, average ion mass in kilograms, and all other inputs dimensionless.
    """
    epsilon_0 = 5.5263493580350935e+7  # e^2 eV^-1 m^-1
    electron_mass = 9.1093837015e-31  # kg
    eV_to_J = 1.602176634e-19  # Coulomb

    # Calculate the Coulomb logarithm, for electron density in per-cubic-metre and electron temp in electron-volts.
    # From text on page 6 of :cite:`Verdoolaege_2021`
    coulomb_log = 30.9 - np.log(separatrix_electron_density**0.5 * separatrix_electron_temp**-1.0)
    ion_sound_speed = np.sqrt(mean_ion_charge_state * separatrix_electron_temp * eV_to_J / average_ion_mass)

    z_effective_correction = (1.0 - 0.569) * np.exp(-(np.pow(max(z_effective - 1.0, 0.0) / 3.25, 0.85))) + 0.569

    # Calculate the electron-electron collision frequency, using equation B1 from from :cite:`Eich_2020`.
    nu_ee = (
        (4.0 / 3.0)
        * np.sqrt(2.0 * np.pi)
        * separatrix_electron_density
        * coulomb_log
        / ((4.0 * np.pi * epsilon_0) ** 2 * np.sqrt(electron_mass) * separatrix_electron_temp**1.5)
    ) * np.sqrt(eV_to_J)
    # Calculate the electron-ion collision frequency, using equation B2 from from :cite:`Eich_2020`.
    nu_ei = nu_ee * z_effective_correction * z_effective

    alpha_t = (
        1.02
        * nu_ei
        / ion_sound_speed
        * (1.0 * electron_mass / average_ion_mass)
        * cylindrical_safety_factor**2
        * major_radius
        * (1.0 + ion_to_electron_temp_ratio / mean_ion_charge_state)
    )

    return alpha_t


def relax(new_value: float, prev_value: float, relaxation_factor: float = 0.4) -> float:
    """Blend a new iterate with the previous one to stabilize fixed-point iteration."""
    return relaxation_factor * new_value + (1 - relaxation_factor) * prev_value


class ConvectionLayerLosses(NamedTuple):
    """Momentum, power and density loss fractions in the convection layer."""

    momentum: float
    power: float
    density: float


def calc_convection_layer_losses(target_electron_temp: float) -> ConvectionLayerLosses:
    """Evaluate the three convection-layer loss functions at the given target electron temperature (eV)."""
    return ConvectionLayerLosses(
        momentum=temperature_fit_function(target_electron_temp, **MOMENTUM_LOSS_FIT),
        power=temperature_fit_function(target_electron_temp, **POWER_LOSS_FIT),
        density=temperature_fit_function(target_electron_temp, **DENSITY_LOSS_FIT),
    )


def calc_cc_interface_temp(target_electron_temp: float, losses: ConvectionLayerLosses) -> float:
    """Calculate the electron temperature at the convection-conduction interface (eV)."""
    return target_electron_temp / ((1.0 - losses.momentum) / (2.0 * losses.density))


class Geometry(NamedTuple):
    """Loop-invariant magnetic field and geometry quantities."""

    separatrix_average_poloidal_field: float
    cylindrical_safety_factor: float
    fieldline_pitch_at_omp: float
    fraction_of_power_entering_flux_tube: float
    ratio_of_upstream_to_average_lambda_q: float
    parallel_to_perp_factor: float


def calc_geometry(p: "ModelInputs") -> Geometry:
    """Calculate the loop-invariant magnetic field and geometry quantities from the model inputs."""
    shaping_factor = np.sqrt(
        (1.0 + p.elongation_psi95**2 * (1.0 + 2.0 * p.triangularity_psi95**2 - 1.2 * p.triangularity_psi95**3)) / 2.0
    )
    poloidal_circumference = 2.0 * np.pi * p.minor_radius * shaping_factor

    upstream_toroidal_field = p.magnetic_field_on_axis * (p.major_radius / (p.major_radius + p.minor_radius))
    separatrix_average_poloidal_field = mu_0 * p.plasma_current / poloidal_circumference
    upstream_poloidal_field = p.ratio_of_upstream_to_average_poloidal_field * separatrix_average_poloidal_field

    cylindrical_safety_factor = (
        p.magnetic_field_on_axis / separatrix_average_poloidal_field * p.minor_radius / p.major_radius * shaping_factor
    )

    fieldline_pitch_at_omp = np.sqrt(upstream_toroidal_field**2 + upstream_poloidal_field**2) / upstream_poloidal_field

    fraction_of_power_entering_flux_tube = (1.0 - 1.0 / np.e) * p.fraction_of_P_SOL_to_divertor

    ratio_of_upstream_to_average_lambda_q = (
        p.ratio_of_upstream_to_average_poloidal_field * (p.major_radius + p.minor_radius) / p.major_radius
    )

    return Geometry(
        separatrix_average_poloidal_field=separatrix_average_poloidal_field,
        cylindrical_safety_factor=cylindrical_safety_factor,
        fieldline_pitch_at_omp=fieldline_pitch_at_omp,
        fraction_of_power_entering_flux_tube=fraction_of_power_entering_flux_tube,
        ratio_of_upstream_to_average_lambda_q=ratio_of_upstream_to_average_lambda_q,
        parallel_to_perp_factor=np.sin(p.target_angle_of_incidence),
    )


def calc_z_effective(
    impurity_fraction: float, electron_temp_eV: float, p: "ModelInputs", starting_z_effective: float = 1.0
) -> float:
    """Calculate the effective charge from the seed and fixed impurity content at the given electron temperature."""
    seed_mean_z = p.mean_charge_for_seed_impurities.unitless_eval(electron_temp_eV)
    fixed_mean_z = p.mean_charge_for_fixed_impurities.unitless_eval(electron_temp_eV)
    seed_c_z = impurity_fraction * p.CzLINT_for_seed_impurities.weights
    fixed_c_z = p.CzLINT_for_fixed_impurities.weights
    z_effective = (
        starting_z_effective
        + (seed_mean_z * (seed_mean_z - 1.0) * seed_c_z).sum(dim="dim_species")
        + (fixed_mean_z * (fixed_mean_z - 1.0) * fixed_c_z).sum(dim="dim_species")
    )
    return z_effective.values


def calc_q_parallel(
    separatrix_electron_temp: float, alpha_t: float, geo: Geometry, p: "ModelInputs", clamp_alpha_t: bool
) -> float:
    """Calculate the parallel heat flux entering the flux tube, consistent with alpha_t and the separatrix electron temperature."""
    separatrix_average_rho_s_pol = (
        np.sqrt(separatrix_electron_temp * p.average_ion_mass)
        / geo.separatrix_average_poloidal_field
        * np.sqrt(amu_to_kg / elementary_charge)
    )  # in metres, for Te in eV, mi in amu and B0 in T

    if clamp_alpha_t:
        alpha_t = np.maximum(alpha_t, 0.0)
    separatrix_average_lambda_Te = 2.1 * (1 + 2.1 * alpha_t**1.7) * separatrix_average_rho_s_pol
    separatrix_average_lambda_q = 2.0 / 7.0 * separatrix_average_lambda_Te

    lambda_q_outboard_midplane = separatrix_average_lambda_q / geo.ratio_of_upstream_to_average_lambda_q  # in metres

    return (
        p.power_crossing_separatrix
        * geo.fraction_of_power_entering_flux_tube
        / (2.0 * np.pi * (p.major_radius + p.minor_radius) * lambda_q_outboard_midplane)
        * geo.fieldline_pitch_at_omp
    )  # in watts per metres-squared


def calc_kappa_e(divertor_z_effective: float) -> float:
    """Calculate the impurity-corrected electron heat conductivity coefficient.

    Uses equation 10 from Brown and Goldston, 2021, NME 27 101002.
    """
    kappa_z = 0.672 + 0.076 * np.sqrt(divertor_z_effective) + 0.252 * divertor_z_effective
    return kappa_e0 / kappa_z


def calc_temperature_ladder(
    electron_temp_at_cc_interface: float, q_parallel: float, kappa_e: float, p: "ModelInputs"
) -> tuple[float, float]:
    """Integrate the conduction equation to get the divertor entrance and separatrix electron temperatures (eV)."""
    divertor_entrance_electron_temp = (
        electron_temp_at_cc_interface**3.5
        + 3.5 * p.SOL_conduction_fraction * q_parallel / p.divertor_broadening_factor * p.divertor_parallel_length / kappa_e
    ) ** (2.0 / 7.0)  # in electron-volts

    separatrix_electron_temp = (
        divertor_entrance_electron_temp**3.5
        + 3.5 * p.SOL_conduction_fraction * q_parallel * (p.parallel_connection_length - p.divertor_parallel_length) / kappa_e
    ) ** (2.0 / 7.0)  # in electron-volts

    return divertor_entrance_electron_temp, separatrix_electron_temp


def calc_separatrix_total_pressure(
    separatrix_electron_density: float, separatrix_electron_temp: float, p: "ModelInputs"
) -> float:
    """Calculate the separatrix total pressure in Pascals, for density in n20 and temperature in eV."""
    return (
        (1.0 + p.separatrix_mach_number**2)
        * separatrix_electron_density
        * separatrix_electron_temp
        * (1.0 + p.separatrix_ratio_of_ion_to_electron_temp / p.separatrix_ratio_of_electron_to_ion_density)
    ) * (n20_to_m3 * elementary_charge)  # in Pascals


def calc_target_electron_temp_basic(q_parallel: float, separatrix_total_pressure: float, p: "ModelInputs") -> float:
    """Calculate the basic two-point-model target electron temperature (eV), before loss corrections."""
    return (
        (8.0 * p.average_ion_mass / p.sheath_heat_transmission_factor**2) * (q_parallel**2 / separatrix_total_pressure**2)
    ) * amu_to_kg / elementary_charge


def calc_f_other(p: "ModelInputs") -> float:
    """Calculate the combined target-condition correction factor for the two-point model."""
    return (
        ((1.0 + p.target_ratio_of_ion_to_electron_temp / p.target_ratio_of_electron_to_ion_density) / 2.0)
        * ((1.0 + p.target_mach_number**2) ** 2 / (4.0 * p.target_mach_number**2))
        * p.toroidal_flux_expansion**-2
    )


class LengyelIntegrals(NamedTuple):
    """Radiated-power integrals for the seed (Ls) and fixed (Lf) impurities between temperature points."""

    Ls_cc_div: float
    Ls_div_u: float
    Lf_cc_div: float
    Lf_div_u: float


def calc_lengyel_integrals(
    electron_temp_at_cc_interface: float,
    divertor_entrance_electron_temp: float,
    separatrix_electron_temp: float,
    p: "ModelInputs",
) -> LengyelIntegrals:
    """Evaluate the Lengyel radiated-power integrals between the cc-interface, divertor entrance and separatrix temperatures."""
    T_cc, T_div, T_u = electron_temp_at_cc_interface, divertor_entrance_electron_temp, separatrix_electron_temp

    return LengyelIntegrals(
        # Seed impurities
        Ls_cc_div=p.CzLINT_for_seed_impurities.unitless_eval(T_cc, T_div) * n20_to_m3**2,
        Ls_div_u=p.CzLINT_for_seed_impurities.unitless_eval(T_div, T_u) * n20_to_m3**2,
        # Fixed impurities
        Lf_cc_div=p.CzLINT_for_fixed_impurities.unitless_eval(T_cc, T_div) * n20_to_m3**2,
        Lf_div_u=p.CzLINT_for_fixed_impurities.unitless_eval(T_div, T_u) * n20_to_m3**2,
    )


class TargetPostProcessing(NamedTuple):
    """Target quantities derived from the target electron temperature and parallel heat flux."""

    parallel_ion_flux_to_target: float
    neutral_pressure_in_divertor: float
    heat_flux_perp_to_target: float


def calc_target_postprocessing(
    target_electron_temp: float, parallel_heat_flux_at_target: float, geo: Geometry, p: "ModelInputs"
) -> TargetPostProcessing:
    """Calculate the ion flux, divertor neutral pressure and perpendicular heat flux at the target."""
    sound_speed_at_target = np.sqrt(2.0 * target_electron_temp * (eV_to_J / amu_to_kg) / p.average_ion_mass)  # m / s

    electron_density_at_target = parallel_heat_flux_at_target / (
        p.sheath_heat_transmission_factor * target_electron_temp * eV_to_J * sound_speed_at_target
    )  # m^-3

    # From equation 57 of Body, Kallenbach and Eich, NF 2025
    flux_density_to_pascals_factor = np.sqrt(
        2.0 / (np.pi * p.ratio_of_molecular_to_ion_mass * p.average_ion_mass * p.wall_temperature)
    ) / np.sqrt(amu_to_kg * boltzmann_constant)  # (m**-2 / s) / Pa

    parallel_ion_flux_to_target = electron_density_at_target * sound_speed_at_target  # m**-2 / s
    neutral_pressure_in_divertor = parallel_ion_flux_to_target * geo.parallel_to_perp_factor / flux_density_to_pascals_factor  # Pa

    heat_flux_perp_to_target = parallel_heat_flux_at_target * geo.parallel_to_perp_factor  # W/m^2

    return TargetPostProcessing(
        parallel_ion_flux_to_target=parallel_ion_flux_to_target,
        neutral_pressure_in_divertor=neutral_pressure_in_divertor,
        heat_flux_perp_to_target=heat_flux_perp_to_target,
    )
