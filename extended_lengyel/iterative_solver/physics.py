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
    impurity_fraction: float,
    electron_temp_eV: float,
    p: "ModelInputs",
    starting_z_effective: float = 1.0,
    enrichment_factor: float = 1.0,
) -> float:
    """Calculate the effective charge from the seed and fixed impurity content at the given electron temperature.

    Args:
        impurity_fraction: seed-impurity concentration multiplier (the divertor concentration c_z).
        electron_temp_eV: electron temperature at which to evaluate the mean charge states.
        p: model inputs (provides the seed/fixed mean-charge interpolators and concentration weights).
        starting_z_effective: baseline Z_eff of the background plasma (1.0 for pure hydrogenic).
        enrichment_factor: divertor enrichment f_enrich. All (seed and fixed) impurity concentrations
            are divided by it, so passing ``p.divertor_enrichment_factor`` yields the *upstream*
            (main-chamber) Z_eff, where the concentration is c_z / f_enrich; the default 1.0 yields the
            divertor Z_eff (full concentration).
    """
    seed_mean_z = p.mean_charge_for_seed_impurities.unitless_eval(electron_temp_eV)
    fixed_mean_z = p.mean_charge_for_fixed_impurities.unitless_eval(electron_temp_eV)
    seed_c_z = impurity_fraction * p.CzLINT_for_seed_impurities.weights / enrichment_factor
    fixed_c_z = p.CzLINT_for_fixed_impurities.weights / enrichment_factor
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


def calc_heat_conductivity_factor(z_effective: float) -> float:
    """Calculate the impurity-corrected electron heat-conductivity factor kappa_e0 / kappa_z.

    kappa_z is the Z_eff correction to the Spitzer-Harm conductivity (equation 10 from Brown and
    Goldston, 2021, NME 27 101002); the returned factor kappa_e0 / kappa_z is the quantity that
    enters the conduction integral and the Lengyel power balance.
    """
    kappa_z = 0.672 + 0.076 * np.sqrt(z_effective) + 0.252 * z_effective
    return kappa_e0 / kappa_z


def calc_temperature_ladder(
    electron_temp_at_cc_interface: float,
    q_parallel: float,
    divertor_heat_conductivity_factor: float,
    upstream_heat_conductivity_factor: float,
    p: "ModelInputs",
) -> tuple[float, float]:
    """Integrate the conduction equation to get the divertor entrance and separatrix electron temperatures (eV).

    Args:
        electron_temp_at_cc_interface: electron temperature at the convection-conduction interface (eV).
        q_parallel: parallel heat flux entering the flux tube (W/m^2).
        divertor_heat_conductivity_factor: heat-conductivity factor (kappa_e0/kappa_z) for the
            cc->divertor (divertor) leg, from the divertor Z_eff.
        upstream_heat_conductivity_factor: heat-conductivity factor (kappa_e0/kappa_z) for the
            divertor->upstream (main-chamber) leg, from the upstream Z_eff.
        p: model inputs (connection lengths, SOL conduction fraction, heat-flux broadening,
            flux-expansion field ratios).
    """
    # Flux-expansion factors: the conductive flux is anchored upstream and scales as q ~ B along
    # each leg (q_parallel / B conserved), so the leg-mean flux entering the conduction integral is
    # q_parallel times the leg-mean of B / B_u. For a constant field gradient (B linear in s) that
    # mean is the average of the leg's endpoint field ratios. Both reduce to 1 at the default ratios.
    fx_t, fx_xpt = p.flux_expansion_Bt_over_Bu, p.flux_expansion_Bxpt_over_Bu
    cc_div_flux_expansion_factor = 0.5 * (fx_t + fx_xpt)
    div_u_flux_expansion_factor = 0.5 * (fx_xpt + 1.0)

    divertor_entrance_electron_temp = (
        electron_temp_at_cc_interface**3.5
        + 3.5
        * p.SOL_conduction_fraction
        * q_parallel
        / p.divertor_broadening_factor
        * cc_div_flux_expansion_factor
        * p.divertor_parallel_length
        / divertor_heat_conductivity_factor
    ) ** (2.0 / 7.0)  # in electron-volts

    separatrix_electron_temp = (
        divertor_entrance_electron_temp**3.5
        + 3.5
        * p.SOL_conduction_fraction
        * q_parallel
        * div_u_flux_expansion_factor
        * (p.parallel_connection_length - p.divertor_parallel_length)
        / upstream_heat_conductivity_factor
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
        * p.flux_expansion_Bt_over_Bu**2
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


def calc_tB(
    q_parallel: float,
    k_base: float,
    upstream_heat_conductivity_factor: float,
    Lint_div_u: float,
    divertor_enrichment_factor: float,
    flux_expansion_Bt_over_Bu: float,
    flux_expansion_Bxpt_over_Bu: float,
) -> float:
    """Calculate the magnetic-flux-expansion correction t_B for the Lengyel power balance.

    The field strength varies along the field line, so the conserved conductive quantity is
    q_parallel / B rather than q_parallel. Integrating the conservative field-line energy
    equation introduces ``t_B = - integral (q_parallel**2 / B) dB`` (Moulton et al. 2021,
    NF 61 046029, eqs 10-11), which is zero when B is constant. The upstream parallel power
    balance combines the two conduction legs as ``q_||u^2 - b_q^2 q_cc^2 + 2 t_B``, with the
    combined correction ``t_B = b_q^2 t_B^{cc->div} + t_B^{div->u}`` (the cc->div leg carries
    the same b_q^2 as its radiation integral).

    Assumptions (see the appendix of the manuscript). t_B is evaluated in the upstream-referenced
    flux ``q_up = q_parallel * B_u / B`` (which removes the flux expansion, so q_up = q_||u
    everywhere in the no-radiation limit), giving ``t_B^{a->b} = -<q_up^2> (B_b^2 - B_a^2) / (2
    B_u^2)`` (angle brackets: along-leg average) with only radiation varying q_up along the
    field line:

    - **Radiation reduces the flux.** The X-point value follows from the div->u radiation,
      ``q_xpt^2 = q_||u^2 - k_base * (kappa_e0/kappa_z)_u * L_INT^{div->u} / f_enrich`` (the
      upstream balance with t_B itself neglected -- a correction to a correction).
    - **Per-leg representative flux.** The div->u leg (radiation weak at high T) uses the mean of
      its squared endpoints ``0.5 (q_xpt^2 + q_||u^2)``; the cc->div leg (radiation peaks near the
      target, so q_up stays near its X-point value ``q_xpt / b_q`` over most of the leg) uses
      ``q_xpt^2``.
    - **Reference field B_u, not the local B_xpt.** Both legs divide by B_u, consistently with how
      q_xpt is defined (from the flux-expansion-free balance above); referencing the divertor leg
      to the local B_xpt would double-count the flux expansion and break the no-radiation limit.

    This is exact in the no-radiation limit (q_xpt -> q_||u gives 0.5 q_||u^2 ((B_t/B_u)^2 - 1))
    and agrees with the spatial Lengyel reference (``spatial_lengyel.py``) to within ~2.5% for
    B_t/B_u up to ~2. NOTE it uses the radiation-reduced flux whereas the conduction ladder
    (``calc_temperature_ladder``) uses the radiation-free q_||u, so the two are not strictly
    consistent -- accepted because the conduction integral depends only weakly on q_parallel
    (through the 7/2 power) while t_B is quadratic in it. A temperature-integral treatment (or
    reducing q_parallel for radiation in the ladder too) would remove both the residual and the
    inconsistency.

    Args:
        q_parallel: upstream parallel heat flux q_||u (W/m^2).
        k_base: ``2 * separatrix_electron_density**2 * separatrix_electron_temp**2`` (density in n20).
        upstream_heat_conductivity_factor: kappa_e0/kappa_z for the div->upstream leg (upstream Z_eff).
        Lint_div_u: impurity-weighted (seed + fixed) radiation integral over the div->upstream leg.
        divertor_enrichment_factor: divertor enrichment f_enrich.
        flux_expansion_Bt_over_Bu: target/upstream field-strength ratio B_t/B_u.
        flux_expansion_Bxpt_over_Bu: X-point/upstream field-strength ratio B_xpt/B_u.

    Returns:
        The flux-expansion correction t_B (W^2/m^4) for the ``q_||u^2 - b_q^2 q_cc^2 + 2 t_B``
        numerator; zero when both field ratios are 1.0.
    """
    q_parallel_squared = q_parallel**2
    # Upstream-referenced X-point heat flux from the divertor->upstream radiation.
    qxpt_up_squared = max(q_parallel_squared - k_base * upstream_heat_conductivity_factor * Lint_div_u / divertor_enrichment_factor, 0.0)
    mean_q2_div_u = 0.5 * (qxpt_up_squared + q_parallel_squared)

    fx_t, fx_xpt = flux_expansion_Bt_over_Bu, flux_expansion_Bxpt_over_Bu
    return mean_q2_div_u * 0.5 * (fx_xpt**2 - 1.0) + qxpt_up_squared * 0.5 * (fx_t**2 - fx_xpt**2)


class TargetPostProcessing(NamedTuple):
    """Target quantities derived from the target electron temperature and parallel heat flux."""

    parallel_ion_flux_to_target: float
    neutral_pressure_in_divertor: float
    heat_flux_perp_to_target: float


def calc_target_postprocessing(
    target_electron_temp: float, parallel_heat_flux_at_target: float, geo: Geometry, p: "ModelInputs"
) -> TargetPostProcessing:
    """Calculate the ion flux, divertor neutral pressure and perpendicular heat flux at the target.

    ``parallel_heat_flux_at_target`` is the actual parallel heat flux at the target sheath, already
    carrying the B_t/B_u field-strength amplification (applied in the solver), so it feeds the sheath
    relations directly.
    """
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
