"""Uses the extended Lengyel model to calculate the impurity concentration required to reach a fixed target electron temperature."""

import numpy as np
from typing import Any, Optional
from cfspopcon.unit_handling import wraps_ufunc, ureg, Unitfull
from .Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator

np.seterr(over="raise",under="raise")

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
    epsilon_0 = 5.5263493580350935e+7 # e^2 eV^-1 m^-1
    electron_mass = 9.1093837015e-31 # kg
    eV_to_J = 1.602176634e-19 # Coulomb

    # Calculate the Coulomb logarithm, for electron density in per-cubic-metre and electron temp in electron-volts.
    # From text on page 6 of :cite:`Verdoolaege_2021`
    coulomb_log = 30.9 - np.log(separatrix_electron_density**0.5 * separatrix_electron_temp**-1.0)
    ion_sound_speed = np.sqrt(mean_ion_charge_state * separatrix_electron_temp * eV_to_J / average_ion_mass)

    z_effective_correction = (1.0 - 0.569) * np.exp(-(np.pow((z_effective - 1.0) / 3.25, 0.85))) + 0.569

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

@wraps_ufunc(
    input_units=dict(
        target_electron_temp = ureg.eV,
        power_crossing_separatrix = ureg.MW,
        separatrix_electron_density = ureg.m**-3,
        divertor_broadening_factor = ureg.dimensionless,
        CzLINT_for_seed_impurities = None,
        mean_charge_for_seed_impurities = None,
        magnetic_field_on_axis = ureg.T,
        plasma_current = ureg.MA,
        parallel_connection_length = ureg.m,
        divertor_parallel_length = ureg.m,
        major_radius = ureg.m,
        minor_radius = ureg.m,
        elongation_psi95 = ureg.dimensionless,
        triangularity_psi95 = ureg.dimensionless,
        target_angle_of_incidence = ureg.degree,
        fraction_of_P_SOL_to_divertor = ureg.dimensionless,
        CzLINT_for_fixed_impurities = None,
        mean_charge_for_fixed_impurities = None,
        average_ion_mass = ureg.amu,
        sheath_heat_transmission_factor = ureg.dimensionless,
        ratio_of_upstream_to_average_poloidal_field = ureg.dimensionless,
        wall_temperature = ureg.K,
        SOL_conduction_fraction = ureg.dimensionless,
        ratio_of_molecular_to_ion_mass = ureg.dimensionless,
        separatrix_mach_number = ureg.dimensionless,
        separatrix_ratio_of_ion_to_electron_temp = ureg.dimensionless,
        separatrix_ratio_of_electron_to_ion_density = ureg.dimensionless,
        target_ratio_of_ion_to_electron_temp = ureg.dimensionless,
        target_ratio_of_electron_to_ion_density = ureg.dimensionless,
        target_mach_number = ureg.dimensionless,
        toroidal_flux_expansion = ureg.dimensionless,
        iterations = None,
    ),
    return_units=dict(
        c_z = ureg.dimensionless,
        parallel_ion_flux_to_target = ureg.m**-2 * ureg.s**-1,
        neutral_pressure_in_divertor = ureg.Pa,
        alpha_t = ureg.dimensionless,
        q_parallel = ureg.W / ureg.m**2,
        heat_flux_perp_to_target = ureg.W / ureg.m**2,
        separatrix_z_effective = ureg.dimensionless,
        converged = None,
    ),
    output_core_dims = ((), (), (), (), (), (), (), ()),
)
def run_inverse_extended_lengyel_model(
    target_electron_temp: Unitfull,
    power_crossing_separatrix: Unitfull,
    separatrix_electron_density: Unitfull,
    divertor_broadening_factor: Unitfull,
    CzLINT_for_seed_impurities: CzLINT_integrator,
    mean_charge_for_seed_impurities: Mean_charge_interpolator,
    magnetic_field_on_axis: Unitfull,
    plasma_current: Unitfull,
    parallel_connection_length: Unitfull,
    divertor_parallel_length: Unitfull,
    major_radius: Unitfull,
    minor_radius: Unitfull,
    elongation_psi95: Unitfull,
    triangularity_psi95: Unitfull,
    target_angle_of_incidence: Unitfull,
    fraction_of_P_SOL_to_divertor: Unitfull,
    CzLINT_for_fixed_impurities: Optional[CzLINT_integrator] = None,
    mean_charge_for_fixed_impurities: Optional[Mean_charge_interpolator] = None,
    average_ion_mass: Unitfull = 2.0 * ureg.amu,
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    ratio_of_upstream_to_average_poloidal_field: Unitfull = 4./3. * ureg.dimensionless,
    wall_temperature: Unitfull = 300.0 * ureg.K,
    SOL_conduction_fraction: Unitfull = 1.0 * ureg.dimensionless,
    ratio_of_molecular_to_ion_mass: Unitfull = 2.0 * ureg.dimensionless,
    separatrix_mach_number: float = 0.0,
    separatrix_ratio_of_ion_to_electron_temp: float = 1.0,
    separatrix_ratio_of_electron_to_ion_density: float = 1.0,
    target_ratio_of_ion_to_electron_temp: float = 1.0,
    target_ratio_of_electron_to_ion_density: float = 1.0,
    target_mach_number: float = 1.0,
    toroidal_flux_expansion: float = 1.0,
    iterations: int = 1000,
):
    """Calculate the impurity concentration required to reach a given target electron temperature."""
    if CzLINT_for_fixed_impurities is None:
        CzLINT_for_fixed_impurities = CzLINT_integrator.empty()
    if mean_charge_for_fixed_impurities is None:
        mean_charge_for_fixed_impurities = Mean_charge_interpolator.empty()

    def calc_z_effective(c_z, electron_temp_eV, starting_z_effective = 1.0) -> float:
        seed_mean_z = mean_charge_for_seed_impurities.unitless_eval(electron_temp_eV)
        fixed_mean_z = mean_charge_for_fixed_impurities.unitless_eval(electron_temp_eV)
        seed_c_z = c_z * CzLINT_for_seed_impurities.weights
        fixed_c_z = CzLINT_for_fixed_impurities.weights
        z_effective = (
            starting_z_effective
            + (seed_mean_z * (seed_mean_z - 1.0) * seed_c_z).sum(dim="dim_species")
            + (fixed_mean_z * (fixed_mean_z - 1.0) * fixed_c_z).sum(dim="dim_species")
        )
        return z_effective.values

    convergence: dict[str, Any] = dict(equal_nan=False, atol=0.0, rtol=1e-6)

    mu_0 = 1.2566370621250601e-6 # meter * tesla / ampere
    elementary_charge = 1.602176634e-19 # coulomb
    MA_to_A = 1.0e6 # MA / A
    amu_to_kg = 1.6605390666e-27 # amu / kilogram
    MW_to_W = 1.0e6 # MW / W
    eV_to_J = elementary_charge
    boltzmann_constant = 1.380649e-23 # joule/kelvin
    n20_to_m3 = 1.0e20 # 10^20 / m^3 to 1 / m^3
    GW_to_W = 1e9

    # Convert all inputs to SI units, except for electron-volts
    plasma_current = plasma_current * MA_to_A
    target_angle_of_incidence = np.deg2rad(target_angle_of_incidence)
    power_crossing_separatrix = power_crossing_separatrix * MW_to_W
    separatrix_electron_density = separatrix_electron_density / n20_to_m3

    shaping_factor = np.sqrt((1.0 + elongation_psi95**2 * (1.0 + 2.0 * triangularity_psi95**2 - 1.2 * triangularity_psi95**3)) / 2.0)
    poloidal_circumference = 2.0 * np.pi * minor_radius * shaping_factor

    upstream_toroidal_field = magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))
    separatrix_average_poloidal_field = mu_0 * plasma_current / poloidal_circumference
    upstream_poloidal_field = ratio_of_upstream_to_average_poloidal_field * separatrix_average_poloidal_field

    cylindrical_safety_factor = magnetic_field_on_axis / separatrix_average_poloidal_field * minor_radius / major_radius * shaping_factor

    fieldline_pitch_at_omp = np.sqrt(upstream_toroidal_field**2 + upstream_poloidal_field**2) / upstream_poloidal_field

    momentum_loss_in_convection_layer = temperature_fit_function(
        target_electron_temp,
        amplitude=0.8858679172531956,
        width=3.8263045353064467,
        shape=0.8282347762381935,
    )

    density_loss_in_convection_layer = temperature_fit_function(
        target_electron_temp,
        amplitude=0.5587910467003282,
        width=2.020427078509838,
        shape=0.9600157520406738,
    )

    power_loss_in_convection_layer = temperature_fit_function(
        target_electron_temp,
        amplitude=0.8532115334413933,
        width=5.195481324376164,
        shape=0.9642427916765323,
    )

    electron_temp_at_cc_interface = target_electron_temp \
        / ((1.0 - momentum_loss_in_convection_layer) / (2.0 * density_loss_in_convection_layer))

    kappa_e0 = 2390.0 # W / (m * eV**3.5)
    fraction_of_power_entering_flux_tube = (1.0 - 1.0 / np.e) * fraction_of_P_SOL_to_divertor

    # Starting values for the iterative solver
    separatrix_electron_temp = 100.0 # eV
    alpha_t = 0.0
    result_valid = True

    prev_separatrix_electron_temp = np.nan
    prev_alpha_t = np.nan
    prev_c_z = np.nan

    for _it in range(iterations):
        first_loop = _it == 0

        separatrix_average_rho_s_pol = \
            np.sqrt(separatrix_electron_temp * average_ion_mass) / (separatrix_average_poloidal_field) \
                * np.sqrt(amu_to_kg / elementary_charge)# in metres, for Te in eV, mi in amu and B0 in T

        separatrix_average_lambda_Te = 2.1 * (1 + 2.1 * np.pow(alpha_t, 1.7)) * separatrix_average_rho_s_pol
        separatrix_average_lambda_q = 2.0 / 7.0 * separatrix_average_lambda_Te

        ratio_of_upstream_to_average_lambda_q = ratio_of_upstream_to_average_poloidal_field * (major_radius + minor_radius) / major_radius
        lambda_q_outboard_midplane = separatrix_average_lambda_q / ratio_of_upstream_to_average_lambda_q # in metres

        q_parallel = (
            power_crossing_separatrix
            * fraction_of_power_entering_flux_tube
            / (2.0 * np.pi * (major_radius + minor_radius) * lambda_q_outboard_midplane)
            * fieldline_pitch_at_omp
        ) # in watts per metres-squared

        if first_loop: divertor_z_effective = 1.0

        # Calculate the impact of impurities on electron heat conductivity, using
        # equation 10 from Brown and Goldston, 2021, NME 27 101002
        kappa_z = 0.672 + 0.076 * np.sqrt(divertor_z_effective) + 0.252 * divertor_z_effective
        kappa_e = kappa_e0 / kappa_z

        divertor_entrance_electron_temp = np.pow((
            electron_temp_at_cc_interface**3.5
            + 3.5 * SOL_conduction_fraction * q_parallel / divertor_broadening_factor * divertor_parallel_length / kappa_e
        ), 2. / 7.) # in electron-volts

        separatrix_electron_temp = np.pow((
            divertor_entrance_electron_temp**3.5
            + 3.5 * SOL_conduction_fraction * q_parallel * (parallel_connection_length - divertor_parallel_length) / kappa_e
        ), 2. / 7.) # in electron-volts

        separatrix_total_pressure = (
            (1.0 + separatrix_mach_number**2) * separatrix_electron_density * separatrix_electron_temp \
                * (n20_to_m3 * elementary_charge)
                * (1.0 + separatrix_ratio_of_ion_to_electron_temp / separatrix_ratio_of_electron_to_ion_density)
        ) # in Pascals

        # Run the two-point-model to calculate the required power loss fraction to achieve
        # a desired target electron temperature
        target_electron_temp_basic = (
            (8.0 * average_ion_mass / sheath_heat_transmission_factor**2)
            * ((q_parallel / GW_to_W)**2 / separatrix_total_pressure**2)
        ) * (amu_to_kg / elementary_charge * GW_to_W**2)

        f_other_target_electron_temp = (
            ((1.0 + target_ratio_of_ion_to_electron_temp / target_ratio_of_electron_to_ion_density) / 2.0)
            * ((1.0 + target_mach_number**2) ** 2 / (4.0 * target_mach_number**2))
            * toroidal_flux_expansion**-2
        )

        required_power_loss = (
            1.0
            - np.sqrt(
                target_electron_temp
                / target_electron_temp_basic
                * (1.0 - momentum_loss_in_convection_layer) ** 2
                / f_other_target_electron_temp
            )
        )

        parallel_heat_flux_at_target = q_parallel * (1.0 - required_power_loss) # W/m^2

        parallel_heat_flux_at_cc_interface = parallel_heat_flux_at_target / (1.0 - power_loss_in_convection_layer)

        # Seed impurities
        Ls_cc_div = CzLINT_for_seed_impurities.unitless_eval(electron_temp_at_cc_interface, divertor_entrance_electron_temp) * n20_to_m3**2
        Ls_div_u = CzLINT_for_seed_impurities.unitless_eval(divertor_entrance_electron_temp, separatrix_electron_temp) * n20_to_m3**2
        Ls_cc_u = CzLINT_for_seed_impurities.unitless_eval(electron_temp_at_cc_interface, separatrix_electron_temp) * n20_to_m3**2

        # Fixed impurities
        Lf_cc_div = CzLINT_for_fixed_impurities.unitless_eval(electron_temp_at_cc_interface, divertor_entrance_electron_temp) * n20_to_m3**2
        Lf_div_u = CzLINT_for_fixed_impurities.unitless_eval(divertor_entrance_electron_temp, separatrix_electron_temp) * n20_to_m3**2
        Lf_cc_u = CzLINT_for_fixed_impurities.unitless_eval(electron_temp_at_cc_interface, separatrix_electron_temp) * n20_to_m3**2

        qu = q_parallel
        qcc = parallel_heat_flux_at_cc_interface
        b = divertor_broadening_factor
        k = 2.0 * kappa_e * separatrix_electron_density**2 * separatrix_electron_temp**2

        q_div_squared = (
            (Ls_div_u * (qcc**2 + k * Lf_cc_div) + Ls_cc_div * (qu**2 - k * Lf_div_u))
            / (Ls_div_u / b**2  + Ls_cc_div)
        )
        if q_div_squared < 0:
            q_div_squared = 0.0

        c_z = (
            (qu**2 + (1 / b**2 - 1) * q_div_squared - qcc**2) / (k * Ls_cc_u)
            - Lf_cc_u / Ls_cc_u
        )

        # Use the divertor entrance temperature to calculate the divertor Zeff, which is used for
        # calculating the corrected electron heat conductivity
        divertor_z_effective = calc_z_effective(c_z, divertor_entrance_electron_temp)

        # Use the separatrix electron temperature to calculate Z-eff for alpha-t
        separatrix_z_effective = calc_z_effective(c_z, separatrix_electron_temp)

        alpha_t = calc_alpha_t(
            separatrix_electron_density=separatrix_electron_density * n20_to_m3,
            separatrix_electron_temp=separatrix_electron_temp,
            cylindrical_safety_factor=cylindrical_safety_factor,
            major_radius=major_radius,
            average_ion_mass=average_ion_mass * amu_to_kg,
            z_effective=separatrix_z_effective,
            mean_ion_charge_state=1.0,
        )

        converged = np.allclose(
            [alpha_t, c_z, separatrix_electron_temp],
            [prev_alpha_t, prev_c_z, prev_separatrix_electron_temp],
            **convergence
        )

        prev_alpha_t = alpha_t
        prev_c_z = c_z
        prev_separatrix_electron_temp = separatrix_electron_temp

    # Post-processing
    sound_speed_at_target = np.sqrt(2.0 * target_electron_temp * (eV_to_J / amu_to_kg) / average_ion_mass) # m / s

    electron_density_at_target = parallel_heat_flux_at_target / (sheath_heat_transmission_factor * target_electron_temp * eV_to_J * sound_speed_at_target) # m^-3

    # From equation 57 of Body, Kallenbach and Eich, NF 2025
    flux_density_to_pascals_factor = np.sqrt(2.0 / (np.pi * ratio_of_molecular_to_ion_mass * average_ion_mass * wall_temperature)) / np.sqrt(amu_to_kg * boltzmann_constant)# (m**-2 / s) / Pa

    parallel_to_perp_factor = np.sin(target_angle_of_incidence)

    parallel_ion_flux_to_target = electron_density_at_target * sound_speed_at_target # m**-2 / s
    neutral_pressure_in_divertor = parallel_ion_flux_to_target * parallel_to_perp_factor / flux_density_to_pascals_factor # Pa

    heat_flux_perp_to_target = parallel_heat_flux_at_target * parallel_to_perp_factor # W/m^ureg.W / ureg.m**22

    return (
        c_z,
        parallel_ion_flux_to_target,
        neutral_pressure_in_divertor,
        alpha_t,
        q_parallel,
        heat_flux_perp_to_target,
        separatrix_z_effective,
        converged,
    )
