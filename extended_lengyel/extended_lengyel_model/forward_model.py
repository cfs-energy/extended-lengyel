"""Uses the extended Lengyel model to calculate the target electron temperature resulting from a fixed impurity seeding."""

import numpy as np
from typing import Any, Optional
from cfspopcon.unit_handling import wraps_ufunc, ureg, Unitfull

try:
    from .inverse_model import temperature_fit_function, calc_alpha_t
    from .Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
except ImportError:
    from inverse_model import temperature_fit_function, calc_alpha_t
    from Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator

@wraps_ufunc(
    input_units=dict(
        c_z = ureg.dimensionless,
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
        target_electron_temp = ureg.eV,
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
def run_forward_extended_lengyel_model(
    c_z: Unitfull,
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

    def relax(new_value, prev_value, relaxation_factor=0.4):
        return relaxation_factor * new_value + (1 - relaxation_factor) * prev_value

    mu_0 = 1.2566370621250601e-6 # meter * tesla / ampere
    elementary_charge = 1.602176634e-19 # coulomb
    MA_to_A = 1.0e6 # MA / A
    amu_to_kg = 1.6605390666e-27 # amu / kilogram
    MW_to_W = 1.0e6 # MW / W
    eV_to_J = elementary_charge
    boltzmann_constant = 1.380649e-23 # joule/kelvin
    n20_to_m3 = 1.0e20 # 10^20 / m^3 to 1 / m^3

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

    kappa_e0 = 2390.0 # W / (m * eV**3.5)
    fraction_of_power_entering_flux_tube = (1.0 - 1.0 / np.e) * fraction_of_P_SOL_to_divertor

    # Starting values for the iterative solver
    target_electron_temp = 2.0 # eV
    electron_temp_at_cc_interface = 2.5 # eV
    divertor_entrance_electron_temp = 50.0 # eV
    separatrix_electron_temp = 100.0 # eV
    alpha_t = 0.0

    prev_target_electron_temp = np.nan
    prev_parallel_heat_flux_at_cc_interface = np.nan
    prev_divertor_entrance_electron_temp = np.nan
    prev_separatrix_electron_temp = np.nan
    prev_alpha_t = np.nan

    target_electron_temp_its = np.zeros(iterations)
    parallel_heat_flux_at_cc_interface_its = np.zeros(iterations)
    separatrix_electron_temp_its = np.zeros(iterations)
    alpha_t_its = np.zeros(iterations)

    for _it in range(iterations):

        # Calculate q_parallel consistent with alpha-t and the separatrix electron temperature
        first_loop = (_it == 0)

        separatrix_average_rho_s_pol = \
            np.sqrt(separatrix_electron_temp * average_ion_mass) \
                / (separatrix_average_poloidal_field) \
                * np.sqrt(amu_to_kg / elementary_charge)# in metres, for Te in eV, mi in amu and B0 in T

        separatrix_average_lambda_Te = 2.1 * (1 + 2.1 * alpha_t**1.7) * separatrix_average_rho_s_pol
        separatrix_average_lambda_q = 2.0 / 7.0 * separatrix_average_lambda_Te

        ratio_of_upstream_to_average_lambda_q = ratio_of_upstream_to_average_poloidal_field * (major_radius + minor_radius) / major_radius
        lambda_q_outboard_midplane = separatrix_average_lambda_q / ratio_of_upstream_to_average_lambda_q # in metres

        q_parallel = (
            power_crossing_separatrix
            * fraction_of_power_entering_flux_tube
            / (2.0 * np.pi * (major_radius + minor_radius) * lambda_q_outboard_midplane)
            * fieldline_pitch_at_omp
        ) # in watts per metres-squared

        # Calculate the impact of impurities on electron heat conductivity, using
        # equation 10 from Brown and Goldston, 2021, NME 27 101002
        divertor_z_effective = calc_z_effective(c_z, divertor_entrance_electron_temp)
        kappa_z = 0.672 + 0.076 * np.sqrt(divertor_z_effective) + 0.252 * divertor_z_effective
        kappa_e = kappa_e0 / kappa_z

        divertor_entrance_electron_temp = (
            electron_temp_at_cc_interface**3.5
            + 3.5 * SOL_conduction_fraction * q_parallel / divertor_broadening_factor * divertor_parallel_length / kappa_e
        ) ** (2. / 7.) # in electron-volts

        separatrix_electron_temp = (
            divertor_entrance_electron_temp**3.5
            + 3.5 * SOL_conduction_fraction * q_parallel * (parallel_connection_length - divertor_parallel_length) / kappa_e
        ) ** (2. / 7.) # in electron-volts

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

        # Calculate the power loss due to impurities
        # Seed impurities
        Ls_cc_div = CzLINT_for_seed_impurities.unitless_eval(electron_temp_at_cc_interface, divertor_entrance_electron_temp) * n20_to_m3**2
        Ls_div_u = CzLINT_for_seed_impurities.unitless_eval(divertor_entrance_electron_temp, separatrix_electron_temp) * n20_to_m3**2

        # Fixed impurities
        Lf_cc_div = CzLINT_for_fixed_impurities.unitless_eval(electron_temp_at_cc_interface, divertor_entrance_electron_temp) * n20_to_m3**2
        Lf_div_u = CzLINT_for_fixed_impurities.unitless_eval(divertor_entrance_electron_temp, separatrix_electron_temp) * n20_to_m3**2

        Lint_cc_div = c_z * Ls_cc_div + Lf_cc_div
        Lint_div_u = c_z * Ls_div_u + Lf_div_u

        qu = q_parallel
        b = divertor_broadening_factor
        k = 2.0 * kappa_e * separatrix_electron_density**2 * separatrix_electron_temp**2

        qcc_squared = qu**2 / b**2 - k * (Lint_div_u / b**2 + Lint_cc_div)

        if qcc_squared < 0:
            qcc_squared = 0.0

        parallel_heat_flux_at_cc_interface = np.sqrt(qcc_squared)

        separatrix_total_pressure = (
            (1.0 + separatrix_mach_number**2) * separatrix_electron_density * separatrix_electron_temp
                * (1.0 + separatrix_ratio_of_ion_to_electron_temp / separatrix_ratio_of_electron_to_ion_density)
        ) * (n20_to_m3 * elementary_charge) # in Pascals

        target_electron_temp_basic = (
            (8.0 * average_ion_mass / sheath_heat_transmission_factor**2)
            * (q_parallel**2 / separatrix_total_pressure**2)
        ) * amu_to_kg / elementary_charge

        f_other = (
            ((1.0 + target_ratio_of_ion_to_electron_temp / target_ratio_of_electron_to_ion_density) / 2.0)
            * ((1.0 + target_mach_number**2) ** 2 / (4.0 * target_mach_number**2))
            * toroidal_flux_expansion**-2
        )

        # Calculate Te_tar consistent with parallel_heat_flux_at_cc_interface
        momentum_loss_in_convection_layer = temperature_fit_function(
            target_electron_temp,
            amplitude=0.8858679172531956,
            width=3.8263045353064467,
            shape=0.8282347762381935,
        )

        power_loss_in_convection_layer = temperature_fit_function(
            target_electron_temp,
            amplitude=0.8532115334413933,
            width=5.195481324376164,
            shape=0.9642427916765323,
        )

        density_loss_in_convection_layer = temperature_fit_function(
            target_electron_temp,
            amplitude=0.5587910467003282,
            width=2.020427078509838,
            shape=0.9600157520406738,
        )

        parallel_heat_flux_at_target = (1.0 - power_loss_in_convection_layer) * parallel_heat_flux_at_cc_interface
        SOL_power_loss_fraction = 1.0 - parallel_heat_flux_at_target / q_parallel
        f_vol_loss = (1.0 - SOL_power_loss_fraction) ** 2 / (1.0 - momentum_loss_in_convection_layer) ** 2

        target_electron_temp = target_electron_temp_basic * f_vol_loss * f_other

        electron_temp_at_cc_interface = target_electron_temp \
            / ((1.0 - momentum_loss_in_convection_layer) / (2.0 * density_loss_in_convection_layer))

        converged = np.allclose(
            [
                alpha_t,
                target_electron_temp,
                divertor_entrance_electron_temp,
                separatrix_electron_temp,
                parallel_heat_flux_at_cc_interface
            ],
            [
                prev_alpha_t,
                prev_target_electron_temp,
                prev_divertor_entrance_electron_temp,
                prev_separatrix_electron_temp,
                prev_parallel_heat_flux_at_cc_interface
            ],
            **convergence
        )

        alpha_t_its[_it] = alpha_t
        target_electron_temp_its[_it] = target_electron_temp
        separatrix_electron_temp_its[_it] = separatrix_electron_temp
        parallel_heat_flux_at_cc_interface_its[_it] = parallel_heat_flux_at_cc_interface
        if converged:
            alpha_t_its = alpha_t_its[:_it+1]
            target_electron_temp_its = target_electron_temp_its[:_it+1]
            separatrix_electron_temp_its = separatrix_electron_temp_its[:_it+1]
            parallel_heat_flux_at_cc_interface_its = parallel_heat_flux_at_cc_interface_its[:_it+1]
            break

        if _it > 0:
            parallel_heat_flux_at_cc_interface = relax(parallel_heat_flux_at_cc_interface, prev_parallel_heat_flux_at_cc_interface)
            target_electron_temp = relax(target_electron_temp, prev_target_electron_temp)
            divertor_entrance_electron_temp = relax(divertor_entrance_electron_temp, prev_divertor_entrance_electron_temp)
            separatrix_electron_temp = relax(separatrix_electron_temp, prev_separatrix_electron_temp)
            alpha_t = relax(alpha_t, prev_alpha_t)

        prev_parallel_heat_flux_at_cc_interface = parallel_heat_flux_at_cc_interface
        prev_target_electron_temp = target_electron_temp
        prev_divertor_entrance_electron_temp = divertor_entrance_electron_temp
        prev_separatrix_electron_temp = separatrix_electron_temp
        prev_alpha_t = alpha_t

    # Post-processing
    sound_speed_at_target = np.sqrt(2.0 * target_electron_temp * (eV_to_J / amu_to_kg) / average_ion_mass) # m / s

    electron_density_at_target = parallel_heat_flux_at_target / (sheath_heat_transmission_factor * target_electron_temp * eV_to_J * sound_speed_at_target) # m^-3

    # From equation 57 of Body, Kallenbach and Eich, NF 2025
    flux_density_to_pascals_factor = np.sqrt(2.0 / (np.pi * ratio_of_molecular_to_ion_mass * average_ion_mass * wall_temperature)) / np.sqrt(amu_to_kg * boltzmann_constant)# (m**-2 / s) / Pa

    parallel_to_perp_factor = np.sin(target_angle_of_incidence)

    parallel_ion_flux_to_target = electron_density_at_target * sound_speed_at_target # m**-2 / s
    neutral_pressure_in_divertor = parallel_ion_flux_to_target * parallel_to_perp_factor / flux_density_to_pascals_factor # Pa

    heat_flux_perp_to_target = parallel_heat_flux_at_target * parallel_to_perp_factor # W/m^2

    return (
        target_electron_temp,
        parallel_ion_flux_to_target,
        neutral_pressure_in_divertor,
        alpha_t,
        q_parallel,
        heat_flux_perp_to_target,
        separatrix_z_effective,
        converged,
    )
