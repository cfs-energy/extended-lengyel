"""Uses the extended Lengyel model to calculate the target electron temperature resulting from a fixed impurity seeding.

Avoids the use of non-standard libraries, except for the "test_against_reference_values" method of MavrinData.

To run in testing mode, run run_forward_extended_lengyel_model(testing=True) and leave everything else as defaults.
This will make sure that this version (which is not unit-aware) matches the unit-aware version of
the extended Lengyel model.
"""

import numpy as np

try:
    from .run_unitless_extended_lengyel_model import MavrinData, temperature_fit_function, calc_alpha_t, run_inverse_extended_lengyel_model
except ImportError:
    from run_unitless_extended_lengyel_model import MavrinData, temperature_fit_function, calc_alpha_t, run_inverse_extended_lengyel_model

def run_forward_extended_lengyel_model(
    power_crossing_separatrix: float = 5.5,
    separatrix_electron_density: float = 3.3e19,
    divertor_broadening_factor: float = 3.0,
    impurity_concentrations: dict[str, float] = {"Nitrogen": 0.038183522427857504, "Argon": 0.0019091761213928752, "Helium": 1.0e-2},  # noqa: B006
    magnetic_field_on_axis: float = 2.5,
    plasma_current: float = 1.0,
    parallel_connection_length: float = 20.0,
    divertor_parallel_length: float = 5.0,
    major_radius: float = 1.65,
    minor_radius: float = 0.5,
    elongation_psi95: float = 1.6,
    triangularity_psi95: float = 0.3,
    average_ion_mass: float = 2.0,
    ratio_of_upstream_to_average_poloidal_field: float = 4./3.,
    ne_tau: float = 0.5e+17,
    sheath_heat_transmission_factor: float = 8.,
    target_angle_of_incidence: float = 3.,
    fraction_of_P_SOL_to_divertor = 2./3.,
    SOL_conduction_fraction: float = 1.0,
    ratio_of_molecular_to_ion_mass = 2.0,
    wall_temperature = 300.0,
    separatrix_mach_number: float = 0.0,
    separatrix_ratio_of_ion_to_electron_temp: float = 1.0,
    separatrix_ratio_of_electron_to_ion_density: float = 1.0,
    target_ratio_of_ion_to_electron_temp: float = 1.0,
    target_ratio_of_electron_to_ion_density: float = 1.0,
    target_mach_number: float = 1.0,
    toroidal_flux_expansion: float = 1.0,
    inner_loop_1_iterations: int = 5,
    inner_loop_2_iterations: int = 5,
    outer_loop_iterations: int = 5,
    testing: bool = False
):
    """Calculate the impurity concentration required to reach a given target electron temperature.

    Inputs:
        divertor_parallel_length: length along a magnetic fieldline from the divertor target to X-point, in metres
        parallel_connection_length: length along a magnetic fieldline from the divertor target to the outboard midplane, in metres
        major_radius: major radius of the magnetic axis, in metres
        minor_radius: minor radius from the magnetic axis to the outboard midplane, in metres
        elongation_psi95: elongation at the psiN=0.95 surface
        triangularity_psi95: triangularity at the psiN=0.95 surface
        magnetic_field_on_axis: magnetic field strength at the magnetic axis, in tesla
        plasma_current: plasma current, in mega-amperes
        ratio_of_upstream_to_average_poloidal_field: Bpol at the outboard midplane divided by Bpol averaged over the separatrix
        average_ion_mass: average main-ion mass, in atomic mass units
        ne_tau: product of electron density and ion residence time, in seconds per cubic metre
        sheath_heat_transmission_factor: gamma factor used to calculate heat flux through the sheath from convective heat flux to sheath-entrance
        target_angle_of_incidence: angle of incidence between magnetic fieldline and divertor target, in degrees
        divertor_broadening_factor: divertor heat flux width (lambda_INT) divided by upstream heat flux width (lambda_q)
        power_crossing_separatrix: total power crossing the separatrix, in megawatts
        fraction_of_P_SOL_to_divertor: fraction of power directed to the outer divertor
        separatrix_electron_density: electron density at the outboard midplane, in per cubic metre
        target_electron_temp: desired electron temperature at the sheath entrance, in electron-volts
        SOL_conduction_fraction: fraction of power carried by electron heat conduction
        ratio_of_molecular_to_ion_mass: ratio of molecular mass to ion mass (typically 2 for hydrogenic species)
        wall_temperature: temperature of divertor walls, in kelvin

    If testing = True, the function checks against values calculated with the default input values.
    """
    mu_0 = 1.2566370621250601e-6 # meter * tesla / ampere
    elementary_charge = 1.602176634e-19 # coulomb
    MA_to_A = 1.0e6 # MA / A
    amu_to_kg = 1.6605390666e-27 # amu / kilogram
    MW_to_W = 1.0e6 # MW / W
    eV_to_J = elementary_charge
    boltzmann_constant = 1.380649e-23 # joule/kelvin

    impurities: dict[str, tuple[MavrinData, float]] = {}
    for species, concentration in impurity_concentrations.items():
        impurities[species] = (MavrinData(species), concentration)

    if testing: MavrinData("nitrogen").test_against_reference_values()

    def calc_cz_LINT(start_temp_eV: float, stop_temp_eV: float) -> float:
        """Calculate the sum of cz * LINT for all impurities, between the start and stop temperatures."""
        weighted_LINT = [
            concentration * mavrin_data.get_Lint(start_temp_eV, stop_temp_eV, ne_tau)
            for (mavrin_data, concentration) in impurities.values()
        ]
        return np.sum(weighted_LINT)

    def calc_z_effective(electron_temp_eV: float) -> float:
        """Calculate Z_effective at the given electron temperature."""
        z_effective = 1.0

        for (mavrin_data, concentration) in impurities.values():
            mean_z = mavrin_data.get_mean_charge(electron_temp_eV, ne_tau)
            z_effective = z_effective + (mean_z * (mean_z - 1.0) * concentration)

        return z_effective

    # Convert all inputs to SI units, except for electron-volts
    plasma_current = plasma_current * MA_to_A
    average_ion_mass = average_ion_mass * amu_to_kg
    target_angle_of_incidence = np.deg2rad(target_angle_of_incidence)
    power_crossing_separatrix = power_crossing_separatrix * MW_to_W

    shaping_factor = np.sqrt((1.0 + elongation_psi95**2 * (1.0 + 2.0 * triangularity_psi95**2 - 1.2 * triangularity_psi95**3)) / 2.0)
    poloidal_circumference = 2.0 * np.pi * minor_radius * shaping_factor

    upstream_toroidal_field = magnetic_field_on_axis * (major_radius / (major_radius + minor_radius))
    if testing: assert np.isclose(upstream_toroidal_field, 1.9186046511627908)
    separatrix_average_poloidal_field = mu_0 * plasma_current / poloidal_circumference
    upstream_poloidal_field = ratio_of_upstream_to_average_poloidal_field * separatrix_average_poloidal_field
    if testing: assert np.isclose(upstream_poloidal_field, 0.38008769560627165)

    cylindrical_safety_factor = magnetic_field_on_axis / separatrix_average_poloidal_field * minor_radius / major_radius * shaping_factor
    if testing: assert np.isclose(cylindrical_safety_factor, 3.729030300985294)

    fieldline_pitch_at_omp = np.sqrt(upstream_toroidal_field**2 + upstream_poloidal_field**2) / upstream_poloidal_field
    if testing: assert np.isclose(fieldline_pitch_at_omp, 5.145894598644929)

    kappa_e0 = 2390.0 # W / (m * eV**3.5)
    fraction_of_power_entering_flux_tube = (1.0 - 1.0 / np.e) * fraction_of_P_SOL_to_divertor

    # Starting values for the iterative solver
    target_electron_temp = 2.0 # eV
    electron_temp_at_cc_interface = 2.5 # eV
    divertor_entrance_electron_temp = 50.0 # eV
    separatrix_electron_temp = 100.0 # eV
    alpha_t = 0.0

    for _outer_it in range(outer_loop_iterations):

        for _inner_loop_it in range(inner_loop_1_iterations):
            # Calculate q_parallel consistent with alpha-t and the separatrix electron temperature
            first_loop = (_outer_it == 0) and (_inner_loop_it == 0)

            separatrix_average_rho_s_pol = \
                np.sqrt(separatrix_electron_temp * elementary_charge * average_ion_mass) \
                    / (elementary_charge * separatrix_average_poloidal_field) # in metres, for Te in eV, mi in amu and B0 in T
            if testing and first_loop: assert np.isclose(separatrix_average_rho_s_pol, 0.005050556986156449), separatrix_average_rho_s_pol

            separatrix_average_lambda_Te = 2.1 * (1 + 2.1 * alpha_t**1.7) * separatrix_average_rho_s_pol
            separatrix_average_lambda_q = 2.0 / 7.0 * separatrix_average_lambda_Te

            ratio_of_upstream_to_average_lambda_q = ratio_of_upstream_to_average_poloidal_field * (major_radius + minor_radius) / major_radius
            lambda_q_outboard_midplane = separatrix_average_lambda_q / ratio_of_upstream_to_average_lambda_q # in metres
            if testing and first_loop: assert np.isclose(lambda_q_outboard_midplane, 1.7442039824284483e-3), lambda_q_outboard_midplane

            q_parallel = (
                power_crossing_separatrix
                * fraction_of_power_entering_flux_tube
                / (2.0 * np.pi * (major_radius + minor_radius) * lambda_q_outboard_midplane)
                * fieldline_pitch_at_omp
            ) # in watts per metres-squared
            if testing and first_loop: assert np.isclose(q_parallel, 0.5061935771095335 * 1e9), q_parallel

            # Calculate the impact of impurities on electron heat conductivity, using
            # equation 10 from Brown and Goldston, 2021, NME 27 101002
            divertor_z_effective = calc_z_effective(divertor_entrance_electron_temp)
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

            separatrix_z_effective = calc_z_effective(separatrix_electron_temp)
            alpha_t = calc_alpha_t(
                separatrix_electron_density=separatrix_electron_density,
                separatrix_electron_temp=separatrix_electron_temp,
                cylindrical_safety_factor=cylindrical_safety_factor,
                major_radius=major_radius,
                average_ion_mass=average_ion_mass,
                z_effective=separatrix_z_effective,
                mean_ion_charge_state=1.0,
            )

        # Calculate the power loss due to impurities

        # Seed impurities
        Lint_cc_div = calc_cz_LINT(electron_temp_at_cc_interface, divertor_entrance_electron_temp)
        Lint_div_u = calc_cz_LINT(divertor_entrance_electron_temp, separatrix_electron_temp)

        qu = q_parallel
        b = divertor_broadening_factor
        k = 2.0 * kappa_e * separatrix_electron_density**2 * separatrix_electron_temp**2

        qcc = np.sqrt(qu**2 / b**2 - k * (Lint_div_u / b**2 + Lint_cc_div))
        parallel_heat_flux_at_cc_interface = qcc

        separatrix_total_pressure = (
            (1.0 + separatrix_mach_number**2) * separatrix_electron_density * separatrix_electron_temp * elementary_charge
                * (1.0 + separatrix_ratio_of_ion_to_electron_temp / separatrix_ratio_of_electron_to_ion_density)
        ) # in Pascals

        target_electron_temp_basic = (
            (8.0 * average_ion_mass / sheath_heat_transmission_factor**2)
            * (q_parallel**2 / separatrix_total_pressure**2)
        ) / elementary_charge

        f_other = (
            ((1.0 + target_ratio_of_ion_to_electron_temp / target_ratio_of_electron_to_ion_density) / 2.0)
            * ((1.0 + target_mach_number**2) ** 2 / (4.0 * target_mach_number**2))
            * toroidal_flux_expansion**-2
        )

        # Calculate Te_tar consistent with parallel_heat_flux_at_cc_interface

        for _inner_loop_it in range(inner_loop_2_iterations):
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

    # Post-processing
    sound_speed_at_target = np.sqrt(2.0 * target_electron_temp * eV_to_J / average_ion_mass) # m / s
    # if testing: assert np.isclose(sound_speed_at_target, 15025.833662282057), sound_speed_at_target

    electron_density_at_target = parallel_heat_flux_at_target / (sheath_heat_transmission_factor * target_electron_temp * eV_to_J * sound_speed_at_target) # m^-3
    # if testing: assert np.isclose(electron_density_at_target, 3.359214345710722e+20, rtol=1e-2), electron_density_at_target

    # From equation 57 of Body, Kallenbach and Eich, NF 2025
    flux_density_to_pascals_factor = np.sqrt(2.0 / (np.pi * ratio_of_molecular_to_ion_mass * average_ion_mass * boltzmann_constant * wall_temperature)) # (m**-2 / s) / Pa
    if testing: assert np.isclose(flux_density_to_pascals_factor, 1.521189252551778e+23, rtol=1e-2)

    parallel_to_perp_factor = np.sin(target_angle_of_incidence)
    # if testing: assert np.isclose(parallel_to_perp_factor, 0.052335956242943835, rtol=1e-2)

    parallel_ion_flux_to_target = electron_density_at_target * sound_speed_at_target # m**-2 / s
    perp_ion_flux_to_target = parallel_ion_flux_to_target * parallel_to_perp_factor # m**-2 / s

    # if testing:
    #     assert np.isclose(parallel_ion_flux_to_target, 5.047499599460096e+24, rtol=1e-2)
    #     assert np.isclose(perp_ion_flux_to_target, 2.641657181736201e+23, rtol=1e-2)

    neutral_pressure_in_divertor = parallel_ion_flux_to_target * parallel_to_perp_factor / flux_density_to_pascals_factor # Pa
    # if testing: assert np.isclose(neutral_pressure_in_divertor, 1.736573655976632, rtol=1e-2)

    heat_flux_perp_to_target = parallel_heat_flux_at_target * parallel_to_perp_factor # W/m^2
    # if testing: assert np.isclose(heat_flux_perp_to_target, 792305.5442545213, rtol=1e-2)

    return_values = dict(
        neutral_pressure_in_divertor = neutral_pressure_in_divertor,
        alpha_t = alpha_t,
        q_parallel = q_parallel,
        heat_flux_perp_to_target = heat_flux_perp_to_target,
        separatrix_z_effective = separatrix_z_effective,
        target_electron_temp = target_electron_temp
    )

    return return_values

if __name__=="__main__":

    result = run_forward_extended_lengyel_model(
        testing = True
    )

    print(result)
