"""Uses the extended Lengyel model to calculate the impurity concentration required to reach a fixed target electron temperature."""

import numpy as np
from typing import Literal, Any
from cfspopcon.unit_handling import wraps_ufunc, ureg, Unitfull

np.seterr(over="raise",under="raise")

class MavrinData:

    def __init__(self,
                 species: Literal["helium", "lithium", "beryllium", "carbon", "nitrogen", "oxygen", "neon", "argon"],
    ) -> None:
        from pathlib import Path
        import yaml

        filepath = Path(__file__).parent / "mavrin_data.yaml"
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath.absolute()} doesn't exist.")

        with open(filepath) as file:
            mavrin_data = yaml.load(file, Loader=yaml.FullLoader)

        self.species = species.lower() # type:ignore
        species_available = {key.removesuffix("_Lz").removesuffix("_mean_charge").lower() for key in mavrin_data.keys()}

        if self.species not in species_available:
            raise NotImplementedError(f"Species {self.species} not in {', '.join(species_available)}")

        self.Lz_coeffs = mavrin_data[f"{self.species}_Lz"]
        self.mean_charge_coeffs = mavrin_data[f"{self.species}_mean_charge"]

    @staticmethod
    def compute_polynomial_fit(
        Te_eV: float, ne_tau_s_per_m3: float, coeff: dict[str, list[float | int]], info: bool = False
    ) -> float:
        """Inner loop for computing the Lz or mean_charge polynomial fit from Mavrin, J. Fus. Eng., 2017."""
        Tmin_eV = coeff["Tmin_eV"]
        Tmax_eV = coeff["Tmax_eV"]

        if not Tmin_eV[0] <= Te_eV <= Tmax_eV[-1]:
            if info:
                print(f"{Te_eV}eV outside fitted range {Tmin_eV[0]}eV to {Tmax_eV[-1]}eV")
            return np.nan
        if ne_tau_s_per_m3 < 1e15:
            if info:
                print(f"{ne_tau_s_per_m3} outside fitted range above 1e16 m^-3 s")
            return np.nan

        X = np.log10(Te_eV)
        Y = np.log10(ne_tau_s_per_m3 / 1e19)
        if Y > 0.0:
            print("Warning: treating points with ne_tau_s_per_m3 > 1e19 m^-3 s as coronal.")
        Y = np.minimum(Y, 0.0)

        N_bins = len(Tmin_eV)
        assert len(Tmax_eV) == N_bins

        for i in range(N_bins):
            if Tmin_eV[i] <= Te_eV <= Tmax_eV[i]:
                T_bin = i

        A = np.zeros(10)
        for i in range(10):
            A[i] = coeff[f"A{i}"][T_bin]

        F = (
            A[0]
            + A[1] * X
            + A[2] * Y
            + A[3] * X**2
            + A[4] * X * Y
            + A[5] * Y**2
            + A[6] * X**3
            + A[7] * X**2 * Y
            + A[8] * X * Y**2
            + A[9] * Y**3
        )

        return np.power(10, F)

    def get_Lz(self, electron_temp_eV: float, ne_tau_s_per_m3: float) -> float:
        """Calculate the species Lz factor in watt metre-cubed, using the Mavrin polynomials."""
        return self.compute_polynomial_fit(electron_temp_eV, ne_tau_s_per_m3, self.Lz_coeffs)

    def get_mean_charge(self, electron_temp_eV: float, ne_tau_s_per_m3: float) -> float:
        """Calculate the species mean charge, using the Mavrin polynomials."""
        return self.compute_polynomial_fit(electron_temp_eV, ne_tau_s_per_m3, self.mean_charge_coeffs)

    def get_Lint(self, start_temp_eV: float, stop_temp_eV: float, ne_tau_s_per_m3: float, resolution: int=100) -> float:
        """Calculate the integral from the start temp to the stop temp of Lz * sqrt(Te).

        Returns in electron-volt^1.5 * metre^3 * watt.
        """
        Tmin = np.min(self.Lz_coeffs["Tmin_eV"])
        Tmax = np.max(self.Lz_coeffs["Tmax_eV"])

        if np.isnan(start_temp_eV) or np.isnan(stop_temp_eV):
            return np.nan
        assert start_temp_eV < stop_temp_eV, f"Stop temp must be larger than start temp. {start_temp_eV}<{stop_temp_eV}"
        assert start_temp_eV > Tmin, f"Temperature out of range (too low). {start_temp_eV} > {Tmin}"
        assert stop_temp_eV < Tmax, f"Temperature out of range (too high). {stop_temp_eV} < {Tmax}"

        Lz_values = np.zeros(resolution)
        electron_temp = np.logspace(np.log10(start_temp_eV), np.log10(stop_temp_eV), num = resolution)

        for i in range(resolution):
            Lz_values[i] = self.compute_polynomial_fit(electron_temp[i], ne_tau_s_per_m3, self.Lz_coeffs)

        Lz_sqrt_Te = Lz_values * np.sqrt(electron_temp)

        return np.trapezoid(x = electron_temp, y = Lz_sqrt_Te)

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
        seed_impurity_weights = None,
        fixed_impurity_concentrations = None,
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
        ne_tau = ureg.m**-3 * ureg.s,
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
    seed_impurity_weights: dict[str, float],
    fixed_impurity_concentrations: dict[str, float],
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
    ne_tau: Unitfull = 0.5e17 * ureg.m**-3 * ureg.s,
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
    """
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

    seed_impurities: dict[str, tuple[MavrinData, float]] = {}
    for species, weight in seed_impurity_weights.items():
        seed_impurities[species] = (MavrinData(species), weight)

    fixed_impurities: dict[str, tuple[MavrinData, float]] = {}
    for species, concentration in fixed_impurity_concentrations.items():
        fixed_impurities[species] = (MavrinData(species), concentration)

    def calc_weighted_seed_LINT(start_temp_eV: float, stop_temp_eV: float) -> float:
        """Calculate the sum of wz * LINT for seed impurities, between the start and stop temperatures."""
        weighted_LINT = [
            weight * mavrin_data.get_Lint(start_temp_eV, stop_temp_eV, ne_tau) * n20_to_m3**2
            for (mavrin_data, weight) in seed_impurities.values()
        ]
        return np.sum(weighted_LINT)

    def calc_fixed_cz_LINT(start_temp_eV: float, stop_temp_eV: float) -> float:
        """Calculate the sum of cz * LINT for fixed background impurities, between the start and stop temperatures."""
        weighted_LINT = [
            concentration * mavrin_data.get_Lint(start_temp_eV, stop_temp_eV, ne_tau) * n20_to_m3**2
            for (mavrin_data, concentration) in fixed_impurities.values()
        ]
        return np.sum(weighted_LINT)

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
        Ls_cc_div = calc_weighted_seed_LINT(electron_temp_at_cc_interface, divertor_entrance_electron_temp)
        Ls_div_u = calc_weighted_seed_LINT(divertor_entrance_electron_temp, separatrix_electron_temp)
        Ls_cc_u = calc_weighted_seed_LINT(electron_temp_at_cc_interface, separatrix_electron_temp)

        # Fixed impurities
        Lf_cc_div = calc_fixed_cz_LINT(electron_temp_at_cc_interface, divertor_entrance_electron_temp)
        Lf_div_u = calc_fixed_cz_LINT(divertor_entrance_electron_temp, separatrix_electron_temp)
        Lf_cc_u = calc_fixed_cz_LINT(electron_temp_at_cc_interface, separatrix_electron_temp)

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
        divertor_z_effective = 1.0
        for (mavrin_data, weight) in seed_impurities.values():
            mean_z = mavrin_data.get_mean_charge(divertor_entrance_electron_temp, ne_tau)
            divertor_z_effective = divertor_z_effective + (mean_z * (mean_z - 1.0) * c_z * weight)
        for (mavrin_data, concentration) in fixed_impurities.values():
            mean_z = mavrin_data.get_mean_charge(divertor_entrance_electron_temp, ne_tau)
            divertor_z_effective = divertor_z_effective + (mean_z * (mean_z - 1.0) * concentration)

        # Use the separatrix electron temperature to calculate Z-eff for alpha-t
        separatrix_z_effective = 1.0
        for (mavrin_data, weight) in seed_impurities.values():
            mean_z = mavrin_data.get_mean_charge(separatrix_electron_temp, ne_tau)
            separatrix_z_effective = separatrix_z_effective + (mean_z * (mean_z - 1.0) * c_z * weight)
        for (mavrin_data, concentration) in fixed_impurities.values():
            mean_z = mavrin_data.get_mean_charge(separatrix_electron_temp, ne_tau)
            separatrix_z_effective = separatrix_z_effective + (mean_z * (mean_z - 1.0) * concentration)

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


if __name__=="__main__":
    import xarray as xr

    (
        c_z,
        parallel_ion_flux_to_target,
        neutral_pressure_in_divertor,
        alpha_t,
        q_parallel,
        heat_flux_perp_to_target,
        separatrix_z_effective,
        converged,
    ) = run_inverse_extended_lengyel_model(
        target_electron_temp = xr.DataArray([2.34, 3.0], dims="dim_0") * ureg.eV,
        power_crossing_separatrix = 5.5 * ureg.MW,
        separatrix_electron_density = 3.3 * ureg.n19,
        divertor_broadening_factor = 3.0,
        seed_impurity_weights = {"Nitrogen": 1.0, "Argon": 0.05},
        fixed_impurity_concentrations = {"Helium": 1.0e-2},
        magnetic_field_on_axis = 2.5 * ureg.T,
        plasma_current = 1.0 * ureg.MA,
        parallel_connection_length = 20.0 * ureg.m,
        divertor_parallel_length = 5.0 * ureg.m,
        major_radius = 1.65 * ureg.m,
        minor_radius = 0.5 * ureg.m,
        elongation_psi95 = 1.6,
        triangularity_psi95 = 0.3,
        target_angle_of_incidence = 3.0,
        fraction_of_P_SOL_to_divertor = 2./3.,
        sheath_heat_transmission_factor = 8.,
        iterations = 100,
    )
