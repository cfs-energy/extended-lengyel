"""Public entry points for the four extended Lengyel model variants.

Each wrapper declares the model's units and signature, and delegates to the
unified solver in :mod:`.solver`. The models are registered in the cfspopcon
Algorithm registry under their function names.
"""

from cfspopcon.algorithm_class import Algorithm
from cfspopcon.unit_handling import Unitfull, ureg, wraps_ufunc

from ..extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator

from .constants import n20_to_m3
from .solver import ModelInputs, run_extended_lengyel_solver
from .units import (
    FORWARD_PDIV_RETURN_KEYS,
    FORWARD_RETURN_KEYS,
    INVERSE_PDIV_RETURN_KEYS,
    INVERSE_RETURN_KEYS,
    build_input_units,
    return_units_for,
)


@Algorithm.register_algorithm(return_keys=list(FORWARD_RETURN_KEYS))
@wraps_ufunc(
    input_units=build_input_units("impurity_fraction", "separatrix_electron_density"),
    return_units=return_units_for(FORWARD_RETURN_KEYS),
    output_core_dims=tuple(() for _ in FORWARD_RETURN_KEYS),
)
def run_forward_extended_lengyel_model(
    impurity_fraction: Unitfull,
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
    CzLINT_for_fixed_impurities: CzLINT_integrator | None = None,
    mean_charge_for_fixed_impurities: Mean_charge_interpolator | None = None,
    average_ion_mass: Unitfull = 2.0 * ureg.amu,
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    ratio_of_upstream_to_average_poloidal_field: Unitfull = 4.0 / 3.0 * ureg.dimensionless,
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
    """Calculate the target electron temperature resulting from a fixed impurity seeding."""
    p = ModelInputs.from_wrapper_args(**locals())
    r = run_extended_lengyel_solver(p, mode="forward", pdiv=False)
    return (
        r.target_electron_temp,
        r.parallel_ion_flux_to_target,
        r.neutral_pressure_in_divertor,
        r.alpha_t,
        r.q_parallel,
        r.heat_flux_perp_to_target,
        r.separatrix_z_effective,
        r.converged,
    )


@Algorithm.register_algorithm(return_keys=list(FORWARD_PDIV_RETURN_KEYS))
@wraps_ufunc(
    input_units=build_input_units("impurity_fraction", "neutral_pressure_in_divertor"),
    return_units=return_units_for(FORWARD_PDIV_RETURN_KEYS),
    output_core_dims=tuple(() for _ in FORWARD_PDIV_RETURN_KEYS),
)
def run_forward_extended_lengyel_model_pdiv(
    impurity_fraction: Unitfull,
    power_crossing_separatrix: Unitfull,
    neutral_pressure_in_divertor: Unitfull,
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
    CzLINT_for_fixed_impurities: CzLINT_integrator | None = None,
    mean_charge_for_fixed_impurities: Mean_charge_interpolator | None = None,
    average_ion_mass: Unitfull = 2.0 * ureg.amu,
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    ratio_of_upstream_to_average_poloidal_field: Unitfull = 4.0 / 3.0 * ureg.dimensionless,
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
    """Calculate the target electron temperature from a fixed impurity seeding and divertor neutral pressure."""
    p = ModelInputs.from_wrapper_args(**locals())
    r = run_extended_lengyel_solver(p, mode="forward", pdiv=True)
    return (
        r.target_electron_temp,
        r.separatrix_electron_density * n20_to_m3,  # n20 -> m^-3
        r.parallel_ion_flux_to_target,
        r.alpha_t,
        r.q_parallel,
        r.heat_flux_perp_to_target,
        r.separatrix_z_effective,
        r.converged,
    )


@Algorithm.register_algorithm(return_keys=list(INVERSE_RETURN_KEYS))
@wraps_ufunc(
    input_units=build_input_units("target_electron_temp", "separatrix_electron_density"),
    return_units=return_units_for(INVERSE_RETURN_KEYS),
    output_core_dims=tuple(() for _ in INVERSE_RETURN_KEYS),
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
    CzLINT_for_fixed_impurities: CzLINT_integrator | None = None,
    mean_charge_for_fixed_impurities: Mean_charge_interpolator | None = None,
    average_ion_mass: Unitfull = 2.0 * ureg.amu,
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    ratio_of_upstream_to_average_poloidal_field: Unitfull = 4.0 / 3.0 * ureg.dimensionless,
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
    p = ModelInputs.from_wrapper_args(**locals())
    r = run_extended_lengyel_solver(p, mode="inverse", pdiv=False)
    return (
        r.impurity_fraction,
        r.parallel_ion_flux_to_target,
        r.neutral_pressure_in_divertor,
        r.alpha_t,
        r.q_parallel,
        r.heat_flux_perp_to_target,
        r.separatrix_z_effective,
        r.converged,
    )


@Algorithm.register_algorithm(return_keys=list(INVERSE_PDIV_RETURN_KEYS))
@wraps_ufunc(
    input_units=build_input_units("target_electron_temp", "neutral_pressure_in_divertor"),
    return_units=return_units_for(INVERSE_PDIV_RETURN_KEYS),
    output_core_dims=tuple(() for _ in INVERSE_PDIV_RETURN_KEYS),
)
def run_inverse_extended_lengyel_model_with_pdiv(
    target_electron_temp: Unitfull,
    power_crossing_separatrix: Unitfull,
    neutral_pressure_in_divertor: Unitfull,
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
    CzLINT_for_fixed_impurities: CzLINT_integrator | None = None,
    mean_charge_for_fixed_impurities: Mean_charge_interpolator | None = None,
    average_ion_mass: Unitfull = 2.0 * ureg.amu,
    sheath_heat_transmission_factor: Unitfull = 7.5 * ureg.dimensionless,
    ratio_of_upstream_to_average_poloidal_field: Unitfull = 4.0 / 3.0 * ureg.dimensionless,
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
    """Calculate the impurity concentration required to reach a given target electron temperature and divertor neutral pressure."""
    p = ModelInputs.from_wrapper_args(**locals())
    r = run_extended_lengyel_solver(p, mode="inverse", pdiv=True)
    return (
        r.impurity_fraction,
        r.separatrix_electron_density * n20_to_m3,  # n20 -> m^-3
        r.parallel_ion_flux_to_target,
        r.alpha_t,
        r.q_parallel,
        r.heat_flux_perp_to_target,
        r.separatrix_z_effective,
        r.separatrix_electron_temp,
        r.converged,
    )
