"""Shared unit declarations for the wraps_ufunc decorators of the public model wrappers."""

from cfspopcon.unit_handling import ureg

# Units for the mode-specific leading argument
DRIVING_UNITS = {
    "impurity_fraction": ureg.dimensionless,  # forward model
    "target_electron_temp": ureg.eV,  # inverse solve
}

# Units for the arguments shared by both models, in signature order
COMMON_TAIL_INPUT_UNITS = dict(
    divertor_broadening_factor=ureg.dimensionless,
    CzLINT_for_seed_impurities=None,
    mean_charge_for_seed_impurities=None,
    magnetic_field_on_axis=ureg.T,
    plasma_current=ureg.MA,
    parallel_connection_length=ureg.m,
    divertor_parallel_length=ureg.m,
    major_radius=ureg.m,
    minor_radius=ureg.m,
    elongation_psi95=ureg.dimensionless,
    triangularity_psi95=ureg.dimensionless,
    target_angle_of_incidence=ureg.degree,
    fraction_of_P_SOL_to_divertor=ureg.dimensionless,
    CzLINT_for_fixed_impurities=None,
    mean_charge_for_fixed_impurities=None,
    average_ion_mass=ureg.amu,
    sheath_heat_transmission_factor=ureg.dimensionless,
    ratio_of_upstream_to_average_poloidal_field=ureg.dimensionless,
    wall_temperature=ureg.K,
    SOL_conduction_fraction=ureg.dimensionless,
    ratio_of_molecular_to_ion_mass=ureg.dimensionless,
    separatrix_mach_number=ureg.dimensionless,
    separatrix_ratio_of_ion_to_electron_temp=ureg.dimensionless,
    separatrix_ratio_of_electron_to_ion_density=ureg.dimensionless,
    target_ratio_of_ion_to_electron_temp=ureg.dimensionless,
    target_ratio_of_electron_to_ion_density=ureg.dimensionless,
    target_mach_number=ureg.dimensionless,
    toroidal_flux_expansion=ureg.dimensionless,
    iterations=None,
)


def build_input_units(driving_key: str) -> dict:
    """Build the full wraps_ufunc input_units dict for one model variant.

    Args:
        driving_key: "impurity_fraction" (forward) or "target_electron_temp" (inverse).
    """
    return {
        driving_key: DRIVING_UNITS[driving_key],
        "power_crossing_separatrix": ureg.MW,
        "separatrix_electron_density": ureg.m**-3,
        **COMMON_TAIL_INPUT_UNITS,
    }


# Canonical units for every possible return key; each model selects its keys below.
RETURN_UNITS = {
    "impurity_fraction": ureg.dimensionless,
    "target_electron_temp": ureg.eV,
    "parallel_ion_flux_to_target": ureg.m**-2 * ureg.s**-1,
    "neutral_pressure_in_divertor": ureg.Pa,
    "alpha_t": ureg.dimensionless,
    "q_parallel": ureg.W / ureg.m**2,
    "heat_flux_perp_to_target": ureg.W / ureg.m**2,
    "separatrix_z_effective": ureg.dimensionless,
    "converged": None,
}

FORWARD_RETURN_KEYS = (
    "target_electron_temp",
    "parallel_ion_flux_to_target",
    "neutral_pressure_in_divertor",
    "alpha_t",
    "q_parallel",
    "heat_flux_perp_to_target",
    "separatrix_z_effective",
    "converged",
)
INVERSE_RETURN_KEYS = (
    "impurity_fraction",
    "parallel_ion_flux_to_target",
    "neutral_pressure_in_divertor",
    "alpha_t",
    "q_parallel",
    "heat_flux_perp_to_target",
    "separatrix_z_effective",
    "converged",
)


def return_units_for(keys: tuple[str, ...]) -> dict:
    """Select the wraps_ufunc return_units dict for the given return keys."""
    return {key: RETURN_UNITS[key] for key in keys}
