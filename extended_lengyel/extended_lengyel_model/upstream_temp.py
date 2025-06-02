"""Calculate the upstream temperature, taking into account divertor broadening."""

from cfspopcon import Algorithm
from cfspopcon.formulas.scrape_off_layer.two_point_model.separatrix_pressure import calc_upstream_total_pressure
import numpy as np
from cfspopcon.unit_handling import wraps_ufunc, ureg
from scipy.interpolate import InterpolatedUnivariateSpline


@Algorithm.register_algorithm(return_keys=["separatrix_electron_temp"])
def calc_separatrix_electron_temp_no_broadening(
    target_electron_temp,
    q_parallel,
    parallel_connection_length,
    kappa_e0,
    kappa_z,
    SOL_conduction_fraction=1.0,
):
    """Calculate the electron temperature at the separatrix."""
    kappa = kappa_e0 / kappa_z

    separatrix_electron_temp = (
        target_electron_temp**3.5 + 3.5 * SOL_conduction_fraction * q_parallel * parallel_connection_length / kappa
    ) ** (2 / 7)

    return separatrix_electron_temp


@Algorithm.register_algorithm(return_keys=["divertor_entrance_electron_temp", "separatrix_electron_temp"])
def calc_separatrix_electron_temp_with_broadening(
    electron_temp_at_cc_interface,
    q_parallel,
    divertor_broadening_factor,
    parallel_connection_length,
    divertor_parallel_length,
    kappa_e0,
    kappa_z,
    SOL_conduction_fraction=1.0,
):
    """Calculate the electron temperature at the divertor entrance and separatrix."""
    kappa = kappa_e0 / kappa_z

    divertor_entrance_electron_temp = (
        electron_temp_at_cc_interface**3.5
        + 3.5 * SOL_conduction_fraction * q_parallel / divertor_broadening_factor * divertor_parallel_length / kappa
    ) ** (2 / 7)

    separatrix_electron_temp = (
        divertor_entrance_electron_temp**3.5
        + 3.5 * SOL_conduction_fraction * q_parallel * (parallel_connection_length - divertor_parallel_length) / kappa
    ) ** (2 / 7)

    return divertor_entrance_electron_temp, separatrix_electron_temp


@wraps_ufunc(
    input_units=dict(
        start_temp=ureg.eV,
        initial_guess_temp=ureg.eV,
        start_q_parallel=ureg.W / ureg.m**2,
        total_pressure=ureg.eV / ureg.m**3,
        impurity_fraction=ureg.dimensionless,
        kappa=ureg.W / (ureg.eV**3.5 * ureg.m),
        parallel_length=ureg.m,
        CzLINT_for_seed_impurities=None,
        search_factor=None,
        resolution=None,
    ),
    return_units=dict(separatrix_electron_temp=ureg.eV),
)
def calc_electron_temp_from_parallel_length(
    start_temp,
    initial_guess_temp,
    start_q_parallel,
    total_pressure,
    impurity_fraction,
    kappa,
    parallel_length,
    CzLINT_for_seed_impurities,
    search_factor: int = 2,
    resolution: int = 20,
):
    """Compute an electron temperature which is consistent with the parallel length."""
    Te_tests = np.linspace(start=start_temp, stop=initial_guess_temp * search_factor, num=resolution)
    L_int_Te = np.zeros(Te_tests.size)
    for i, Te in enumerate(Te_tests):
        L_int_Te[i] = CzLINT_for_seed_impurities.unitless_eval(start_temp, Te)

    q_start_squared = start_q_parallel**2
    dq_squared = 2.0 * kappa * total_pressure**2 * impurity_fraction * L_int_Te

    q_Te = np.sqrt(q_start_squared + dq_squared)
    L_par_integrand = kappa * Te_tests**2.5 / q_Te

    interpolator = InterpolatedUnivariateSpline(Te_tests, L_par_integrand)
    L_par_vals = np.zeros(Te_tests.size)
    for i, Te in enumerate(Te_tests):
        L_par_vals[i] = interpolator.integral(start_temp, Te)

    Te_stop = InterpolatedUnivariateSpline(L_par_vals, Te_tests)(parallel_length)

    assert Te_stop < initial_guess_temp * search_factor
    return Te_stop


@Algorithm.register_algorithm(return_keys=["separatrix_total_pressure"])
def calc_separatrix_total_pressure_LG(
    separatrix_electron_density,
    separatrix_electron_temp,
    separatrix_ratio_of_ion_to_electron_temp=1.0,
    separatrix_ratio_of_electron_to_ion_density=1.0,
    separatrix_mach_number=0.0,
):
    """Calculate the total (electron + ion, static + dynamic) pressure at the separatrix."""
    return calc_upstream_total_pressure(
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        upstream_ratio_of_ion_to_electron_temp=separatrix_ratio_of_ion_to_electron_temp,
        upstream_ratio_of_electron_to_ion_density=separatrix_ratio_of_electron_to_ion_density,
        upstream_mach_number=separatrix_mach_number,
    )
