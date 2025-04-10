"""Run the reformulated Kallenbach model from the target to the upstream point (usually OMP)."""

import warnings
from typing import Literal

import numpy as np
from cfspopcon import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import ureg, wraps_ufunc
from scipy.integrate import solve_ivp

from .kallenbach_ode import kallenbach_change_in_state_vector


@Algorithm.register_algorithm(
    return_keys=[
        "s_parallel",
        "electron_density",
        "ion_velocity",
        "neutral_density",
        "electron_temp",
        "parallel_heat_flux",
        "divertor_entrance_electron_temp",
        "divertor_entrance_parallel_heat_flux",
        "model_success",
    ]
)
@wraps_ufunc(
    input_units=dict(
        electron_density_at_target=ureg.m**-3,
        sound_speed_at_target=ureg.m / ureg.s,
        neutral_density_at_target=ureg.m**-3,
        target_electron_temp=ureg.eV,
        parallel_heat_flux_at_target=ureg.W / ureg.m**2,
        impurity_fraction=ureg.dimensionless,
        kappa_e0=ureg.W / (ureg.m * ureg.eV**3.5),
        z_effective=ureg.dimensionless,
        ion_mass=ureg.kg,
        franck_condon_neutral_velocity=ureg.m / ureg.s,
        hydrogen_effective_ionization_energy=ureg.eV,
        divertor_broadening_factor=ureg.dimensionless,
        divertor_parallel_length=ureg.m,
        parallel_connection_length=ureg.m,
        deuterium_adas_data=None,
        impurity_adas_data=None,
        ode_method=None,
        number_of_grid_points=None,
        ion_velocity_epsilon=None,
        ode_rtol=None,
        ode_atol=None,
    ),
    return_units=dict(
        s_parallel=ureg.m,
        electron_density=ureg.m**-3,
        ion_velocity=ureg.m / ureg.s,
        neutral_density=ureg.m**-3,
        electron_temp=ureg.eV,
        parallel_heat_flux=ureg.W * ureg.m**-2,
        divertor_entrance_electron_temp=ureg.eV,
        divertor_entrance_parallel_heat_flux=ureg.W * ureg.m**-2,
        model_success=None,
    ),
    output_core_dims=(
        ("dim_s_parallel",),
        ("dim_s_parallel",),
        ("dim_s_parallel",),
        ("dim_s_parallel",),
        ("dim_s_parallel",),
        ("dim_s_parallel",),
        (),
        (),
        (),
    ),
)
def run_kallenbach_model(
    electron_density_at_target,
    sound_speed_at_target,
    neutral_density_at_target,
    target_electron_temp,
    parallel_heat_flux_at_target,
    impurity_fraction,
    kappa_e0,
    z_effective,
    ion_mass,
    franck_condon_neutral_velocity,
    hydrogen_effective_ionization_energy,
    divertor_broadening_factor,
    divertor_parallel_length,
    parallel_connection_length,
    deuterium_adas_data,
    impurity_adas_data,
    ode_method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "BDF",
    number_of_grid_points: int = 50,
    ion_velocity_epsilon: float = 1e-6,
    ode_rtol: float = 1e-3,
    ode_atol: float = 1e-6,
):
    """Integrate the radiation-convection-conduction model along a fieldline."""

    def log_range(minimum, maximum, num=number_of_grid_points):
        x_range = np.logspace(np.log10(minimum), np.log10(maximum), num=num)
        return np.maximum(np.minimum(x_range, maximum), minimum)

    t_eval_1 = log_range(1e-6, divertor_parallel_length, num=number_of_grid_points // 2)
    t_eval_2 = log_range(divertor_parallel_length, parallel_connection_length, num=number_of_grid_points - number_of_grid_points // 2)

    ion_velocity_at_target = -sound_speed_at_target * (1 - ion_velocity_epsilon)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res1 = solve_ivp(
                kallenbach_change_in_state_vector,
                (0.0, divertor_parallel_length),
                y0=[
                    electron_density_at_target,
                    ion_velocity_at_target,
                    neutral_density_at_target,
                    target_electron_temp,
                    parallel_heat_flux_at_target,
                ],
                args=(
                    impurity_fraction,
                    kappa_e0,
                    z_effective,
                    ion_mass,
                    franck_condon_neutral_velocity,
                    hydrogen_effective_ionization_energy,
                    deuterium_adas_data,
                    impurity_adas_data,
                ),
                method=ode_method,
                dense_output=True,
                rtol=ode_rtol,
                atol=ode_atol,
            )
            y1 = res1.sol(t_eval_1)

            divertor_entrance_electron_density = y1[0, -1]
            divertor_entrance_ion_velocity = y1[1, -1]
            divertor_entrance_neutral_density = y1[2, -1]
            divertor_entrance_electron_temp = y1[3, -1]
            # Define q_div to be the value upstream of the divertor entrance
            divertor_entrance_parallel_heat_flux = y1[4, -1] * divertor_broadening_factor

            res2 = solve_ivp(
                kallenbach_change_in_state_vector,
                (divertor_parallel_length, parallel_connection_length),
                y0=[
                    divertor_entrance_electron_density,  # electron_density
                    divertor_entrance_ion_velocity,  # ion_velocity
                    divertor_entrance_neutral_density,  # neutral_density
                    divertor_entrance_electron_temp,  # electron_temp
                    divertor_entrance_parallel_heat_flux,  # parallel_heat_flux
                ],
                args=(
                    impurity_fraction,
                    kappa_e0,
                    z_effective,
                    ion_mass,
                    franck_condon_neutral_velocity,
                    hydrogen_effective_ionization_energy,
                    deuterium_adas_data,
                    impurity_adas_data,
                ),
                method=ode_method,
                dense_output=True,
                rtol=ode_rtol,
                atol=ode_atol,
            )
            y2 = res2.sol(t_eval_2)

            model_success = res1.success and res2.success
        except (ValueError, IndexError, RuntimeWarning):
            model_success = False

    if model_success:
        s_parallel = np.append(t_eval_1, t_eval_2)
        electron_density, ion_velocity, neutral_density, electron_temp, parallel_heat_flux = np.append(y1, y2, axis=1)
        return (
            s_parallel,
            electron_density,
            ion_velocity,
            neutral_density,
            electron_temp,
            parallel_heat_flux,
            divertor_entrance_electron_temp,
            divertor_entrance_parallel_heat_flux,
            model_success,
        )
    else:
        s_parallel = np.append(t_eval_1, t_eval_2)
        nan_array = s_parallel * np.nan
        return (s_parallel, nan_array, nan_array, nan_array, nan_array, nan_array, np.nan, np.nan, model_success)


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "read_deuterium_adas_data",
            "read_impurity_adas_data",
            "calc_parallel_to_perp_factor",
            "calc_parallel_heat_flux_at_target",
            "calc_sound_speed_at_target",
            "calc_franck_condon_neutral_velocity",
            "calc_target_density",
            "calc_neutral_density_at_target",
            "calc_z_effective",
            "calc_kappa_e0",
        ]
    ],
    name="initialize_kallenbach_model",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "calc_ion_temp",
            "calc_ion_density",
            "calc_static_pressure",
            "calc_dynamic_pressure",
            "calc_total_pressure",
            "calc_separatrix_total_pressure",
            "calc_sound_speed",
            "calc_mach_number",
            "calc_charge_exchange_power_loss_density",
            "calc_ionization_power_loss_density",
            "calc_hydrogen_radiated_power_density",
            "calc_impurity_radiated_power_density",
            "calc_upstream_density",
            "calc_upstream_temp",
            "calc_upstream_q_total",
            "calc_target_q_total",
            "calc_SOL_momentum_loss_fraction",
            "calc_SOL_power_loss_fraction",
            "calc_convected_and_conducted_heat_flux",
            "calc_values_at_conduction_convection_interface",
            "calc_SOL_power_loss_fraction_in_convection_layer",
            "calc_SOL_momentum_loss_fraction_in_convection_layer",
            "calc_electron_temp_ratio_in_convection_layer",
            "calc_electron_density_ratio_in_convection_layer",
        ]
    ],
    name="postprocess_kallenbach_model",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "initialize_kallenbach_model",
            "run_kallenbach_model",
            "postprocess_kallenbach_model",
        ]
    ],
    name="kallenbach_model",
    register=True,
)
