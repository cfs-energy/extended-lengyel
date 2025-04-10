"""Run the spatial Lengyel model from the target to the upstream point (usually OMP)."""

import warnings
from typing import Literal

import numpy as np
from cfspopcon import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import ureg, wraps_ufunc
from scipy.integrate import solve_ivp

from .spatial_lengyel_ode import spatial_lengyel_change_in_state_vector


@Algorithm.register_algorithm(
    return_keys=[
        "divertor_entrance_parallel_heat_flux",
        "divertor_entrance_electron_density",
        "divertor_entrance_electron_temp",
        "q_parallel",
        "separatrix_electron_density",
        "separatrix_electron_temp",
    ]
)
@wraps_ufunc(
    input_units=dict(
        electron_density_at_cc_interface=ureg.m**-3,
        electron_temp_at_cc_interface=ureg.eV,
        parallel_heat_flux_at_cc_interface=ureg.W / ureg.m**2,
        impurity_fraction=ureg.dimensionless,
        kappa_e0=ureg.W / (ureg.m * ureg.eV**3.5),
        z_effective=ureg.dimensionless,
        divertor_broadening_factor=ureg.dimensionless,
        s_parallel_at_cc_interface=ureg.m,
        divertor_parallel_length=ureg.m,
        parallel_connection_length=ureg.m,
        impurity_adas_data=None,
        ode_method=None,
        ode_rtol=None,
        ode_atol=None,
    ),
    return_units=dict(
        divertor_entrance_parallel_heat_flux=ureg.W * ureg.m**-2,
        divertor_entrance_electron_density=ureg.m**-3,
        divertor_entrance_electron_temp=ureg.eV,
        q_parallel=ureg.W * ureg.m**-2,
        separatrix_electron_density=ureg.m**-3,
        separatrix_electron_temp=ureg.eV,
    ),
    output_core_dims=(
        (),
        (),
        (),
        (),
        (),
        (),
    ),
)
def run_spatial_lengyel_model(
    electron_density_at_cc_interface,
    electron_temp_at_cc_interface,
    parallel_heat_flux_at_cc_interface,
    impurity_fraction,
    kappa_e0,
    z_effective,
    divertor_broadening_factor,
    s_parallel_at_cc_interface,
    divertor_parallel_length,
    parallel_connection_length,
    impurity_adas_data,
    ode_method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "BDF",
    ode_rtol: float = 1e-3,
    ode_atol: float = 1e-6,
):
    """Integrate the Lengyel model along a fieldline."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res1 = solve_ivp(
                spatial_lengyel_change_in_state_vector,
                (s_parallel_at_cc_interface, divertor_parallel_length),
                y0=[
                    parallel_heat_flux_at_cc_interface,
                    electron_density_at_cc_interface,
                    electron_temp_at_cc_interface,
                ],
                args=(
                    impurity_fraction,
                    kappa_e0,
                    z_effective,
                    impurity_adas_data,
                ),
                method=ode_method,
                dense_output=True,
                rtol=ode_rtol,
                atol=ode_atol,
            )

            res2 = solve_ivp(
                spatial_lengyel_change_in_state_vector,
                (divertor_parallel_length, parallel_connection_length),
                y0=[
                    res1.y[0, -1] * divertor_broadening_factor,  # parallel_heat_flux
                    res1.y[1, -1],  # electron_density
                    res1.y[2, -1],  # electron_temp
                ],
                args=(
                    impurity_fraction,
                    kappa_e0,
                    z_effective,
                    impurity_adas_data,
                ),
                method=ode_method,
                dense_output=True,
                rtol=ode_rtol,
                atol=ode_atol,
            )
            model_success = res1.success and res2.success
        except (ValueError, IndexError, RuntimeWarning):
            model_success = False

    if model_success:
        divertor_entrance_parallel_heat_flux = res1.y[0, -1]
        divertor_entrance_electron_density = res1.y[1, -1]
        divertor_entrance_electron_temp = res1.y[2, -1]

        q_parallel = res2.y[0, -1]
        separatrix_electron_density = res2.y[1, -1]
        separatrix_electron_temp = res2.y[2, -1]

        return (
            divertor_entrance_parallel_heat_flux,
            divertor_entrance_electron_density,
            divertor_entrance_electron_temp,
            q_parallel,
            separatrix_electron_density,
            separatrix_electron_temp,
        )
    else:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "read_impurity_adas_data",
            "calc_parallel_to_perp_factor",
            "calc_parallel_heat_flux_at_target",
            "calc_sound_speed_at_target",
            "calc_target_density",
            "calc_z_effective",
            "calc_kappa_e0",
            "calc_electron_temp_from_cc_fit",
            "calc_electron_density_from_cc_fit",
            "calc_power_loss_from_cc_fit",
            "calc_parallel_heat_flux_from_conv_loss",
            "ignore_s_parallel_width_for_cc_interface",
        ]
    ],
    name="initialize_spatial_lengyel_model",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "calc_momentum_loss_from_cc_fit",
            "calc_SOL_power_loss_fraction",
        ]
    ],
    name="postprocess_spatial_lengyel_model",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "initialize_spatial_lengyel_model",
            "run_spatial_lengyel_model",
            "postprocess_spatial_lengyel_model",
        ]
    ],
    name="spatial_lengyel_model",
    register=True,
)
