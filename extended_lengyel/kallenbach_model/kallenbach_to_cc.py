"""Run the reformulated Kallenbach model up to the conduction-convection interface."""

import warnings
from typing import Literal

import numpy as np
from cfspopcon import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import ureg, wraps_ufunc
from scipy.integrate import solve_ivp

from .kallenbach_ode import eV_to_J, kallenbach_change_in_state_vector


@Algorithm.register_algorithm(
    return_keys=[
        "s_parallel_at_cc_interface",
        "electron_density_at_cc_interface",
        "ion_velocity_at_cc_interface",
        "neutral_density_at_cc_interface",
        "electron_temp_at_cc_interface",
        "parallel_heat_flux_at_cc_interface",
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
        divertor_parallel_length=ureg.m,
        deuterium_adas_data=None,
        impurity_adas_data=None,
        ode_method=None,
        ion_velocity_epsilon=None,
        ode_rtol=None,
        ode_atol=None,
        conducted_fraction_at_cc_interface=None,
    ),
    return_units=dict(
        s_parallel=ureg.m,
        electron_density=ureg.m**-3,
        ion_velocity=ureg.m / ureg.s,
        neutral_density=ureg.m**-3,
        electron_temp=ureg.eV,
        parallel_heat_flux=ureg.W * ureg.m**-2,
        model_success=None,
    ),
    output_core_dims=(
        (),
        (),
        (),
        (),
        (),
        (),
        (),
    ),
)
def run_kallenbach_model_to_cc(
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
    divertor_parallel_length,
    deuterium_adas_data,
    impurity_adas_data,
    ode_method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "BDF",
    ion_velocity_epsilon: float = 1e-6,
    ode_rtol: float = 1e-3,
    ode_atol: float = 1e-6,
    conducted_fraction_at_cc_interface: float = 0.99,
):
    """Integrate the radiation-convection-conduction model along a fieldline, up to the convective-conduction interface."""
    ion_velocity_at_target = -sound_speed_at_target * (1 - ion_velocity_epsilon)

    def conduction_dominated(
        _,
        state_vector,
        *args,
    ):
        """Determine the point where conduction-dominated region starts."""
        electron_density, ion_velocity, _, electron_temp, parallel_heat_flux = state_vector

        parallel_convective_heat_flux = (
            -(5.0 * electron_density * (electron_temp * eV_to_J) + 0.5 * ion_mass * electron_density * ion_velocity**2) * ion_velocity
        )
        parallel_conductive_heat_flux = parallel_heat_flux - parallel_convective_heat_flux

        return parallel_conductive_heat_flux / parallel_heat_flux - conducted_fraction_at_cc_interface

    conduction_dominated.terminal = True

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = solve_ivp(
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
                events=conduction_dominated,
            )
            model_success = res.success
            assert len(res.t_events) == 1
        except (ValueError, IndexError, RuntimeWarning, AssertionError):
            model_success = False

    if model_success:
        s_parallel = res.t[-1]
        electron_density, ion_velocity, neutral_density, electron_temp, parallel_heat_flux = res.y[:, -1]

        return s_parallel, electron_density, ion_velocity, neutral_density, electron_temp, parallel_heat_flux, model_success
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, model_success


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "calc_SOL_power_loss_fraction_in_convection_layer",
            "calc_SOL_momentum_loss_fraction_in_convection_layer",
            "calc_electron_temp_ratio_in_convection_layer",
            "calc_electron_density_ratio_in_convection_layer",
        ]
    ],
    name="postprocess_kallenbach_model_at_cc",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "initialize_kallenbach_model",
            "run_kallenbach_model_to_cc",
            "postprocess_kallenbach_model_at_cc",
        ]
    ],
    name="kallenbach_model_to_cc",
    register=True,
)
