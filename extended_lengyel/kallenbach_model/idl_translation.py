"""A direct translation of the original IDL raddivmom program (written by A. Kallenbach).

For reference and verification.
"""

import numpy as np
from cfspopcon.algorithm_class import Algorithm, CompositeAlgorithm
from cfspopcon.unit_handling import magnitude_in_units, ureg, wraps_ufunc

from ..adas_data import AtomicSpeciesAdasData

J_to_eV = float(magnitude_in_units(1.0 * ureg.J, ureg.eV))


@Algorithm.register_algorithm(
    return_keys=[
        "s_parallel",
        "parallel_conductive_heat_flux",
        "parallel_convective_heat_flux",
        "impurity_radiated_power",
        "charge_exchange_power_loss",
        "ionization_power_loss",
        "change_in_parallel_convective_heat_flux",
        "electron_density",
        "franck_condon_neutral_flux",
        "fast_neutral_flux",
        "electron_temp",
        "ionization_integral",
        "sound_speed",
        "charge_exchange_integral",
        "mach_number",
        "hydrogen_radiated_power",
        "static_pressure",
        "dynamic_pressure",
        "neutral_density",
        "model_success",
    ]
)
@wraps_ufunc(
    input_units=dict(
        parallel_connection_length=ureg.m,
        divertor_parallel_length=ureg.m,
        flux_tube_cross_section_area_in_divertor=ureg.m**2,
        flux_tube_cross_section_area_out_of_divertor=ureg.m**2,
        electron_density_at_target=ureg.m**-3,
        target_electron_temp=ureg.J,
        sound_speed_at_target=ureg.m / ureg.s,
        ion_mass=ureg.kg,
        conductive_heat_flux_at_target=ureg.W / ureg.m**2,
        franck_condon_neutral_flux_at_target=ureg.m**-2 / ureg.s,
        fast_neutral_flux_at_target=ureg.m**-2 / ureg.s,
        franck_condon_neutral_velocity=ureg.m / ureg.s,
        fast_neutral_velocity=ureg.m / ureg.s,
        z_effective=ureg.dimensionless,
        hydrogen_effective_ionization_energy=ureg.J,
        impurity_fraction=ureg.dimensionless,
        convective_heat_flux_at_target=ureg.W / ureg.m**2,
        kappa_e0=ureg.s**-1 * ureg.m**-1 * ureg.J**-2.5,
        deuterium_adas_data=None,
        impurity_adas_data=None,
        mach_number_at_target=ureg.dimensionless,
        number_of_grid_points=None,
    ),
    return_units=dict(
        s_parallel=ureg.m,
        parallel_conductive_heat_flux=ureg.W / ureg.m**2,
        parallel_convective_heat_flux=ureg.W / ureg.m**2,
        impurity_radiated_power=ureg.W / ureg.m**3,
        charge_exchange_power_loss=ureg.W / ureg.m**3,
        ionization_power_loss=ureg.W / ureg.m**3,
        change_in_parallel_convective_heat_flux=ureg.W / ureg.m**2,
        electron_density=ureg.m**-3,
        franck_condon_neutral_flux=ureg.m**-2 / ureg.s,
        fast_neutral_flux=ureg.m**-2 / ureg.s,
        electron_temp=ureg.J,
        ionization_integral=ureg.m**-2 / ureg.s,
        sound_speed=ureg.m / ureg.s,
        charge_exchange_integral=ureg.J / ureg.m**3,
        mach_number=ureg.dimensionless,
        hydrogen_radiated_power=ureg.W / ureg.m**3,
        static_pressure=ureg.Pa,
        dynamic_pressure=ureg.Pa,
        neutral_density=ureg.m**-3,
        model_success=None,
    ),
    output_core_dims=(
        ("dim_s_parallel",),  # s_parallel
        ("dim_s_parallel",),  # parallel_conductive_heat_flux
        ("dim_s_parallel",),  # parallel_convective_heat_flux
        ("dim_s_parallel",),  # impurity_radiated_power
        ("dim_s_parallel",),  # charge_exchange_power_loss
        ("dim_s_parallel",),  # ionization_power_loss
        ("dim_s_parallel",),  # change_in_parallel_convective_heat_flux
        ("dim_s_parallel",),  # electron_density
        ("dim_s_parallel",),  # franck_condon_neutral_flux
        ("dim_s_parallel",),  # fast_neutral_flux
        ("dim_s_parallel",),  # electron_temp
        ("dim_s_parallel",),  # ionization_integral
        ("dim_s_parallel",),  # sound_speed
        ("dim_s_parallel",),  # charge_exchange_integral
        ("dim_s_parallel",),  # mach_number
        ("dim_s_parallel",),  # hydrogen_radiated_power
        ("dim_s_parallel",),  # static_pressure
        ("dim_s_parallel",),  # dynamic_pressure
        ("dim_s_parallel",),  # neutral_density
        (),  # model_success
    ),
)
def run_kallenbach_idl_translation(  # noqa: PLR0915
    parallel_connection_length,
    divertor_parallel_length,
    flux_tube_cross_section_area_in_divertor,
    flux_tube_cross_section_area_out_of_divertor,
    electron_density_at_target,
    target_electron_temp,
    sound_speed_at_target,
    ion_mass,
    conductive_heat_flux_at_target,
    franck_condon_neutral_flux_at_target,
    fast_neutral_flux_at_target,
    franck_condon_neutral_velocity,
    fast_neutral_velocity,
    z_effective,
    hydrogen_effective_ionization_energy,
    impurity_fraction,
    convective_heat_flux_at_target,
    kappa_e0,
    deuterium_adas_data: AtomicSpeciesAdasData,
    impurity_adas_data: AtomicSpeciesAdasData,
    mach_number_at_target=-1.0,
    number_of_grid_points: int = 12000,
):
    """Run the main loop for the Kallenbach model."""
    # Build the parallel grid
    ds_helper = 3.0e-1 + np.power(np.arange(number_of_grid_points - 1), 2)
    grid_spacing = ds_helper / np.sum(ds_helper) * parallel_connection_length

    s_parallel = np.zeros(number_of_grid_points)

    for i in range(1, len(s_parallel)):
        s_parallel[i] = s_parallel[i - 1] + grid_spacing[i - 1]

    flux_tube_cross_section_area = np.where(
        s_parallel < divertor_parallel_length,
        flux_tube_cross_section_area_in_divertor,
        flux_tube_cross_section_area_out_of_divertor,
    )

    parallel_conductive_heat_flux = np.zeros(number_of_grid_points)
    parallel_convective_heat_flux = np.zeros(number_of_grid_points)
    impurity_radiated_power = np.zeros(number_of_grid_points)
    charge_exchange_power_loss = np.zeros(number_of_grid_points)
    ionization_power_loss = np.zeros(number_of_grid_points)
    change_in_parallel_convective_heat_flux = np.zeros(number_of_grid_points)
    electron_density = np.zeros(number_of_grid_points)
    franck_condon_neutral_flux = np.zeros(number_of_grid_points)
    fast_neutral_flux = np.zeros(number_of_grid_points)
    electron_temp = np.zeros(number_of_grid_points)
    ionization_integral = np.zeros(number_of_grid_points)
    sound_speed = np.zeros(number_of_grid_points)
    charge_exchange_integral = np.zeros(number_of_grid_points)
    mach_number = np.zeros(number_of_grid_points)

    # Diagnostics
    hydrogen_radiated_power = np.zeros(number_of_grid_points)
    static_pressure = np.zeros(number_of_grid_points)
    dynamic_pressure = np.zeros(number_of_grid_points)
    neutral_density = np.zeros(number_of_grid_points)

    parallel_conductive_heat_flux[0] = conductive_heat_flux_at_target
    parallel_convective_heat_flux[0] = convective_heat_flux_at_target
    electron_density[0] = electron_density_at_target
    electron_temp[0] = target_electron_temp
    sound_speed[0] = sound_speed_at_target
    franck_condon_neutral_flux[0] = franck_condon_neutral_flux_at_target
    fast_neutral_flux[0] = fast_neutral_flux_at_target
    charge_exchange_power_loss[0] = 0.0
    ionization_power_loss[0] = 0.0
    change_in_parallel_convective_heat_flux[0] = 0.0
    ionization_integral[0] = 0.0
    charge_exchange_integral[0] = 0.0
    mach_number[0] = mach_number_at_target

    hydrogen_radiated_power[0] = (
        deuterium_adas_data.radiative_power_coeff.unitless_eval(electron_density[0], electron_temp[0] * J_to_eV)
        * neutral_density[0]
        * electron_density[0]
    )
    static_pressure[0] = electron_density_at_target * target_electron_temp
    dynamic_pressure[0] = ion_mass * electron_density_at_target * (mach_number_at_target * sound_speed_at_target) ** 2
    neutral_density[0] = (
        franck_condon_neutral_flux_at_target / franck_condon_neutral_velocity + fast_neutral_flux_at_target / fast_neutral_velocity
    )
    impurity_radiated_power[0] = (
        impurity_adas_data.radiative_power_coeff.unitless_eval(electron_density[0], electron_temp[0] * J_to_eV)
        * impurity_fraction
        * electron_density[0] ** 2
    )

    momentum_factor = electron_density_at_target * (
        ion_mass * (mach_number_at_target * sound_speed_at_target) ** 2 + 2.0 * target_electron_temp
    )

    failed = False

    for i in range(1, len(s_parallel)):
        parallel_conductive_heat_flux[i] = (
            parallel_conductive_heat_flux[i - 1]
            + grid_spacing[i - 1] * (impurity_radiated_power[i - 1] + charge_exchange_power_loss[i - 1] + ionization_power_loss[i - 1])
            - change_in_parallel_convective_heat_flux[i - 1]
        )

        if parallel_conductive_heat_flux[i] < 0.0:
            print(f"parallel_conductive_heat_flux < 0.0 at step {i}")
            failed = True
            break

        if flux_tube_cross_section_area[i] < flux_tube_cross_section_area[i - 1]:
            parallel_conductive_heat_flux[i] = (
                parallel_conductive_heat_flux[i] * flux_tube_cross_section_area[i - 1] / flux_tube_cross_section_area[i]
            )

        franck_condon_neutral_density = franck_condon_neutral_flux[i - 1] / franck_condon_neutral_velocity
        fast_neutral_density = fast_neutral_flux[i - 1] / fast_neutral_velocity
        neutral_density[i] = franck_condon_neutral_density + fast_neutral_density
        ion_density = electron_density[i - 1]

        franck_condon_neutral_ionization_rate = (
            deuterium_adas_data.ionization_rate.unitless_eval(electron_density[i - 1], electron_temp[i - 1] * J_to_eV)
            * electron_density[i - 1]
            * franck_condon_neutral_density
        )
        fast_neutral_ionization_rate = (
            deuterium_adas_data.ionization_rate.unitless_eval(electron_density[i - 1], electron_temp[i - 1] * J_to_eV)
            * electron_density[i - 1]
            * fast_neutral_density
        )
        recombination_rate = (
            deuterium_adas_data.recombination_rate.unitless_eval(electron_density[i - 1], electron_temp[i - 1] * J_to_eV)
            * electron_density[i - 1]
            * ion_density
        )

        ionization_integral[i] = (
            ionization_integral[i - 1]
            + (franck_condon_neutral_ionization_rate + fast_neutral_ionization_rate - recombination_rate) * grid_spacing[i - 1]
        )

        franck_condon_neutral_flux[i] = (
            franck_condon_neutral_flux[i - 1]
            - franck_condon_neutral_ionization_rate * grid_spacing[i - 1]
            + recombination_rate * grid_spacing[i - 1]
        )
        fast_neutral_flux[i] = fast_neutral_flux[i - 1] - fast_neutral_ionization_rate * grid_spacing[i - 1]

        franck_condon_neutral_flux[i] = max(franck_condon_neutral_flux[i], 0.0)
        fast_neutral_flux[i] = max(fast_neutral_flux[i], 0.0)

        change_in_electron_temp = (
            parallel_conductive_heat_flux[i - 1] * z_effective**0.3 / electron_temp[i - 1] ** 2.5 / kappa_e0 * grid_spacing[i - 1]
        )
        electron_temp[i] = electron_temp[i - 1] + change_in_electron_temp

        sound_speed[i] = np.sqrt(2.0 * electron_temp[i - 1] / ion_mass)

        franck_condon_neutral_charge_exchange_rate = (
            deuterium_adas_data.charge_exchange_rate.unitless_eval(electron_density[i - 1], electron_temp[i - 1] * J_to_eV)
            * ion_density
            * franck_condon_neutral_density
        )
        fast_neutral_charge_exchange_rate = (
            deuterium_adas_data.charge_exchange_rate.unitless_eval(electron_density[i - 1], electron_temp[i - 1] * J_to_eV)
            * ion_density
            * fast_neutral_density
        )

        charge_exchange_power_loss[i] = electron_temp[i - 1] * (
            franck_condon_neutral_charge_exchange_rate + fast_neutral_charge_exchange_rate
        )
        ionization_power_loss[i] = hydrogen_effective_ionization_energy * (
            franck_condon_neutral_ionization_rate + fast_neutral_ionization_rate
        )

        charge_exchange_integral[i] = charge_exchange_integral[i - 1] + (
            (franck_condon_neutral_charge_exchange_rate + fast_neutral_charge_exchange_rate + recombination_rate)
            * mach_number[i - 1]
            * sound_speed[i - 1]
            * ion_mass
            * grid_spacing[i - 1]
        )

        electron_density[i] = (momentum_factor - charge_exchange_integral[i - 1]) / (
            ion_mass * (mach_number[i - 1] * sound_speed[i - 1]) ** 2 + 2.0 * electron_temp[i - 1]
        )

        plasma_particle_flux = electron_density_at_target * mach_number_at_target * sound_speed_at_target + ionization_integral[i - 1]
        ion_velocity = plasma_particle_flux / electron_density[i - 1]
        mach_number[i] = ion_velocity / sound_speed[i]
        parallel_convective_heat_flux[i] = (
            -(5.0 * electron_temp[i] * electron_density[i] + 0.5 * ion_mass * electron_density[i] * ion_velocity**2) * ion_velocity
        )
        change_in_parallel_convective_heat_flux[i] = parallel_convective_heat_flux[i] - parallel_convective_heat_flux[i - 1]

        mach_limit = 1.3
        mach_number[i] = min(mach_number[i], mach_limit)

        impurity_radiated_power[i] = (
            impurity_adas_data.radiative_power_coeff.unitless_eval(electron_density[i], electron_temp[i] * J_to_eV)
            * impurity_fraction
            * electron_density[i] ** 2
        )

        # Evaluate terms not used in the integration, but useful for comparison to
        # Figure 4
        hydrogen_radiated_power[i] = (
            deuterium_adas_data.radiative_power_coeff.unitless_eval(electron_density[i], electron_temp[i] * J_to_eV)
            * ion_density
            * electron_density[i]
        )

        # Assume electron_temp = ion_temp
        ion_temp = electron_temp[i]
        static_pressure[i] = electron_density[i] * (electron_temp[i] + ion_temp)
        dynamic_pressure[i] = ion_mass * electron_density[i] * ion_velocity**2

    result = (
        s_parallel,
        parallel_conductive_heat_flux,
        parallel_convective_heat_flux,
        impurity_radiated_power,
        charge_exchange_power_loss,
        ionization_power_loss,
        change_in_parallel_convective_heat_flux,
        electron_density,
        franck_condon_neutral_flux,
        fast_neutral_flux,
        electron_temp,
        ionization_integral,
        sound_speed,
        charge_exchange_integral,
        mach_number,
        hydrogen_radiated_power,
        static_pressure,
        dynamic_pressure,
        neutral_density,
    )

    if not failed:
        return (*result, True)
    else:
        return (*tuple([val * np.nan for val in result]), False)


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "read_deuterium_adas_data",
            "read_impurity_adas_data",
            "calc_parallel_to_perp_factor",
            "calc_lambda_q_from_lambda_int",
            "calc_parallel_heat_flux_at_target",
            "calc_magnetic_field_and_safety_factor",
            "calc_fieldline_pitch_at_omp",
            "calc_flux_tube_cross_section_area_in_divertor",
            "calc_flux_tube_cross_section_area_out_of_divertor",
            "calc_sound_speed_at_target",
            "calc_franck_condon_neutral_velocity",
            "calc_target_density",
            "calc_ion_flux_to_target",
            "calc_flux_density_to_pascals_factor",
            "calc_divertor_neutral_pressure",
            "calc_fast_neutral_velocity",
            "calc_neutral_recycling_fluxes",
            "calc_z_effective",
            "calc_q_convective_tar",
            "calc_q_conductive_tar",
            "calc_kappa_e0",
        ]
    ],
    name="initialize_kallenbach_idl_translation",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "calc_upstream_density",
            "calc_upstream_temp",
            "calc_q_total",
            "calc_upstream_q_total",
            "calc_target_q_total",
            "calc_SOL_momentum_loss_fraction",
            "calc_SOL_power_loss_fraction",
            "calc_power_crossing_separatrix_from_heat_flux_in_flux_tube",
            "calc_radiative_efficiency",
            "calc_qdet_ext_7a",
            "calc_qdet_ext_7b",
        ]
    ],
    name="postprocess_kallenbach_idl_translation",
    register=True,
)

CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "initialize_kallenbach_idl_translation",
            "run_kallenbach_idl_translation",
            "postprocess_kallenbach_idl_translation",
        ]
    ],
    name="kallenbach_idl_translation",
    register=True,
)
