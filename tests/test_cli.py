from pathlib import Path
import numpy as np
import yaml
from extended_lengyel.cli import run_extended_lengyel
from cfspopcon.unit_handling import Quantity

def test_cli(tmp_path):
    
    yaml_text = \
"""
algorithms:
  - calc_magnetic_field_and_safety_factor
  - calc_fieldline_pitch_at_omp
  - set_radas_dir
  - read_atomic_data
  - calc_kappa_e0
  - build_CzLINT_for_seed_impurities
  - build_mean_charge_for_seed_impurities
  - build_CzLINT_for_fixed_impurities
  - build_mean_charge_for_fixed_impurities
  - calc_momentum_loss_from_cc_fit
  - calc_power_loss_from_cc_fit
  - calc_electron_temp_from_cc_fit
  - run_extended_lengyel_model_with_S_Zeff_and_alphat_correction
  - calc_sound_speed_at_target
  - calc_target_density
  - calc_flux_density_to_pascals_factor
  - calc_parallel_to_perp_factor
  - calc_ion_flux_to_target
  - calc_divertor_neutral_pressure
  - calc_heat_flux_perp_to_target
input:
    # Length along a magnetic fieldline from divertor target to
    # divertor entrance.
    divertor_parallel_length: 5.0 m
    # Length along a magnetic fieldline from divertor target to
    # outboard midplane or stagnation point.
    parallel_connection_length: 20 m
    # Major radius
    major_radius: 1.65 m
    # Minor radius
    minor_radius: 0.5 m
    # Elongation at psiN = 0.95
    elongation_psi95: 1.6
    # Triangularity at psiN = 0.95
    triangularity_psi95: 0.3
    # Magnetic field on axis
    magnetic_field_on_axis: 2.5 T
    # Plasma current
    plasma_current: 1.0 MA
    # Ratio of Bpol_OMP / Bpol_avg
    ratio_of_upstream_to_average_poloidal_field: 4/3
    # Main ion mass (or average mass, i.e. DT = 2.5amu)
    ion_mass: 2.0 amu
    # List of impurities used for seeding, and their relative concentrations.
    seed_impurity_species: ["Nitrogen", "Argon"]
    seed_impurity_weights: [1.0, 0.05]
    # List of background impurities, and their concentration relative to the electron density.
    fixed_impurity_species: "Tungsten"
    fixed_impurity_weights: 1.5e-5
    # Take any nearest ne or ne_tau value for the atomic data, regardless of how close it is
    # to the reference value.
    rtol_nearest_for_atomic_data: 1.0
    # Impurity residence time multiplied by electron density
    reference_ne_tau: 0.5 ms n20
    # gamma_sh
    sheath_heat_transmission_factor: 8.0
    # Angle of incidence between magnetic fieldline and divertor target
    target_angle_of_incidence: 3 degree
    # Ratio between lambda_INT and lambda_q
    divertor_broadening_factor: 3.0
    # Power crossing separatrix
    power_crossing_separatrix: 5.5MW
    # Fraction of power directed to outer divertor
    fraction_of_P_SOL_to_divertor: 2/3
    # Upstream electron density
    separatrix_electron_density: 3.3e19/m^3
    # Electron temperature at target
    target_electron_temp: 2.34eV
"""
    config_file = tmp_path / "input.yml"
    output_file = tmp_path / "output.yml"
    Path(config_file).write_text(yaml_text)

    run_extended_lengyel(config_file, output_file)

    with open(output_file, "r") as f:
        output = yaml.load(f, Loader=yaml.SafeLoader)

    c_N = output["impurity_fraction"]["seed_impurity"]["Nitrogen"]
    c_Ar = output["impurity_fraction"]["seed_impurity"]["Argon"]

    assert np.isclose(c_N, 0.03674605763789175)
    assert np.isclose(c_Ar, 0.0018373028818945876)
    assert np.isclose(c_N / c_Ar, 1 / 0.05)

    output_file_2 = tmp_path / "output2.yml"
    run_extended_lengyel(config_file, output_file_2, cli_args=dict(ion_mass="2.5 amu"))

    with open(output_file_2, "r") as f:
        output = yaml.load(f, Loader=yaml.SafeLoader)

    assert np.isclose(Quantity(output["ion_mass"]), Quantity(2.5, "amu"))
