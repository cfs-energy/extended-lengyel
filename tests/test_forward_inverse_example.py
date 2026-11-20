"""Inverse-to-forward round trip of the iterative solver, run through the cfspopcon algorithms.

The inverse solve finds the impurity concentration that gives a chosen target electron temperature;
the forward solve, given that concentration, recovers the temperature. Needs a built radas dataset
(set EXTENDED_LENGYEL_RADAS_DIR or run from a directory containing radas_dir).
"""

import cfspopcon
import numpy as np
import xarray as xr
from cfspopcon.unit_handling import Quantity, ureg

import extended_lengyel

SETUP = ["set_radas_dir", "read_atomic_data", "build_CzLINT_for_seed_impurities", "build_mean_charge_for_seed_impurities"]
TARGET_ELECTRON_TEMP_EV = 5.0


def test_inverse_then_forward_recovers_target_temperature():
    """Inverse for a 5 eV target, then forward at the impurity fraction it found, returns 5 eV."""
    inverse = cfspopcon.CompositeAlgorithm.from_list([*SETUP, "run_inverse_extended_lengyel_model"])
    forward = cfspopcon.CompositeAlgorithm.from_list([*SETUP, "run_forward_extended_lengyel_model"])

    species, weights = extended_lengyel.config.setup_impurities(["Nitrogen", "Argon"], [1.0, 0.05])
    ds = xr.Dataset(
        data_vars=dict(
            seed_impurity_weights=weights,
            seed_impurity_species=species,
            target_electron_temp=Quantity(TARGET_ELECTRON_TEMP_EV, ureg.eV),
            separatrix_electron_density=Quantity(3.3e19, ureg.m**-3),
            power_crossing_separatrix=Quantity(5.5, ureg.MW),
            fraction_of_P_SOL_to_divertor=2 / 3,
            divertor_broadening_factor=3.0,
            plasma_current=Quantity(1.0, ureg.MA),
            magnetic_field_on_axis=Quantity(2.5, ureg.T),
            major_radius=Quantity(1.65, ureg.m),
            minor_radius=Quantity(0.5, ureg.m),
            parallel_connection_length=Quantity(20, ureg.m),
            divertor_parallel_length=Quantity(5.0, ureg.m),
            elongation_psi95=1.6,
            triangularity_psi95=0.3,
            target_angle_of_incidence=Quantity(3.0, ureg.degree),
        )
    )

    inverse.validate_inputs(ds)
    inverse_ds = inverse.update_dataset(ds)
    assert inverse_ds["converged"].item(), "inverse solve did not converge"

    forward_input = ds.drop_vars("target_electron_temp").assign(impurity_fraction=inverse_ds["impurity_fraction"])
    forward.validate_inputs(forward_input)
    forward_ds = forward.update_dataset(forward_input)
    assert forward_ds["converged"].item(), "forward solve did not converge"

    recovered = forward_ds["target_electron_temp"].item().to(ureg.eV).magnitude
    assert np.isclose(recovered, TARGET_ELECTRON_TEMP_EV, rtol=1e-3)
