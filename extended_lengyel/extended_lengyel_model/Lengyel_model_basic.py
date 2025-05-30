"""Run the basic Lengyel model."""

import numpy as np
import xarray as xr
from cfspopcon import Algorithm, CompositeAlgorithm

from ..xr_helpers import item


@Algorithm.register_algorithm(return_keys=["impurity_fraction"])
def run_basic_lengyel_model(
    q_parallel,
    kappa_e0,
    separatrix_electron_density,
    separatrix_electron_temp,
    target_electron_temp,
    CzLINT_for_seed_impurities,
    SOL_power_loss_fraction,
    kappa_z,
):
    """Calculate the impurity fraction required to radiate a given fraction of the power in the scrape-off-layer, using the basic Lengyel model."""
    LINT_t_u = item(CzLINT_for_seed_impurities)(target_electron_temp, separatrix_electron_temp)

    kappa = kappa_e0 / kappa_z

    c_z = (q_parallel**2 - ((1.0 - SOL_power_loss_fraction) * q_parallel) ** 2) / (
        2.0 * kappa * separatrix_electron_density**2 * separatrix_electron_temp**2 * LINT_t_u
    )

    c_z = xr.where(c_z < 0.0, np.nan, c_z)

    return c_z


CompositeAlgorithm(
    algorithms=[
        Algorithm.get_algorithm(alg)
        for alg in [
            "set_radas_dir",
            "read_atomic_data",
            "set_single_impurity_species",
            "build_CzLINT_for_seed_impurities",
            "calc_kappa_e0",
            "calc_Goldston_kappa_z",
            "calc_momentum_loss_from_cc_fit",
            "calc_separatrix_electron_temp_no_broadening",
            "calc_separatrix_total_pressure_LG",
            "calc_required_power_loss_fraction",
            "run_basic_lengyel_model",
        ]
    ],
    name="basic_lengyel_model",
    register=True,
)
