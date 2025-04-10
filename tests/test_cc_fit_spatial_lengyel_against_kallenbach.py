import cfspopcon
import numpy as np
import xarray as xr
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import ureg

import extended_lengyel
import extended_lengyel.directories


def build_dataset(
    run_lengyel_model=True,
    **overrides,
):
    if run_lengyel_model:
        algorithm = cfspopcon.CompositeAlgorithm.from_list(
            [
                "initialize_kallenbach_model",
                "calc_electron_temp_from_cc_fit",
                "calc_electron_density_from_cc_fit",
                "calc_power_loss_from_cc_fit",
                "calc_parallel_heat_flux_from_conv_loss",
                "ignore_s_parallel_width_for_cc_interface",
                "run_spatial_lengyel_model",
                "postprocess_spatial_lengyel_model",
            ]
        )
    else:
        algorithm = cfspopcon.CompositeAlgorithm.from_list(
            ["initialize_kallenbach_model", "run_kallenbach_model", "postprocess_kallenbach_model"]
        )

    ds = xr.Dataset(
        data_vars=extended_lengyel.read_config(
            elements=["base", "machine_geometry", "target_constraints", "field_at_omp"],
            keys=algorithm.input_keys,
            allowed_missing=algorithm.default_keys,
        )
    )

    algorithm.validate_inputs(ds)

    ds = algorithm.update_dataset(ds)

    return ds


def test_cc_fit_spatial_lengyel_against_kallenbach():
    def promote_to_coordinate(array, units, dims):
        return xr.DataArray(array * units, coords={f"dim_{dims}": array})

    inputs = dict(
        heat_flux_perp_to_target=promote_to_coordinate(
            np.logspace(np.log10(0.1), np.log10(10.0), num=10), ureg.MW / ureg.m**2, dims="heat_flux_perp_to_target"
        ),
        target_electron_temp=promote_to_coordinate(np.logspace(np.log10(2.0), np.log10(50), num=10), ureg.eV, dims="target_electron_temp"),
        impurity_fraction=promote_to_coordinate([1.0, 2.0, 4.0], ureg.percent, dims="impurity_fraction"),
    )

    lengyel_ds = build_dataset(**inputs, run_lengyel_model=True)
    rcc_reference_ds = build_dataset(**inputs, run_lengyel_model=False)

    # Allow up to 5% error
    assert np.abs(1.0 - lengyel_ds["separatrix_electron_density"] / rcc_reference_ds["separatrix_electron_density"]).mean() < 5e-2
    assert np.abs(1.0 - lengyel_ds["separatrix_electron_temp"] / rcc_reference_ds["separatrix_electron_temp"]).mean() < 5e-2
    assert np.abs(1.0 - lengyel_ds["q_parallel"] / rcc_reference_ds["q_parallel"]).mean() < 5e-2
