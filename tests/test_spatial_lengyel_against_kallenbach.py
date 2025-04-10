import cfspopcon
import numpy as np
import xarray as xr
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import ureg

import extended_lengyel
import extended_lengyel.directories


def build_dataset(
    run_lengyel_model_in_conductive_layer=True,
    **overrides,
):
    if run_lengyel_model_in_conductive_layer:
        algorithm = cfspopcon.CompositeAlgorithm.from_list(
            ["kallenbach_model_to_cc", "run_spatial_lengyel_model", "postprocess_spatial_lengyel_model"]
        )
    else:
        algorithm = cfspopcon.Algorithm.get_algorithm("kallenbach_model")

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


def test_spatial_lengyel_against_kallenbach():
    def promote_to_coordinate(array, units, dims):
        return xr.DataArray(array * units, coords={f"dim_{dims}": array})

    inputs = dict(
        heat_flux_perp_to_target=promote_to_coordinate(
            np.logspace(np.log10(0.1), np.log10(10.0), num=10), ureg.MW / ureg.m**2, dims="heat_flux_perp_to_target"
        ),
        target_electron_temp=promote_to_coordinate(np.logspace(np.log10(2.0), np.log10(50), num=10), ureg.eV, dims="target_electron_temp"),
        impurity_fraction=promote_to_coordinate([1.0, 2.0, 4.0], ureg.percent, dims="impurity_fraction"),
    )

    rcc_plus_lengyel_ds = build_dataset(**inputs, run_lengyel_model_in_conductive_layer=True)
    rcc_reference_ds = build_dataset(**inputs, run_lengyel_model_in_conductive_layer=False)

    xr.testing.assert_allclose(
        rcc_plus_lengyel_ds["separatrix_electron_density"], rcc_reference_ds["separatrix_electron_density"], rtol=1e-2
    )
    xr.testing.assert_allclose(rcc_plus_lengyel_ds["separatrix_electron_temp"], rcc_reference_ds["separatrix_electron_temp"], rtol=1e-2)
    xr.testing.assert_allclose(rcc_plus_lengyel_ds["q_parallel"], rcc_reference_ds["q_parallel"], rtol=1e-2)
