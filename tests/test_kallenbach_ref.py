"""Make sure that the Kallenbach translation gives the same result as the paper."""

import pytest
import cfspopcon
import xarray as xr
from cfspopcon.unit_handling import ureg
import numpy as np
from scipy.interpolate import interp1d
from cfspopcon.unit_handling import magnitude as mag

from extended_lengyel.kallenbach_model.reference_data import read_kallenbach_figure_4_reference
from extended_lengyel import read_config, directories


@pytest.fixture(scope="session")
def dataset():
    algorithm = cfspopcon.Algorithm.get_algorithm("kallenbach_idl_translation")

    ds = xr.Dataset(
        data_vars=read_config(
            filepath=directories.notebook_dir / "config.yml",
            elements=["base", "machine_geometry", "target_constraints", "fast_neutrals", "field_at_omp"],
            keys=algorithm.input_keys,
            allowed_missing=algorithm.default_keys,
        )
    )

    assert algorithm.validate_inputs(ds)

    ds = algorithm.update_dataset(ds)
    ds["dim_s_parallel"] = ds["s_parallel"].pint.to(ureg.m).pint.magnitude

    return ds


@pytest.fixture(scope="session")
def reference_data():
    return read_kallenbach_figure_4_reference()


def assert_close(ref, test, rtol=0.25):
    interpolator = interp1d(mag(test["dim_s_parallel"]), mag(test), bounds_error=False)

    if not np.nanmax(np.abs((interpolator(ref["x"]) - ref["y"]) / ref["y"])) < rtol:
        max_difference = np.nanmax(np.abs((interpolator(ref["x"]) - ref["y"]) / ref["y"]))
        raise AssertionError(f"Maximum difference {max_difference} larger than rtol {rtol}")


def test_kallenbach_ref_charge_exchange_power_loss_left(dataset, reference_data):
    ref = reference_data["charge_exchange_power_loss_left"]
    test = dataset["charge_exchange_power_loss"].pint.to(ureg.MW / ureg.m**3)
    assert_close(ref, test)


def test_kallenbach_ref_ionization_power_loss_left(dataset, reference_data):
    ref = reference_data["ionization_power_loss_left"]
    test = dataset["ionization_power_loss"].pint.to(ureg.MW / ureg.m**3)
    assert_close(ref, test)


# def test_kallenbach_ref_hydrogen_radiation_left(dataset, reference_data):
#     # Seems like we have a different definition to the paper for this quantity.
# ref  = reference_data["hydrogen_radiation_left"]
# test = dataset["hydrogen_radiated_power"].pint.to(ureg.MW/ureg.m**3)
#     assert_close(ref, test)


def test_kallenbach_ref_nitrogen_radiation_left(dataset, reference_data):
    ref = reference_data["nitrogen_radiation_left"]
    test = dataset["impurity_radiated_power"].pint.to(ureg.MW / ureg.m**3)
    assert_close(ref, test)


def test_kallenbach_ref_nitrogen_radiation_right(dataset, reference_data):
    ref = reference_data["nitrogen_radiation_right"]
    test = dataset["impurity_radiated_power"].pint.to(ureg.MW / ureg.m**3)
    assert_close(ref, test)


def test_kallenbach_ref_static_pressure_left(dataset, reference_data):
    ref = reference_data["static_pressure_left"]
    test = dataset["static_pressure"].pint.to(ureg.Pa)
    assert_close(ref, test)


def test_kallenbach_ref_total_pressure_left(dataset, reference_data):
    ref = reference_data["total_pressure_left"]
    test = (dataset["static_pressure"] + dataset["dynamic_pressure"]).pint.to(ureg.Pa)
    assert_close(ref, test)


def test_kallenbach_ref_total_pressure_right(dataset, reference_data):
    ref = reference_data["total_pressure_right"]
    test = (dataset["static_pressure"] + dataset["dynamic_pressure"]).pint.to(ureg.Pa)
    assert_close(ref, test)


def test_kallenbach_ref_electron_temp_left(dataset, reference_data):
    ref = reference_data["electron_temp_left"]
    test = dataset["electron_temp"].pint.to(ureg.eV)
    assert_close(ref, test)


def test_kallenbach_ref_electron_temp_right(dataset, reference_data):
    ref = reference_data["electron_temp_right"]
    test = dataset["electron_temp"].pint.to(ureg.eV)
    assert_close(ref, test)


def test_kallenbach_ref_electron_density_left(dataset, reference_data):
    ref = reference_data["electron_density_left"]
    test = dataset["electron_density"].pint.to(ureg.n20)
    assert_close(ref, test)


def test_kallenbach_ref_neutral_density_left(dataset, reference_data):
    ref = reference_data["neutral_density_left"]
    test = dataset["neutral_density"].pint.to(ureg.n20)
    assert_close(ref, test)


def test_kallenbach_ref_electron_density_right(dataset, reference_data):
    ref = reference_data["electron_density_right"]
    test = dataset["electron_density"].pint.to(ureg.n20)
    assert_close(ref, test)


def test_kallenbach_ref_mach_number_left(dataset, reference_data):
    ref = reference_data["mach_number_left"]
    test = np.abs(dataset["mach_number"])
    assert_close(ref, test)
