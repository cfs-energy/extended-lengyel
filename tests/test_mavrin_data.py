import pytest
from cfspopcon.formulas.atomic_data.atomic_data import read_atomic_data, AtomicData
from extended_lengyel.directories import radas_dir
from extended_lengyel.mavrin_data import read_mavrin_data, MavrinData, SpeciesMavrinData
from cfspopcon.named_options import AtomicSpecies
from extended_lengyel.extended_lengyel_model.Lengyel_model_core import (
    build_CzLINT_for_seed_impurities,
    build_mean_charge_for_seed_impurities,
    set_single_impurity_species
)
import numpy as np
import xarray as xr
from cfspopcon.unit_handling import magnitude_in_units, ureg

@pytest.fixture()
def radas_data() -> AtomicData:
    atomic_data, _ = read_atomic_data(radas_dir=radas_dir)
    return atomic_data

@pytest.fixture()
def mavrin_data() -> MavrinData:
    return read_mavrin_data()

@pytest.fixture()
def impurity_species() -> AtomicSpecies:
    return AtomicSpecies["Neon"]

@pytest.fixture()
def seed_impurity_species(impurity_species):
    seed_impurity_species, _ = set_single_impurity_species(impurity_species)
    return seed_impurity_species

@pytest.fixture()
def seed_impurity_weights(impurity_species):
    _, seed_impurity_weights = set_single_impurity_species(impurity_species)
    return seed_impurity_weights

def test_CzLINT_match(radas_data, mavrin_data, seed_impurity_species, seed_impurity_weights):

    CzLINT_radas = build_CzLINT_for_seed_impurities(seed_impurity_species, seed_impurity_weights, radas_data)
    CzLINT_mavrin = build_CzLINT_for_seed_impurities(seed_impurity_species, seed_impurity_weights, mavrin_data)

    start_temp = 10.0 * ureg.eV
    stop_temp = 20.0 * ureg.eV

    assert np.isclose(
        magnitude_in_units(CzLINT_radas(start_temp, stop_temp), ureg.eV**1.5 * ureg.m**3 * ureg.W),
        magnitude_in_units(CzLINT_mavrin(start_temp, stop_temp), ureg.eV**1.5 * ureg.m**3 * ureg.W),
        atol = 0.0, # very small values, so use zero absolute tolerance
        rtol = 1e-1 # only need an approximate match. Data is different, so we expect some differences.
    )

def test_mean_z_match(radas_data, mavrin_data, seed_impurity_species):

    mean_z_radas = build_mean_charge_for_seed_impurities(seed_impurity_species, radas_data)
    mean_z_mavrin = build_mean_charge_for_seed_impurities(seed_impurity_species, mavrin_data)

    electron_temp = 10.0 * ureg.eV

    assert np.isclose(
        magnitude_in_units(mean_z_radas(electron_temp), ureg.dimensionless),
        magnitude_in_units(mean_z_mavrin(electron_temp), ureg.dimensionless),
        atol = 0.0,
        rtol = 1e-1
    )
