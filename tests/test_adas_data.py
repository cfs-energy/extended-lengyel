import pytest
import cfspopcon
from cfspopcon.unit_handling import ureg, magnitude_in_units
from extended_lengyel.adas_data import AtomicSpeciesAdasData
from extended_lengyel.directories import library_directory
import xarray as xr
import numpy as np


@pytest.fixture()
def reference_ne_tau():
    return 0.5 * ureg.ms * ureg.n20


def test_build_using_string(reference_ne_tau):
    AtomicSpeciesAdasData(species_name="Deuterium", reference_ne_tau=reference_ne_tau)


def test_build_using_lowercase_string(reference_ne_tau):
    AtomicSpeciesAdasData(species_name="deuterium", reference_ne_tau=reference_ne_tau)


def test_build_using_atomic_species(reference_ne_tau):
    AtomicSpeciesAdasData(species_name=cfspopcon.named_options.AtomicSpecies.Deuterium, reference_ne_tau=reference_ne_tau)


@pytest.fixture()
def deuterium_adas(reference_ne_tau):
    return AtomicSpeciesAdasData(species_name="Deuterium", reference_ne_tau=reference_ne_tau)


@pytest.fixture()
def raw_ds():
    filepath = library_directory / "radas_dir" / "output" / "deuterium.nc"

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist. Make sure that you have run radas to build the atomic data files.")
    else:
        return xr.load_dataset(filepath).pint.quantify()


def test_interpolation_with_unit_handling(deuterium_adas, raw_ds):
    ref = raw_ds["effective_ionisation"].sel(dim_charge_state=0).isel(dim_electron_density=10, dim_electron_temp=20)

    electron_density = ref.dim_electron_density * raw_ds.reference_electron_density
    electron_temp = ref.dim_electron_temp * raw_ds.reference_electron_temp

    result = deuterium_adas.ionization_rate.eval(electron_density, electron_temp)

    assert np.isclose(result, ref.item())


def test_unitless_interpolation(deuterium_adas, raw_ds):
    ref = raw_ds["effective_ionisation"].sel(dim_charge_state=0).isel(dim_electron_density=10, dim_electron_temp=20)

    electron_density = ref.dim_electron_density * raw_ds.reference_electron_density
    electron_temp = ref.dim_electron_temp * raw_ds.reference_electron_temp

    result = deuterium_adas.ionization_rate.unitless_eval(
        magnitude_in_units(electron_density, ureg.m**-3), magnitude_in_units(electron_temp, ureg.eV)
    )

    assert np.isclose(result, ref.item().magnitude)
