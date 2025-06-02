import numpy as np
import xarray as xr
import pytest
from cfspopcon.formulas.atomic_data import read_atomic_data
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import ureg

from extended_lengyel.directories import radas_dir
from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator, calc_z_effective
from extended_lengyel.xr_helpers import item
from cfspopcon.unit_handling import magnitude_in_units, ureg, get_units

def compare_magnitude(a, b):
    return magnitude_in_units(a, get_units(b)), magnitude_in_units(b, get_units(b))

def test_mixed_seeding_L_int():
    atomic_data = read_atomic_data(radas_dir)
    ne_tau = 0.5 * ureg.ms * ureg.n20
    electron_density = 1.0 * ureg.n20

    start_temp = 2.0 * ureg.eV
    stop_temp = 20.0 * ureg.eV

    species_list = [AtomicSpecies.Nitrogen, AtomicSpecies.Neon, AtomicSpecies.Argon]

    single_L_int = dict()

    for species in species_list:
        single_L_int[species] = CzLINT_integrator.build_L_int_integrator(
            species_atomic_data=item(atomic_data).get_dataset(item(species)),
            electron_density=electron_density,
            ne_tau=ne_tau,
        )

    for i, species in enumerate(species_list):
        weights = [0.0, 0.0, 0.0]
        weights[i] = 1.0

        mixed_L_int = CzLINT_integrator(
            impurity_species_list=species_list,
            impurity_weights_list=weights,
            atomic_data=atomic_data,
            ne_tau=ne_tau,
            electron_density=electron_density,
        )

        assert np.isclose(*compare_magnitude(single_L_int[species](start_temp, stop_temp), mixed_L_int(start_temp, stop_temp)), atol=0.0)

    weights = [0.1, 0.2, 0.3]

    mixed_L_int = CzLINT_integrator(
        impurity_species_list=species_list,
        impurity_weights_list=weights,
        atomic_data=atomic_data,
        ne_tau=ne_tau,
        electron_density=electron_density,
    )

    assert np.isclose(*compare_magnitude(sum(single_L_int[species](start_temp, stop_temp) * weights[i] for i, species in enumerate(species_list)), mixed_L_int(start_temp, stop_temp)), atol=0.0)

    with pytest.raises(AssertionError):
        assert np.isclose(*compare_magnitude(single_L_int[AtomicSpecies.Nitrogen](start_temp, stop_temp), mixed_L_int(start_temp, stop_temp)), atol=0.0)


def test_mixed_seeding_mean_charge():
    atomic_data = read_atomic_data(radas_dir)
    ne_tau = 0.5 * ureg.ms * ureg.n20
    electron_density = 1.0 * ureg.n20

    temp = 20.0 * ureg.eV

    species_list = [AtomicSpecies.Nitrogen, AtomicSpecies.Neon, AtomicSpecies.Argon]

    single_mean_charge = dict()

    for species in species_list:
        single_mean_charge[species] = Mean_charge_interpolator.build_mean_charge_interpolator(
            species_atomic_data=atomic_data.get_dataset(species),
            electron_density=electron_density,
            ne_tau=ne_tau,
        )
    
    mixed_mean_charge = Mean_charge_interpolator(
        impurity_species_list=species_list,
        atomic_data=atomic_data,
        ne_tau=ne_tau,
        electron_density=electron_density,
    )

    for species in species_list:
        assert np.isclose(*compare_magnitude(single_mean_charge[species](temp), mixed_mean_charge(temp).sel(dim_species=species)), atol=0.0)

    with pytest.raises(AssertionError):
        assert np.allclose(*compare_magnitude(single_mean_charge[AtomicSpecies.Nitrogen](temp), mixed_mean_charge(temp)), atol=0.0)

def test_calc_z_effective():
    atomic_data = read_atomic_data(radas_dir)


    z_eff_no_impurities = calc_z_effective(
        electron_temp=100.0 * ureg.eV,
        c_z = 1e-2,
        mean_charge_for_seed_impurities = Mean_charge_interpolator.empty(),
        mean_charge_for_fixed_impurities = Mean_charge_interpolator.empty(),
        CzLINT_for_seed_impurities = CzLINT_integrator.empty(),
        CzLINT_for_fixed_impurities = CzLINT_integrator.empty(),
    )

    assert np.isclose(z_eff_no_impurities, 1.0)

    z_eff_no_impurities = calc_z_effective(
        electron_temp=xr.DataArray(np.array([100.0, 120.0]) * ureg.eV, dims="dim_Te"),
        c_z = xr.DataArray(np.array([1.0, 2.0]) * ureg.percent, dims="dim_c_z"),
        mean_charge_for_seed_impurities = Mean_charge_interpolator.empty(),
        mean_charge_for_fixed_impurities = Mean_charge_interpolator.empty(),
        CzLINT_for_seed_impurities = CzLINT_integrator.empty(),
        CzLINT_for_fixed_impurities = CzLINT_integrator.empty(),
    )

    assert np.allclose(magnitude_in_units(z_eff_no_impurities, ureg.dimensionless), 1.0)

    z_eff_deuterium = calc_z_effective(
        electron_temp=xr.DataArray(np.array([100.0, 120.0]) * ureg.eV, dims="dim_Te"),
        c_z = xr.DataArray(np.array([1.0, 2.0]) * ureg.percent, dims="dim_c_z"),
        mean_charge_for_seed_impurities = Mean_charge_interpolator.empty(),
        mean_charge_for_fixed_impurities = Mean_charge_interpolator(["Deuterium"], atomic_data),
        CzLINT_for_seed_impurities = CzLINT_integrator.empty(),
        CzLINT_for_fixed_impurities = CzLINT_integrator(["Deuterium"], [1.0], atomic_data),
    )

    assert np.allclose(magnitude_in_units(z_eff_deuterium, ureg.dimensionless), 1.0, rtol=1e-3)
    
    coronal_ne_tau = 1.0e20 * ureg.m**-3 * ureg.s

    mean_charge_N = Mean_charge_interpolator(["Nitrogen"],
                                             atomic_data,
                                             ne_tau=coronal_ne_tau,
                                             rtol_nearest=1
    )
    
    assert np.isclose(magnitude_in_units(mean_charge_N(10.0 * ureg.keV), ureg.dimensionless), 7, rtol=1e-3)

    z_eff_seeding = calc_z_effective(
        electron_temp=xr.DataArray(np.array([10.0, 12.0]) * ureg.keV, dims="dim_Te"),
        c_z = xr.DataArray(np.array([1.0]) * ureg.percent, dims="dim_c_z"),
        mean_charge_for_seed_impurities = mean_charge_N,
        mean_charge_for_fixed_impurities = Mean_charge_interpolator(["Deuterium"], atomic_data),
        CzLINT_for_seed_impurities = CzLINT_integrator(["Nitrogen"], [1.0], atomic_data, ne_tau=coronal_ne_tau, rtol_nearest=1),
        CzLINT_for_fixed_impurities = CzLINT_integrator(["Deuterium"], [1.0], atomic_data),
    )

    z_eff_expected = 1 + 1e-2 * 7 * (7 - 1)

    assert np.allclose(magnitude_in_units(z_eff_seeding, ureg.dimensionless), z_eff_expected, rtol=1e-2)