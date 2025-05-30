import numpy as np
import pytest
from cfspopcon.formulas.atomic_data import read_atomic_data
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import ureg

from extended_lengyel.directories import radas_dir
from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator, item
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
