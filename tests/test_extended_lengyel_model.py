import numpy as np
import pytest
from cfspopcon.formulas.atomic_data import read_atomic_data
from cfspopcon.formulas.impurities.edge_radiator_conc import build_L_int_integrator as _build_L_int_integrator
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import ureg

from extended_lengyel.directories import radas_dir
from extended_lengyel.extended_lengyel_model.Lengyel_model_core import (
    build_L_int_integrator,
    build_mean_charge_interpolator,
    _build_mean_charge_interpolator,
)

def test_mixed_seeding_L_int():
    atomic_data = read_atomic_data(radas_dir)
    reference_ne_tau = 0.5 * ureg.ms * ureg.n20
    reference_electron_density = 1.0 * ureg.n20

    start_temp = 2.0 * ureg.eV
    stop_temp = 20.0 * ureg.eV

    species_list = [AtomicSpecies.Nitrogen, AtomicSpecies.Neon, AtomicSpecies.Argon]

    single_L_int = dict()

    for species in species_list:
        single_L_int[species] = _build_L_int_integrator(
            atomic_data=atomic_data,
            impurity_species=species,
            reference_electron_density=reference_electron_density,
            reference_ne_tau=reference_ne_tau,
        )

    for i, species in enumerate(species_list):
        weights = [0.0, 0.0, 0.0]
        weights[i] = 1.0

        mixed_L_int = build_L_int_integrator(
            impurity_species_list=species_list,
            impurity_weights_list=weights,
            atomic_data=atomic_data,
            reference_ne_tau=reference_ne_tau,
            reference_electron_density=reference_electron_density,
        )

        assert np.isclose(single_L_int[species](start_temp, stop_temp), mixed_L_int(start_temp, stop_temp), atol=0.0)

    weights = [0.1, 0.2, 0.3]

    mixed_L_int = build_L_int_integrator(
        impurity_species_list=species_list,
        impurity_weights_list=weights,
        atomic_data=atomic_data,
        reference_ne_tau=reference_ne_tau,
        reference_electron_density=reference_electron_density,
    )

    assert np.isclose(
        sum(single_L_int[species](start_temp, stop_temp) * weights[i] for i, species in enumerate(species_list)),
        mixed_L_int(start_temp, stop_temp),
        atol=0.0,
    )

    with pytest.raises(AssertionError):
        assert np.isclose(single_L_int[AtomicSpecies.Nitrogen](start_temp, stop_temp), mixed_L_int(start_temp, stop_temp), atol=0.0)


def test_mixed_seeding_mean_charge():
    atomic_data = read_atomic_data(radas_dir)
    reference_ne_tau = 0.5 * ureg.ms * ureg.n20
    reference_electron_density = 1.0 * ureg.n20

    temp = 20.0 * ureg.eV

    species_list = [AtomicSpecies.Nitrogen, AtomicSpecies.Neon, AtomicSpecies.Argon]

    single_mean_charge = dict()

    for species in species_list:
        single_mean_charge[species] = _build_mean_charge_interpolator(
            atomic_data=atomic_data,
            impurity_species=species,
            reference_electron_density=reference_electron_density,
            reference_ne_tau=reference_ne_tau,
        )

    for i, species in enumerate(species_list):
        weights = [0.0, 0.0, 0.0]
        weights[i] = 1.0

        mixed_mean_charge = build_mean_charge_interpolator(
            impurity_species_list=species_list,
            impurity_weights_list=weights,
            atomic_data=atomic_data,
            reference_ne_tau=reference_ne_tau,
            reference_electron_density=reference_electron_density,
        )

        assert np.isclose(single_mean_charge[species](temp), mixed_mean_charge(temp), atol=0.0)

    weights = [0.1, 0.2, 0.3]

    mixed_mean_charge = build_mean_charge_interpolator(
        impurity_species_list=species_list,
        impurity_weights_list=weights,
        atomic_data=atomic_data,
        reference_ne_tau=reference_ne_tau,
        reference_electron_density=reference_electron_density,
    )

    assert np.isclose(
        sum(single_mean_charge[species](temp) * weights[i] for i, species in enumerate(species_list)) / sum(weights),
        mixed_mean_charge(temp),
        atol=0.0,
    )

    with pytest.raises(AssertionError):
        assert np.isclose(single_mean_charge[AtomicSpecies.Nitrogen](temp), mixed_mean_charge(temp), atol=0.0)
