from cfspopcon.unit_handling import magnitude_in_units, ureg
import xarray as xr
import numpy as np
from pathlib import Path

from extended_lengyel.extended_lengyel_model.inverse_model import run_inverse_extended_lengyel_model
from extended_lengyel.extended_lengyel_model.forward_model import run_forward_extended_lengyel_model

from cfspopcon.named_options import AtomicSpecies
from extended_lengyel.config import setup_impurities
from extended_lengyel.extended_lengyel_model.Lengyel_model_core import (
    CzLINT_integrator,
    Mean_charge_interpolator
)
from extended_lengyel.mavrin_data import MavrinData
from cfspopcon.formulas.atomic_data import read_atomic_data

np.seterr(over="raise",under="raise")

def test_iterative_models():

    for i, atomic_data in enumerate([MavrinData(), read_atomic_data(Path(__file__).parents[1] / "radas_dir")[0]]):

        impurity_kwargs = dict(
            ne_tau = 0.5 * ureg.ms * ureg.n20,
            electron_density = 1.0 * ureg.n20,
            rtol_nearest = 1e-6,
        )

        seed_impurity_species, seed_impurity_weights = \
            setup_impurities([AtomicSpecies.Nitrogen, AtomicSpecies.Argon], [1.0, 0.05])

        CzLINT_for_seed_impurities = CzLINT_integrator(
            seed_impurity_species, seed_impurity_weights, atomic_data,
            **impurity_kwargs
        )
        mean_charge_for_seed_impurities = Mean_charge_interpolator(
            seed_impurity_species, atomic_data,
            **impurity_kwargs
        )

        fixed_impurity_species, fixed_impurity_weights = \
            setup_impurities([AtomicSpecies.Helium], [1.0e-2])

        CzLINT_for_fixed_impurities = CzLINT_integrator(
            fixed_impurity_species, fixed_impurity_weights, atomic_data,
            **impurity_kwargs
        )
        mean_charge_for_fixed_impurities = Mean_charge_interpolator(
            fixed_impurity_species, atomic_data,
            **impurity_kwargs
        )

        kwargs = dict(
            power_crossing_separatrix = 5.5 * ureg.MW,
            separatrix_electron_density = 3.3 * ureg.n19,
            divertor_broadening_factor = 3.0,
            CzLINT_for_seed_impurities = CzLINT_for_seed_impurities,
            mean_charge_for_seed_impurities = mean_charge_for_seed_impurities,
            CzLINT_for_fixed_impurities = CzLINT_for_fixed_impurities,
            mean_charge_for_fixed_impurities = mean_charge_for_fixed_impurities,
            magnetic_field_on_axis = 2.5 * ureg.T,
            plasma_current = 1.0 * ureg.MA,
            parallel_connection_length = 20.0 * ureg.m,
            divertor_parallel_length = 5.0 * ureg.m,
            major_radius = 1.65 * ureg.m,
            minor_radius = 0.5 * ureg.m,
            elongation_psi95 = 1.6,
            triangularity_psi95 = 0.3,
            target_angle_of_incidence = 3.0,
            fraction_of_P_SOL_to_divertor = 2./3.,
            sheath_heat_transmission_factor = 8.,
            iterations = 100,
        )

        target_electron_temp_in = xr.DataArray([2.34], dims="dim_test") * ureg.eV

        (
            c_z_out,
            INV_parallel_ion_flux_to_target,
            INV_neutral_pressure_in_divertor,
            INV_alpha_t,
            INV_q_parallel,
            INV_heat_flux_perp_to_target,
            INV_separatrix_z_effective,
            INV_converged,
        ) = run_inverse_extended_lengyel_model(
            target_electron_temp = target_electron_temp_in,
            **kwargs
        )
        assert INV_converged

        (
            target_electron_temp_out,
            FWD_parallel_ion_flux_to_target,
            FWD_neutral_pressure_in_divertor,
            FWD_alpha_t,
            FWD_q_parallel,
            FWD_heat_flux_perp_to_target,
            FWD_separatrix_z_effective,
            FWD_converged,
        ) = run_forward_extended_lengyel_model(
            c_z = c_z_out,
            **kwargs
        )
        assert FWD_converged

        assert np.isclose(magnitude_in_units(target_electron_temp_in, ureg.eV), magnitude_in_units(target_electron_temp_out, ureg.eV))
        assert np.isclose(magnitude_in_units(INV_parallel_ion_flux_to_target, ureg.m**-2/ureg.s), magnitude_in_units(FWD_parallel_ion_flux_to_target, ureg.m**-2/ureg.s))
        assert np.isclose(magnitude_in_units(INV_neutral_pressure_in_divertor, ureg.Pa), magnitude_in_units(FWD_neutral_pressure_in_divertor, ureg.Pa))
        assert np.isclose(magnitude_in_units(INV_alpha_t, ureg.dimensionless), magnitude_in_units(FWD_alpha_t, ureg.dimensionless))
        assert np.isclose(magnitude_in_units(INV_q_parallel, ureg.GW/ureg.m**2), magnitude_in_units(FWD_q_parallel, ureg.GW/ureg.m**2))
        assert np.isclose(magnitude_in_units(INV_heat_flux_perp_to_target, ureg.MW/ureg.m**2), magnitude_in_units(FWD_heat_flux_perp_to_target, ureg.MW/ureg.m**2))
        assert np.isclose(magnitude_in_units(INV_separatrix_z_effective, ureg.dimensionless), magnitude_in_units(FWD_separatrix_z_effective, ureg.dimensionless))

        if i == 0:
            assert np.isclose(magnitude_in_units(target_electron_temp_in, ureg.eV), 2.34, atol=0.0, rtol=1e-2)
            assert np.isclose(magnitude_in_units(c_z_out, ureg.dimensionless), 0.038183522427857504, atol=0.0, rtol=1e-2)
            assert np.isclose(magnitude_in_units(INV_parallel_ion_flux_to_target, ureg.m**-2/ureg.s), 5.047499599460096e+24, atol=0.0, rtol=1e-2)
            assert np.isclose(magnitude_in_units(INV_neutral_pressure_in_divertor, ureg.Pa), 1.736573655976632, atol=0.0, rtol=1e-2)
            assert np.isclose(magnitude_in_units(INV_alpha_t, ureg.dimensionless), 0.36047105992270795, atol=0.0, rtol=1e-2)
            assert np.isclose(magnitude_in_units(INV_heat_flux_perp_to_target, ureg.W/ureg.m**2), 792305.5442545213, atol=0.0, rtol=1e-2)

