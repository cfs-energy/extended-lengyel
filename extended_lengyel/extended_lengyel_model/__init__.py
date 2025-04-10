"""Extended Lengyel model."""

from .convective_loss_fits import (
    calc_electron_density_from_cc_fit,
    calc_electron_temp_from_cc_fit,
    calc_parallel_heat_flux_from_conv_loss,
    calc_power_loss_from_cc_fit,
    calc_momentum_loss_from_cc_fit,
    ignore_s_parallel_width_for_cc_interface,
    temperature_fit_function,
)
from .Lengyel_model_core import (
    build_mixed_seeding_L_int_integrator,
    build_mixed_seeding_mean_charge_interpolator,
)
from .power_loss import (
    calc_parallel_heat_flux_at_target_from_power_loss_fraction,
    calc_required_power_loss_fraction,
)
from .upstream_temp import (
    calc_separatrix_electron_temp_with_broadening,
    calc_separatrix_total_pressure_LG,
)

from .Lengyel_model_basic import run_basic_lengyel_model
from .Lengyel_model_extended_S import run_extended_lengyel_model_with_S_correction
from .Lengyel_model_extended_S_Zeff import run_extended_lengyel_model_with_S_and_Zeff_correction
from .Lengyel_model_extended_S_Zeff_alphat import run_extended_lengyel_model_with_S_Zeff_and_alphat_correction

from enum import Enum


class LengyelModel(Enum):
    """Enum of possible Lengyel models."""

    basic = 1
    S_correction = 2
    S_and_Zeff_correction = 3
    S_Zeff_and_fconv_correction = 4
    S_Zeff_fconv_and_alpha_t_correction = 5


__all__ = [
    "temperature_fit_function",
    "calc_momentum_loss_from_cc_fit",
    "calc_power_loss_from_cc_fit",
    "calc_electron_temp_from_cc_fit",
    "calc_electron_density_from_cc_fit",
    "calc_parallel_heat_flux_from_conv_loss",
    "ignore_s_parallel_width_for_cc_interface",
    "calc_required_power_loss_fraction",
    "calc_parallel_heat_flux_at_target_from_power_loss_fraction",
    "calc_separatrix_electron_temp_with_broadening",
    "calc_separatrix_total_pressure_LG",
    "build_mixed_seeding_L_int_integrator",
    "build_mixed_seeding_mean_charge_interpolator",
    "run_basic_lengyel_model",
    "run_extended_lengyel_model_with_S_correction",
    "run_extended_lengyel_model_with_S_and_Zeff_correction",
    "run_extended_lengyel_model_with_S_Zeff_and_alphat_correction",
]
