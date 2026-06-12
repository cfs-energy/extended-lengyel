"""Unified iterative solver for the four extended Lengyel model variants.

The four public models (forward/inverse, with/without a divertor neutral
pressure constraint) share a single iteration loop, parameterized by:

- ``mode="forward"``: the impurity fraction is given and the target electron
  temperature is found by relaxed fixed-point iteration.
- ``mode="inverse"``: the target electron temperature is given and the required
  impurity fraction is solved for directly each iteration (no relaxation).
- ``pdiv=True``: the separatrix electron density is unknown and is iterated so
  that the computed divertor neutral pressure matches the required one.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator

from .constants import CONVERGENCE, MA_to_A, MW_to_W, amu_to_kg, n20_to_m3
from .physics import (
    calc_alpha_t,
    calc_cc_interface_temp,
    calc_convection_layer_losses,
    calc_f_other,
    calc_geometry,
    calc_kappa_e,
    calc_lengyel_integrals,
    calc_q_parallel,
    calc_separatrix_total_pressure,
    calc_target_electron_temp_basic,
    calc_target_postprocessing,
    calc_temperature_ladder,
    calc_z_effective,
    relax,
)


@dataclass(frozen=True)
class ModelInputs:
    """Model inputs as unitless magnitudes (SI except temperatures in eV, densities in n20, ion mass in amu)."""

    power_crossing_separatrix: float  # W
    divertor_broadening_factor: float
    CzLINT_for_seed_impurities: CzLINT_integrator
    mean_charge_for_seed_impurities: Mean_charge_interpolator
    magnetic_field_on_axis: float  # T
    plasma_current: float  # A
    parallel_connection_length: float  # m
    divertor_parallel_length: float  # m
    major_radius: float  # m
    minor_radius: float  # m
    elongation_psi95: float
    triangularity_psi95: float
    target_angle_of_incidence: float  # rad
    fraction_of_P_SOL_to_divertor: float
    CzLINT_for_fixed_impurities: CzLINT_integrator
    mean_charge_for_fixed_impurities: Mean_charge_interpolator
    average_ion_mass: float  # amu
    sheath_heat_transmission_factor: float
    ratio_of_upstream_to_average_poloidal_field: float
    wall_temperature: float  # K
    SOL_conduction_fraction: float
    ratio_of_molecular_to_ion_mass: float
    separatrix_mach_number: float
    separatrix_ratio_of_ion_to_electron_temp: float
    separatrix_ratio_of_electron_to_ion_density: float
    target_ratio_of_ion_to_electron_temp: float
    target_ratio_of_electron_to_ion_density: float
    target_mach_number: float
    toroidal_flux_expansion: float
    iterations: int
    # Mode-specific inputs: forward models set impurity_fraction, inverse models set
    # target_electron_temp; pdiv variants set required_neutral_pressure_in_divertor
    # instead of separatrix_electron_density.
    impurity_fraction: float | None = None
    target_electron_temp: float | None = None  # eV
    separatrix_electron_density: float | None = None  # n20
    required_neutral_pressure_in_divertor: float | None = None  # Pa

    @classmethod
    def from_wrapper_args(cls, neutral_pressure_in_divertor: float | None = None, **kwargs: Any) -> "ModelInputs":
        """Build ModelInputs from the public wrapper arguments (magnitudes in the declared input units)."""
        if kwargs.get("CzLINT_for_fixed_impurities") is None:
            kwargs["CzLINT_for_fixed_impurities"] = CzLINT_integrator.empty()
        if kwargs.get("mean_charge_for_fixed_impurities") is None:
            kwargs["mean_charge_for_fixed_impurities"] = Mean_charge_interpolator.empty()

        # Convert all inputs to SI units, except for electron-volts
        kwargs["plasma_current"] = kwargs["plasma_current"] * MA_to_A
        kwargs["target_angle_of_incidence"] = np.deg2rad(kwargs["target_angle_of_incidence"])
        kwargs["power_crossing_separatrix"] = kwargs["power_crossing_separatrix"] * MW_to_W
        if kwargs.get("separatrix_electron_density") is not None:
            kwargs["separatrix_electron_density"] = kwargs["separatrix_electron_density"] / n20_to_m3

        return cls(required_neutral_pressure_in_divertor=neutral_pressure_in_divertor, **kwargs)


@dataclass
class SolverResult:
    """Solver outputs as unitless magnitudes (same conventions as ModelInputs)."""

    impurity_fraction: float
    target_electron_temp: float  # eV
    separatrix_electron_density: float  # n20
    separatrix_electron_temp: float  # eV
    alpha_t: float
    q_parallel: float  # W/m^2
    separatrix_z_effective: float
    parallel_ion_flux_to_target: float  # m^-2 s^-1
    neutral_pressure_in_divertor: float  # Pa
    heat_flux_perp_to_target: float  # W/m^2
    converged: bool


def run_extended_lengyel_solver(  # noqa: PLR0912, PLR0915
    p: ModelInputs, mode: Literal["forward", "inverse"], pdiv: bool
) -> SolverResult:
    """Run the unified extended Lengyel iteration loop for the requested model variant."""
    # The original inverse-pdiv model clamps alpha_t and the divertor Z_eff to >= 0 (the
    # direct solve can produce negative impurity fractions); the other variants do not.
    clamp_negative_values = mode == "inverse" and pdiv

    geo = calc_geometry(p)

    # Starting values for the iterative solver
    if mode == "inverse":
        # Target temperature is fixed, so the convection-layer losses are loop-invariant
        losses = calc_convection_layer_losses(p.target_electron_temp)
        target_electron_temp = p.target_electron_temp
        electron_temp_at_cc_interface = calc_cc_interface_temp(target_electron_temp, losses)
        impurity_fraction = np.nan
        prev_impurity_fraction = np.nan
    else:
        impurity_fraction = p.impurity_fraction
        target_electron_temp = 2.0  # eV
        electron_temp_at_cc_interface = 2.5  # eV
        divertor_entrance_electron_temp = 50.0  # eV
        prev_target_electron_temp = np.nan
        prev_divertor_entrance_electron_temp = np.nan
        prev_parallel_heat_flux_at_cc_interface = np.nan

    separatrix_electron_temp = 100.0  # eV
    alpha_t = 0.0
    prev_separatrix_electron_temp = np.nan
    prev_alpha_t = np.nan

    if pdiv:
        separatrix_electron_density = 0.265 * p.required_neutral_pressure_in_divertor**0.31  # in n20 units
    else:
        separatrix_electron_density = p.separatrix_electron_density

    # When pdiv, convergence is blocked for one iteration after any guard below trips
    convergence_allowed = False

    for it in range(p.iterations):
        # Calculate q_parallel consistent with alpha-t and the separatrix electron temperature
        q_parallel = calc_q_parallel(separatrix_electron_temp, alpha_t, geo, p, clamp_alpha_t=clamp_negative_values)

        # Divertor Z_eff for the conductivity correction: the forward models recompute it from the
        # known impurity fraction; the inverse models carry it over from the end of the previous
        # iteration (starting from 1.0), since the impurity fraction is not yet known.
        if mode == "forward":
            divertor_z_effective = calc_z_effective(impurity_fraction, divertor_entrance_electron_temp, p)
        elif it == 0:
            divertor_z_effective = 1.0
        if clamp_negative_values:
            divertor_z_effective = np.maximum(divertor_z_effective, 0.0)
        kappa_e = calc_kappa_e(divertor_z_effective)

        divertor_entrance_electron_temp, separatrix_electron_temp = calc_temperature_ladder(
            electron_temp_at_cc_interface, q_parallel, kappa_e, p
        )

        qu = q_parallel
        b = p.divertor_broadening_factor

        if mode == "forward":
            separatrix_z_effective = calc_z_effective(impurity_fraction, separatrix_electron_temp, p)
            alpha_t = calc_alpha_t(
                separatrix_electron_density=separatrix_electron_density * n20_to_m3,
                separatrix_electron_temp=separatrix_electron_temp,
                cylindrical_safety_factor=geo.cylindrical_safety_factor,
                major_radius=p.major_radius,
                average_ion_mass=p.average_ion_mass * amu_to_kg,
                z_effective=separatrix_z_effective,
                mean_ion_charge_state=1.0,
            )

            # Calculate the power loss due to impurities: all concentrations are known, so the
            # heat flux at the cc-interface follows from the power balance with the seed
            # impurities folded into the fixed-concentration radiation term.
            L = calc_lengyel_integrals(electron_temp_at_cc_interface, divertor_entrance_electron_temp, separatrix_electron_temp, p)
            Lint_cc_div = impurity_fraction * L.Ls_cc_div + L.Lf_cc_div
            Lint_div_u = impurity_fraction * L.Ls_div_u + L.Lf_div_u

            k = 2.0 * kappa_e * separatrix_electron_density**2 * separatrix_electron_temp**2
            qcc_squared = (qu**2 - k * (b**2 * Lint_cc_div + Lint_div_u)) / b**2
            if qcc_squared < 0:
                convergence_allowed = False
                qcc_squared = 0.0
            parallel_heat_flux_at_cc_interface = np.sqrt(qcc_squared)

            separatrix_total_pressure = calc_separatrix_total_pressure(separatrix_electron_density, separatrix_electron_temp, p)
            target_electron_temp_basic = calc_target_electron_temp_basic(q_parallel, separatrix_total_pressure, p)
            f_other = calc_f_other(p)

            # Calculate Te_tar consistent with parallel_heat_flux_at_cc_interface
            losses = calc_convection_layer_losses(target_electron_temp)
            parallel_heat_flux_at_target = (1.0 - losses.power) * parallel_heat_flux_at_cc_interface
            SOL_power_loss_fraction = 1.0 - parallel_heat_flux_at_target / q_parallel
            f_vol_loss = (1.0 - SOL_power_loss_fraction) ** 2 / (1.0 - losses.momentum) ** 2

            target_electron_temp = target_electron_temp_basic * f_vol_loss * f_other
            if pdiv and target_electron_temp <= 0:
                convergence_allowed = False
                target_electron_temp = 0.01

            electron_temp_at_cc_interface = calc_cc_interface_temp(target_electron_temp, losses)

            current = [
                alpha_t,
                target_electron_temp,
                divertor_entrance_electron_temp,
                separatrix_electron_temp,
                parallel_heat_flux_at_cc_interface,
            ]
            previous = [
                prev_alpha_t,
                prev_target_electron_temp,
                prev_divertor_entrance_electron_temp,
                prev_separatrix_electron_temp,
                prev_parallel_heat_flux_at_cc_interface,
            ]
        else:  # inverse
            separatrix_total_pressure = calc_separatrix_total_pressure(separatrix_electron_density, separatrix_electron_temp, p)
            target_electron_temp_basic = calc_target_electron_temp_basic(q_parallel, separatrix_total_pressure, p)
            f_other = calc_f_other(p)

            # Run the two-point-model to calculate the required power loss fraction to achieve
            # a desired target electron temperature
            required_power_loss = 1.0 - np.sqrt(
                target_electron_temp / target_electron_temp_basic * (1.0 - losses.momentum) ** 2 / f_other
            )
            parallel_heat_flux_at_target = q_parallel * (1.0 - required_power_loss)  # W/m^2
            parallel_heat_flux_at_cc_interface = parallel_heat_flux_at_target / (1.0 - losses.power)

            L = calc_lengyel_integrals(electron_temp_at_cc_interface, divertor_entrance_electron_temp, separatrix_electron_temp, p)
            qcc = parallel_heat_flux_at_cc_interface
            k = 2.0 * kappa_e * separatrix_electron_density**2 * separatrix_electron_temp**2

            # Reformulated Lengyel solve (combining equations 40 and 43 of Body, Kallenbach and
            # Eich, NF 2025): the seed-impurity concentration multiplier follows directly from
            # the power balance, without the intermediate divertor-entrance heat flux q_div.
            Lambda_seed = b**2 * L.Ls_cc_div + L.Ls_div_u
            Lambda_fixed = b**2 * L.Lf_cc_div + L.Lf_div_u

            if np.isclose(Lambda_seed, 0.0):
                convergence_allowed = False
                impurity_fraction = 0.0
            else:
                impurity_fraction = (qu**2 - b**2 * qcc**2) / (k * Lambda_seed) - Lambda_fixed / Lambda_seed

            # Use the divertor entrance temperature to calculate the divertor Zeff, which is used for
            # calculating the corrected electron heat conductivity in the next iteration
            divertor_z_effective = calc_z_effective(impurity_fraction, divertor_entrance_electron_temp, p)

            # Use the separatrix electron temperature to calculate Z-eff for alpha-t
            separatrix_z_effective = calc_z_effective(impurity_fraction, separatrix_electron_temp, p)
            alpha_t = calc_alpha_t(
                separatrix_electron_density=separatrix_electron_density * n20_to_m3,
                separatrix_electron_temp=separatrix_electron_temp,
                cylindrical_safety_factor=geo.cylindrical_safety_factor,
                major_radius=p.major_radius,
                average_ion_mass=p.average_ion_mass * amu_to_kg,
                z_effective=separatrix_z_effective,
                mean_ion_charge_state=1.0,
            )

            current = [alpha_t, impurity_fraction, separatrix_electron_temp]
            previous = [prev_alpha_t, prev_impurity_fraction, prev_separatrix_electron_temp]

        if pdiv:
            # Post-processing inside the loop: the computed neutral pressure drives the density update
            post = calc_target_postprocessing(target_electron_temp, parallel_heat_flux_at_target, geo, p)
            pdiv_factor = post.neutral_pressure_in_divertor / p.required_neutral_pressure_in_divertor
            new_separatrix_electron_density = separatrix_electron_density / pdiv_factor
            if mode == "inverse":
                # Inverse-pdiv updates the density immediately and without relaxation
                separatrix_electron_density = new_separatrix_electron_density
            current.append(post.neutral_pressure_in_divertor)
            previous.append(p.required_neutral_pressure_in_divertor)

        converged = np.allclose(current, previous, **CONVERGENCE)
        if pdiv:
            converged = converged and convergence_allowed

        if converged:
            break

        if mode == "forward":
            if it > 0:
                parallel_heat_flux_at_cc_interface = relax(parallel_heat_flux_at_cc_interface, prev_parallel_heat_flux_at_cc_interface)
                target_electron_temp = relax(target_electron_temp, prev_target_electron_temp)
                divertor_entrance_electron_temp = relax(divertor_entrance_electron_temp, prev_divertor_entrance_electron_temp)
                separatrix_electron_temp = relax(separatrix_electron_temp, prev_separatrix_electron_temp)
                alpha_t = relax(alpha_t, prev_alpha_t)
                if pdiv:
                    # NOTE: argument order is swapped relative to the other relax calls, giving an
                    # effective relaxation factor of 0.6 towards the new density (original behavior).
                    separatrix_electron_density = relax(separatrix_electron_density, new_separatrix_electron_density)
            prev_parallel_heat_flux_at_cc_interface = parallel_heat_flux_at_cc_interface
            prev_target_electron_temp = target_electron_temp
            prev_divertor_entrance_electron_temp = divertor_entrance_electron_temp
        else:
            prev_impurity_fraction = impurity_fraction
        prev_separatrix_electron_temp = separatrix_electron_temp
        prev_alpha_t = alpha_t
        convergence_allowed = True

    if not pdiv:
        post = calc_target_postprocessing(target_electron_temp, parallel_heat_flux_at_target, geo, p)

    return SolverResult(
        impurity_fraction=impurity_fraction,
        target_electron_temp=target_electron_temp,
        separatrix_electron_density=separatrix_electron_density,
        separatrix_electron_temp=separatrix_electron_temp,
        alpha_t=alpha_t,
        q_parallel=q_parallel,
        separatrix_z_effective=separatrix_z_effective,
        parallel_ion_flux_to_target=post.parallel_ion_flux_to_target,
        neutral_pressure_in_divertor=post.neutral_pressure_in_divertor,
        heat_flux_perp_to_target=post.heat_flux_perp_to_target,
        converged=converged,
    )
