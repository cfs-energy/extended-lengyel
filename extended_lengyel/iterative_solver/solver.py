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

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np

from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator

from .constants import CONVERGENCE, MA_to_A, MW_to_W, amu_to_kg, n20_to_m3
from .physics import (
    calc_alpha_t,
    calc_cc_interface_temp,
    calc_convection_layer_losses,
    calc_f_other,
    calc_geometry,
    calc_heat_conductivity_factor,
    calc_lengyel_integrals,
    calc_q_parallel,
    calc_separatrix_total_pressure,
    calc_target_electron_temp_basic,
    calc_target_postprocessing,
    calc_tB,
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
    flux_expansion_Bt_over_Bu: float
    iterations: int
    # Pressure broadening and divertor enrichment. Defaults of 1.0 reproduce the published
    # model. divertor_broadening_factor is the heat-flux broadening b_q;
    # pressure_broadening_factor is the pressure broadening b_pres (divertor static pressure =
    # n_eu*T_eu/b_pres; equivalently divertor density = n_eu/b_pres, since T_e is continuous at the X-point);
    # divertor_enrichment_factor is f_enrich (main-chamber concentration = c_z/f_enrich).
    pressure_broadening_factor: float = 1.0
    divertor_enrichment_factor: float = 1.0
    # X-point/upstream field-strength ratio B_xpt/B_u. With flux_expansion_Bt_over_Bu (= B_t/B_u) it
    # tracks magnetic flux expansion on both conduction legs (u<->xpt and xpt<->t): it sets the per-leg
    # mean field in the conduction ladder (calc_temperature_ladder) and the per-leg field factors in
    # t_B (calc_tB). At the default 1.0 there is no flux expansion and both are inert.
    flux_expansion_Bxpt_over_Bu: float = 1.0
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
    # Upstream Z_eff, carried across iterations to set the main-chamber leg's heat-conductivity factor
    separatrix_z_effective = 1.0
    prev_separatrix_electron_temp = np.nan
    prev_alpha_t = np.nan

    if pdiv:
        # Initial guess
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
        divertor_heat_conductivity_factor = calc_heat_conductivity_factor(divertor_z_effective)
        # Upstream (main-chamber) conduction uses the upstream Z_eff (enriched-reduced, evaluated at
        # the separatrix temperature) carried over from the previous iteration; the divertor leg uses
        # the divertor Z_eff.
        upstream_z_effective = np.maximum(separatrix_z_effective, 0.0) if clamp_negative_values else separatrix_z_effective
        upstream_heat_conductivity_factor = calc_heat_conductivity_factor(upstream_z_effective)

        divertor_entrance_electron_temp, separatrix_electron_temp = calc_temperature_ladder(
            electron_temp_at_cc_interface,
            q_parallel,
            divertor_heat_conductivity_factor,
            upstream_heat_conductivity_factor,
            p,
        )

        qu = q_parallel
        b_q = p.divertor_broadening_factor  # heat-flux broadening b_q
        b_pres = p.pressure_broadening_factor  # pressure broadening b_pres
        f_enrich = p.divertor_enrichment_factor  # divertor enrichment f_enrich

        if mode == "forward":
            # Upstream (main-chamber) Z_eff: the impurity concentration above the X-point is reduced
            # to c_z / f_enrich, for all species (seed and fixed). Unlike the divertor Z_eff above.
            separatrix_z_effective = calc_z_effective(impurity_fraction, separatrix_electron_temp, p, enrichment_factor=f_enrich)
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

            k_base = 2.0 * separatrix_electron_density**2 * separatrix_electron_temp**2
            # Magnetic-flux-expansion correction (Moulton 2021 t_B); inert at the default field ratios.
            tB = calc_tB(
                qu,
                k_base,
                upstream_heat_conductivity_factor,
                Lint_div_u,
                f_enrich,
                p.flux_expansion_Bt_over_Bu,
                p.flux_expansion_Bxpt_over_Bu,
            )
            # Generalized forward power balance: divertor integral weighted b_q^2/b_pres^2, main by 1/f_enrich;
            # each integral carries its own region's heat-conductivity factor (divertor vs upstream Z_eff).
            # The +2 t_B term adds the radiation released by the field-strength variation along the legs.
            qcc_squared = (
                qu**2
                + 2.0 * tB
                - k_base
                * (
                    divertor_heat_conductivity_factor * b_q**2 / b_pres**2 * Lint_cc_div
                    + upstream_heat_conductivity_factor * Lint_div_u / f_enrich
                )
            ) / b_q**2
            if qcc_squared < 0:
                convergence_allowed = False
                qcc_squared = 0.0
            parallel_heat_flux_at_cc_interface = np.sqrt(qcc_squared)

            separatrix_total_pressure = calc_separatrix_total_pressure(separatrix_electron_density, separatrix_electron_temp, p)
            target_electron_temp_basic = calc_target_electron_temp_basic(q_parallel, separatrix_total_pressure, p)
            f_other = calc_f_other(p)

            # Calculate Te_tar consistent with parallel_heat_flux_at_cc_interface
            losses = calc_convection_layer_losses(target_electron_temp)
            # The actual parallel heat flux at the target follows from the convection-layer loss on
            # the cc-interface flux alone (1 - f_pow,c = q_t / q_cc, no flux-expansion factor): the
            # convection layer is only a few mm wide, so cc and the target sit at the same field
            # strength. The B_t/B_u amplification toward the higher-field target is carried by q_cc
            # itself (the Lengyel balance ties it to the actual upstream flux q_parallel). The SOL
            # power-loss fraction is referenced to the upstream power across the conduction region
            # cc -> u (where the field expands from B_t to B_u), so it carries B_u/B_t =
            # 1 / flux_expansion_Bt_over_Bu to stay flux-expansion free. Inert at the default 1.
            parallel_heat_flux_at_target = (1.0 - losses.power) * parallel_heat_flux_at_cc_interface
            SOL_power_loss_fraction = 1.0 - parallel_heat_flux_at_target / p.flux_expansion_Bt_over_Bu / q_parallel
            # b_pres enhances the total (upstream-referenced) momentum loss: (1 - f_mom) = (1 - f_mom,c) / b_pres.
            # On the divergent branch (deep detachment, no pdiv guard to clamp the target temperature)
            # both the target heat flux and the momentum-loss complement go to zero, so this is a 0/0 ->
            # nan. The nan correctly flags non-convergence (allclose then fails), so silence the expected
            # invalid-value warning rather than alter the result. Convergent cases keep both terms finite.
            with np.errstate(divide="ignore", invalid="ignore"):
                f_vol_loss = (1.0 - SOL_power_loss_fraction) ** 2 / ((1.0 - losses.momentum) / b_pres) ** 2

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
            # b_pres enhances the total (upstream-referenced) momentum loss: (1 - f_mom) = (1 - f_mom,c) / b_pres
            required_power_loss = 1.0 - np.sqrt(
                target_electron_temp / target_electron_temp_basic * ((1.0 - losses.momentum) / b_pres) ** 2 / f_other
            )
            # The actual parallel heat flux at the target: q_parallel reduced by the (flux-expansion
            # free) SOL power loss and amplified toward the higher-field target by B_t/B_u =
            # flux_expansion_Bt_over_Bu across the conduction region (cc -> u). The cc-interface flux
            # follows from the convection-layer loss alone (1 - f_pow,c = q_t / q_cc): the convection
            # layer is only a few mm wide, so cc and the target sit at the same field strength and the
            # cc flux carries the same B_t/B_u amplification (no extra factor). Inert at the default 1.
            parallel_heat_flux_at_target = q_parallel * (1.0 - required_power_loss) * p.flux_expansion_Bt_over_Bu  # W/m^2
            parallel_heat_flux_at_cc_interface = parallel_heat_flux_at_target / (1.0 - losses.power)

            L = calc_lengyel_integrals(electron_temp_at_cc_interface, divertor_entrance_electron_temp, separatrix_electron_temp, p)
            qcc = parallel_heat_flux_at_cc_interface
            k_base = 2.0 * separatrix_electron_density**2 * separatrix_electron_temp**2

            # Reformulated Lengyel solve (combining equations 40 and 43 of Body, Kallenbach and
            # Eich, NF 2025): the seed-impurity concentration multiplier follows directly from
            # the power balance, without the intermediate divertor-entrance heat flux q_div.
            # Generalized weights: divertor integral by b_q^2/b_pres^2, main-chamber integral by 1/f_enrich;
            # each integral carries its own region's heat-conductivity factor (divertor vs upstream Z_eff).
            Lambda_seed = (
                divertor_heat_conductivity_factor * b_q**2 / b_pres**2 * L.Ls_cc_div
                + upstream_heat_conductivity_factor * L.Ls_div_u / f_enrich
            )
            Lambda_fixed = (
                divertor_heat_conductivity_factor * b_q**2 / b_pres**2 * L.Lf_cc_div
                + upstream_heat_conductivity_factor * L.Lf_div_u / f_enrich
            )

            # Magnetic-flux-expansion correction (Moulton 2021 t_B), inert at the default field ratios.
            # The div->u radiation integral is weighted by the previous iterate's impurity fraction
            # (lagged, like the upstream Z_eff) so the direct c_z solve stays explicit; nan-safe at
            # the first iteration.
            cz_lag = impurity_fraction if np.isfinite(impurity_fraction) else 0.0
            tB = calc_tB(
                qu,
                k_base,
                upstream_heat_conductivity_factor,
                cz_lag * L.Ls_div_u + L.Lf_div_u,
                f_enrich,
                p.flux_expansion_Bt_over_Bu,
                p.flux_expansion_Bxpt_over_Bu,
            )

            if np.isclose(Lambda_seed, 0.0):
                convergence_allowed = False
                impurity_fraction = 0.0
            else:
                # +2 t_B adds the radiation released by the field-strength variation along the legs.
                impurity_fraction = (qu**2 - b_q**2 * qcc**2 + 2.0 * tB) / (k_base * Lambda_seed) - Lambda_fixed / Lambda_seed

            # Use the divertor entrance temperature to calculate the divertor Zeff, which is used for
            # calculating the corrected electron heat conductivity in the next iteration
            divertor_z_effective = calc_z_effective(impurity_fraction, divertor_entrance_electron_temp, p)

            # Use the separatrix electron temperature to calculate Z-eff for alpha-t
            # Upstream (main-chamber) Z_eff: the impurity concentration above the X-point is reduced
            # to c_z / f_enrich, for all species (seed and fixed). Unlike the divertor Z_eff above.
            separatrix_z_effective = calc_z_effective(impurity_fraction, separatrix_electron_temp, p, enrichment_factor=f_enrich)
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
            if np.isfinite(pdiv_factor) and pdiv_factor > 0:
                new_separatrix_electron_density = separatrix_electron_density / pdiv_factor
            else:
                # No separatrix density satisfies both the pressure constraint and a non-negative
                # cc-interface heat flux at this impurity fraction: the qcc clamp above zeroed the
                # target heat flux, so the computed neutral pressure (hence pdiv_factor) collapsed to
                # zero. Dividing here would send the density to inf and poison later iterations with
                # nan (the historical divergent branch, quirk #3). Instead hold the density and block
                # convergence, so the model reports a finite, non-converged state. Note: cases that
                # reach this branch do not recover (verified) -- the relaxed fixed point has no stable
                # solution here, so this is a clean stop, not a step towards convergence.
                convergence_allowed = False
                new_separatrix_electron_density = separatrix_electron_density
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

    # Physicality gate: reject a converged-but-unphysical fixed point (reported as all-NaN),
    # even when the fixed-point iteration "converges":
    #   - inverse: a negative impurity fraction (equivalently separatrix Z_eff < 1) means the measured
    #     target is unreachable with non-negative seeding.
    #   - forward: a negative target electron temperature is no physical target state.
    # Forward takes the impurity fraction as a non-negative input, so only the target-temperature
    # check applies there; the inverse solves are where the impurity fraction is solved for.
    if converged:
        if mode == "inverse" and (impurity_fraction < 0.0 or separatrix_z_effective < 1.0):
            converged = False
        elif mode == "forward" and target_electron_temp < 0.0:
            converged = False

    if not pdiv:
        post = calc_target_postprocessing(target_electron_temp, parallel_heat_flux_at_target, geo, p)

    result = SolverResult(
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
    if not converged:
        # A non-converged solve has no valid solution: report NaN for every physics output
        # (the last-iterate values are iteration-capped and trajectory-sensitive) while keeping
        # the converged flag so callers can detect and mask the failure.
        result = replace(result, **{f.name: np.nan for f in fields(result) if f.name != "converged"})
    return result
