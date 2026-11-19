"""Spatial (ODE-integrated) Lengyel model, used as a flux-expansion reference.

This is a direct integration of the 1D parallel power balance along the field line, from
the convection-conduction (cc) interface near the target up to the upstream point. Unlike
the reduced extended Lengyel model (``solver.py``), which evaluates the radiation as a
temperature integral under a constant-field assumption, this model integrates the
conservative field-line energy equation

    B d/ds (q_parallel / B) = P_radiation                       (Moulton et al. 2021 eq 1)

with ``scipy.integrate.solve_ivp``. Written out, ``dq_parallel/ds = P_radiation +
q_parallel (1/B) dB/ds``: the second term is the magnetic-flux-expansion correction, whose
integral is the ``t_B = - integral (q_parallel**2 / B) dB`` term of Moulton et al. 2021
(NF 61 046029, eqs 10-11). This model captures it exactly, so it is the ground-truth
reference against which the reduced model's closed-form ``t_B`` treatment is validated; it
is not part of the fitting/figure pipeline.

The radiation uses the same ``CzLINT_integrator`` cooling-curve splines as the reduced
model: the pointwise cooling curve ``sum_z w_z L_z(T)`` is the temperature derivative of
the ``CzLINT`` integral (which integrates ``sum_z w_z L_z(T) sqrt(T)``), so the two models
share their atomic physics exactly and the comparison isolates the flux-expansion
treatment.

The field strength varies **linearly in the parallel coordinate** (a constant gradient
``dB/ds`` within each leg): ``B`` runs ``B_t`` (cc/target) -> ``B_xpt`` (X-point) ->
``B_u`` (upstream). Only the field-strength ratios ``B_t/B_u`` and ``B_xpt/B_u`` enter
(the absolute scale cancels in ``(1/B) dB/ds``); a globally constant gradient is the
special case where ``B_xpt`` is the linear interpolant of ``B_t`` and ``B_u``. The exact
per-leg ``t_B = - integral (q_parallel**2 / B) dB`` is accumulated as an ODE state, so it
is reported without any further approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.integrate import solve_ivp

from .physics import calc_heat_conductivity_factor

if TYPE_CHECKING:
    from extended_lengyel.extended_lengyel_model.Lengyel_model_core import CzLINT_integrator


def cooling_curve(czlint: CzLINT_integrator, electron_temp: float, *, rel_step: float = 1e-3) -> float:
    """Evaluate the weighted impurity cooling curve ``sum_z w_z L_z(T_e)`` (W m^3) at ``electron_temp`` (eV).

    ``CzLINT_integrator.unitless_eval(a, b)`` returns
    ``integral_a^b sum_z w_z L_z(T) sqrt(T) dT``; the integrand at the upper limit is its
    derivative there, recovered by a central difference. Dividing by ``sqrt(T_e)`` removes
    the ``sqrt(T)`` weight, leaving the cooling curve itself. An empty integrator returns 0.
    """
    if czlint.is_empty:
        return 0.0
    h = rel_step * electron_temp
    # unitless_eval clamps stop = max(stop, start); here stop > start so it is a plain integral.
    integrand = czlint.unitless_eval(electron_temp - h, electron_temp + h) / (2.0 * h)  # sum_z w_z L_z sqrt(T)
    return float(integrand) / np.sqrt(electron_temp)


def _change_in_state_vector(
    s_parallel: float,
    state_vector: tuple[float, float, float, float, float],
    impurity_fraction: float,
    czlint_seed: CzLINT_integrator,
    czlint_fixed: CzLINT_integrator,
    heat_conductivity_factor: float,
    s_start: float,
    field_strength_at_start: float,
    field_gradient: float,
) -> tuple[float, float, float, float, float]:
    """Derivative of the state vector w.r.t. the parallel coordinate.

    The state vector is ``(q_parallel, n_e, T_e, integral (q_parallel**2 / B) dB)``.
    ``heat_conductivity_factor`` is ``kappa_e0/kappa_z``. The field varies linearly:
    ``B(s) = field_strength_at_start + field_gradient*(s - s_start)``; the
    ``q_parallel*(1/B)dB/ds`` term is the conservative-form flux-expansion correction
    (vanishes when ``field_gradient`` is zero). The last state accumulates the exact
    ``integral (q_parallel**2 / B) dB`` so the leg's ``t_B`` is just its negative.
    """
    parallel_heat_flux, electron_density, electron_temp, _integral_q2_over_B = state_vector
    field_strength = field_strength_at_start + field_gradient * (s_parallel - s_start)

    weighted_Lz = impurity_fraction * cooling_curve(czlint_seed, electron_temp) + cooling_curve(czlint_fixed, electron_temp)
    radiated_power = weighted_Lz * electron_density**2  # W / m^3

    change_in_electron_temp = parallel_heat_flux / (heat_conductivity_factor * electron_temp**2.5)  # eV / m
    change_in_parallel_heat_flux = radiated_power + parallel_heat_flux * field_gradient / field_strength  # W / m^3
    change_in_electron_density = -electron_density / electron_temp * change_in_electron_temp
    change_in_integral_q2_over_B = parallel_heat_flux**2 * field_gradient / field_strength  # = (q^2/B) dB/ds

    return (
        change_in_parallel_heat_flux,
        change_in_electron_density,
        change_in_electron_temp,
        change_in_integral_q2_over_B,
    )


@dataclass(frozen=True)
class SpatialLengyelResult:
    """Endpoint state and per-leg flux-expansion diagnostics from the spatial Lengyel integration.

    All quantities are SI magnitudes (q in W/m^2, n_e in m^-3, T_e in eV). ``tB_*`` is the
    exact flux-expansion term for that leg, ``- integral (q_parallel**2 / B) dB`` (> 0 for B
    decreasing upstream). ``cc -> div`` is the divertor (target-to-X-point) leg, ``div -> u``
    the main-chamber (X-point-to-upstream) leg.
    """

    converged: bool
    # Divertor entrance (X-point), divertor-leg side (before the b_q broadening jump)
    divertor_entrance_parallel_heat_flux: float
    divertor_entrance_electron_density: float
    divertor_entrance_electron_temp: float
    # Upstream
    q_parallel: float
    separatrix_electron_density: float
    separatrix_electron_temp: float
    # Exact per-leg flux-expansion diagnostics
    tB_cc_div: float
    tB_div_u: float


def run_spatial_lengyel_model(
    *,
    electron_density_at_cc_interface: float,  # m^-3
    electron_temp_at_cc_interface: float,  # eV
    parallel_heat_flux_at_cc_interface: float,  # W/m^2
    impurity_fraction: float,
    CzLINT_for_seed_impurities: CzLINT_integrator,
    CzLINT_for_fixed_impurities: CzLINT_integrator,
    divertor_heat_conductivity_factor: float,
    upstream_heat_conductivity_factor: float,
    divertor_broadening_factor: float,
    s_parallel_at_cc_interface: float,  # m
    divertor_parallel_length: float,  # m
    parallel_connection_length: float,  # m
    flux_expansion_Bt_over_Bu: float = 1.0,
    flux_expansion_Bxpt_over_Bu: float = 1.0,
    ode_method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "LSODA",
    ode_rtol: float = 1e-9,
    ode_atol: float = 1e-12,
) -> SpatialLengyelResult:
    """Integrate the spatial Lengyel model from the cc-interface (near target) to upstream.

    The integration proceeds in two legs: ``cc -> div`` (target to X-point) using the
    divertor heat-conductivity factor, then ``div -> u`` (X-point to upstream) using the
    upstream one. At the divertor entrance the heat flux is broadened by
    ``divertor_broadening_factor`` (b_q), matching the reduced model's step treatment of
    the X-point. The field strength varies linearly with a constant gradient within each
    leg, running ``B_t`` (cc/target) -> ``B_xpt`` (X-point) -> ``B_u`` (upstream); only the
    ratios ``flux_expansion_Bt_over_Bu`` and ``flux_expansion_Bxpt_over_Bu`` enter (default
    1.0 = no flux expansion, inert term).

    Returns all-NaN with ``converged=False`` if either leg fails to integrate.
    """
    # Field strengths (B_u as reference; only ratios enter). B varies linearly in s within each
    # leg, so dB/ds is a constant per leg (a globally constant gradient corresponds to B_xpt at
    # the linear interpolant of B_t and B_u).
    field_strength_at_target = flux_expansion_Bt_over_Bu
    field_strength_at_xpt = flux_expansion_Bxpt_over_Bu
    field_strength_upstream = 1.0
    leg1_length = divertor_parallel_length - s_parallel_at_cc_interface
    leg2_length = parallel_connection_length - divertor_parallel_length
    field_gradient_cc_div = (field_strength_at_xpt - field_strength_at_target) / leg1_length
    field_gradient_div_u = (field_strength_upstream - field_strength_at_xpt) / leg2_length

    atol_vector = [
        ode_atol * scale
        for scale in (
            parallel_heat_flux_at_cc_interface,
            electron_density_at_cc_interface,
            electron_temp_at_cc_interface,
            parallel_heat_flux_at_cc_interface**2,
        )
    ]

    def integrate_leg(s_span, y0, heat_conductivity_factor, field_strength_at_start, field_gradient):
        # Reset the cumulative integral (q^2 / B) dB at the start of each leg.
        return solve_ivp(
            _change_in_state_vector,
            s_span,
            y0=[y0[0], y0[1], y0[2], 0.0],
            args=(
                impurity_fraction,
                CzLINT_for_seed_impurities,
                CzLINT_for_fixed_impurities,
                heat_conductivity_factor,
                s_span[0],
                field_strength_at_start,
                field_gradient,
            ),
            method=ode_method,
            dense_output=False,
            rtol=ode_rtol,
            atol=atol_vector,
        )

    # The ODE underflows the cooling curve to zero at low T and can probe states the stepper
    # rejects; treat any failure as non-convergence rather than poisoning the caller.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        try:
            res1 = integrate_leg(
                (s_parallel_at_cc_interface, divertor_parallel_length),
                (parallel_heat_flux_at_cc_interface, electron_density_at_cc_interface, electron_temp_at_cc_interface),
                divertor_heat_conductivity_factor,
                field_strength_at_target,
                field_gradient_cc_div,
            )
            res2 = integrate_leg(
                (divertor_parallel_length, parallel_connection_length),
                (res1.y[0, -1] * divertor_broadening_factor, res1.y[1, -1], res1.y[2, -1]),
                upstream_heat_conductivity_factor,
                field_strength_at_xpt,
                field_gradient_div_u,
            )
            model_success = bool(res1.success and res2.success)
        except (ValueError, IndexError, FloatingPointError, RuntimeWarning):
            model_success = False

    if not model_success:
        nan = float("nan")
        return SpatialLengyelResult(False, nan, nan, nan, nan, nan, nan, nan, nan)

    return SpatialLengyelResult(
        converged=True,
        divertor_entrance_parallel_heat_flux=res1.y[0, -1],
        divertor_entrance_electron_density=res1.y[1, -1],
        divertor_entrance_electron_temp=res1.y[2, -1],
        q_parallel=res2.y[0, -1],
        separatrix_electron_density=res2.y[1, -1],
        separatrix_electron_temp=res2.y[2, -1],
        # t_B = - integral (q^2 / B) dB, accumulated exactly as the fourth state.
        tB_cc_div=-res1.y[3, -1],
        tB_div_u=-res2.y[3, -1],
    )


def heat_conductivity_factor_from_z_effective(z_effective: float) -> float:
    """Convenience wrapper: the impurity-corrected heat-conductivity factor kappa_e0/kappa_z."""
    return calc_heat_conductivity_factor(z_effective)
