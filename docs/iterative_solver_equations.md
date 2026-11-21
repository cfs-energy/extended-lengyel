# Iterative solver: equations

The equations solved by `extended_lengyel.iterative_solver`, in the order the loop evaluates them. Each
section names the function that implements it. Source: Body, Kallenbach and Eich, NF 2025 (the "paper"),
with the extensions listed at the end. Units inside the code are SI, except electron temperatures in eV,
densities in n20 (10^20 m^-3) and ion masses in amu.

Notation: subscripts `u` upstream (separatrix), `cc` convection-conduction interface, `div` divertor
entrance (X-point), `t` target. `L_par` is `parallel_connection_length`, `L_div` is `divertor_parallel_length`,
`b_q` is `divertor_broadening_factor`, `b_pres` is `pressure_broadening_factor`, `f_enrich` is
`divertor_enrichment_factor`, `f_cond` is `SOL_conduction_fraction`, `fx_t = B_t/B_u`, `fx_xpt = B_xpt/B_u`.
`kappa_e0 = 2390 W m^-1 eV^-3.5`.

## 1. Geometry (`calc_geometry`)

    shape = sqrt((1 + kappa95^2 (1 + 2 delta95^2 - 1.2 delta95^3)) / 2)
    C_pol = 2 pi a shape
    B_pol,avg = mu_0 I_p / C_pol
    q_cyl = B_t0 / B_pol,avg * a / R * shape
    B_tor,u = B_t0 R / (R + a)
    B_pol,u = r_Bpol * B_pol,avg                 (r_Bpol = ratio_of_upstream_to_average_poloidal_field)
    pitch = sqrt(B_tor,u^2 + B_pol,u^2) / B_pol,u
    f_tube = (1 - 1/e) f_SOL->div
    r_lambda = r_Bpol (R + a) / R
    parallel_to_perp_factor = sin(angle of incidence)

## 2. Upstream heat flux (`calc_q_parallel`, `calc_alpha_t`)

    rho_s,pol = sqrt(T_u m_i) / B_pol,avg * sqrt(amu / e)                  [m]
    lambda_Te = 2.1 (1 + 2.1 alpha_t^1.7) rho_s,pol
    lambda_q  = (2/7) lambda_Te
    lambda_q,u = lambda_q / r_lambda
    q_u = P_sep f_tube / (2 pi (R + a) lambda_q,u) * pitch

`alpha_t` (Eich 2020, eq 9, with the extra `(1 + T_i/T_e / Z)` factor; collision frequencies from eqs B1, B2):

    ln Lambda = 30.9 - ln(sqrt(n_u [m^-3]) / T_u)
    nu_ee = (4/3) sqrt(2 pi) n_u lnLambda / ((4 pi eps_0')^2 sqrt(m_e) T_u^1.5) * sqrt(e)   (eps_0' = eps_0 / e)
    nu_ei = nu_ee * g(Z_eff) * Z_eff
    g(Z_eff) = (1 - 0.569) exp(-(max(Z_eff - 1, 0) / 3.25)^0.85) + 0.569
    c_s = sqrt(Z T_u e / m_i)
    alpha_t = 1.02 nu_ei / c_s * (m_e / m_i) q_cyl^2 R (1 + (T_i/T_e) / Z)

Called with Z = 1 and T_i/T_e = 1. `Z_eff` is the upstream value of section 4. Inverse-with-pdiv clamps
`alpha_t >= 0`.

## 3. Heat conductivity (`calc_heat_conductivity_factor`)

    kappa_z = 0.672 + 0.076 sqrt(Z_eff) + 0.252 Z_eff       (Brown and Goldston 2021, eq 10)
    K = kappa_e0 / kappa_z

`K_div` uses the divertor `Z_eff`, `K_up` the upstream `Z_eff` (both below).

## 4. Effective charge (`calc_z_effective`)

    Z_eff(T, f) = 1 + sum_seed Z(Z-1) c_z w / f + sum_fixed Z(Z-1) c_fixed / f,    Z = mean charge at T

`Z_eff,div` is evaluated at `T_div` with f = 1; `Z_eff,u` at `T_u` with f = `f_enrich`. The conductivity
legs use the values carried from the previous iteration (starting from 1).

## 5. Conduction ladder (`calc_electron_temp_at_div_and_u`)

Conductive flux is anchored upstream and scales as B, so each leg uses the mean field ratio of its endpoints:

    fx_cc,div = (fx_t + fx_xpt) / 2         fx_div,u = (fx_xpt + 1) / 2
    T_div = (T_cc^3.5 + 3.5 f_cond q_u / b_q * fx_cc,div * L_div / K_div)^(2/7)
    T_u   = (T_div^3.5 + 3.5 f_cond q_u * fx_div,u * (L_par - L_div) / K_up)^(2/7)

## 6. Target conditions (two-point model)

Convection-layer fit functions of `T_t` (Stangeby 2018, eq 33; coefficients in `constants.py`):

    f_x(T_t) = 1 - A_x (1 - exp(-T_t / w_x))^s_x          x in {mom, pow, n}
    T_cc = T_t * 2 f_n / (1 - f_mom)                       (`calc_cc_interface_temp`)

Basic target temperature and correction factor:

    p_u   = (1 + M_u^2) n_u T_u (1 + (T_i/T_e)_u / (n_e/n_i)_u) * (1e20 e)           [Pa]
    T_t,basic = 8 m_i / gamma^2 * q_u^2 / p_u^2 * amu / e                            (`calc_target_electron_temp_basic`)
    f_other = (1 + (T_i/T_e)_t / (n_e/n_i)_t) / 2 * (1 + M_t^2)^2 / (4 M_t^2) * fx_t^2   (`calc_f_other`)
    T_t = T_t,basic * f_vol_loss * f_other
    f_vol_loss = (1 - f_pow,SOL)^2 / ((1 - f_mom,c) / b_pres)^2

with `gamma` the sheath heat transmission factor and `f_pow,SOL` the power lost in the SOL above the cc interface,
referenced to the upstream flux:

    q_t = (1 - f_pow,c) q_cc                       (convection layer thin: cc and target share the field)
    1 - f_pow,SOL = q_t / (fx_t q_u)

## 7. Lengyel power balance (`solver.py`)

Radiated-power integrals `L_INT` of the seed (`Ls`) and fixed (`Lf`) impurities between temperature points
(`calc_lengyel_integrals`; `n20_to_m3^2` converts n20 to m^-3 squared):

    Lint_cc,div = c_z Ls(T_cc, T_div) + Lf(T_cc, T_div)
    Lint_div,u  = c_z Ls(T_div, T_u)  + Lf(T_div, T_u)
    k_base = 2 n_u^2 T_u^2                                                       (n_u in n20)

Power balance (the integral of the conduction equation with radiation, one equation per leg combined into one):

    b_q^2 q_cc^2 = q_u^2 + 2 t_B - k_base (K_div b_q^2 / b_pres^2 * Lint_cc,div + K_up * Lint_div,u / f_enrich)

**Forward** (c_z given): solve for `q_cc` (clamped at 0 when the right side is negative), then `q_t`, `f_pow,SOL`,
`T_t` of section 6.

**Inverse** (`T_t` given): section 6 inverted gives the loss and fluxes,

    1 - f_pow,SOL = sqrt(T_t / (T_t,basic f_other)) (1 - f_mom,c) / b_pres
    q_t  = q_u (1 - f_pow,SOL) fx_t
    q_cc = q_t / (1 - f_pow,c)

and the balance above is linear in `c_z` (this is the reformulated solve, combining eqs 40 and 43 of the paper,
which removes the intermediate divertor-entrance heat flux):

    Lambda_x = K_div b_q^2 / b_pres^2 * L_x(T_cc, T_div) + K_up * L_x(T_div, T_u) / f_enrich     x in {seed, fixed}
    c_z = (q_u^2 - b_q^2 q_cc^2 + 2 t_B) / (k_base Lambda_seed) - Lambda_fixed / Lambda_seed

`t_B` in the inverse solve uses the previous iterate's `c_z` (zero at the first), so the solve stays explicit.

## 8. Flux-expansion term (`calc_tB`)

Moulton et al. 2021, NF 61 046029, eqs 10-11: the conserved conductive quantity is `q/B`, which adds
`t_B = -integral (q^2/B) dB` to the balance. With the X-point flux reduced by div->u radiation,

    q_xpt,u^2 = max(q_u^2 - k_base K_up Lint_div,u / f_enrich, 0)
    <q^2>_div,u = (q_xpt,u^2 + q_u^2) / 2
    t_B = <q^2>_div,u (fx_xpt^2 - 1) / 2 + q_xpt,u^2 (fx_t^2 - fx_xpt^2) / 2

Zero when `fx_t = fx_xpt = 1`. Assumptions and accuracy are in the `calc_tB` docstring.

## 9. Divertor neutral pressure (`calc_target_postprocessing`)

    c_s,t = sqrt(2 T_t e / (amu m_i))                                     (`m_i` in amu)
    n_t   = q_t / (gamma T_t e c_s,t)
    Gamma_t = n_t c_s,t
    F = sqrt(2 / (pi r_mol m_i T_wall)) / sqrt(amu k_B)                    (paper eq 57; flux per pascal)
    p_n = Gamma_t sin(angle) / F
    q_perp = q_t sin(angle)

`r_mol` is `ratio_of_molecular_to_ion_mass`, `T_wall` the wall temperature in K.

## 10. Iteration

- State carried between iterations: `T_u`, `alpha_t`, `Z_eff,u`; forward also `T_t`, `T_cc`, `T_div`, `q_cc`.
- Forward starts at `T_t = 2 eV`, `T_cc = 2.5 eV`, `T_div = 50 eV`, `T_u = 100 eV`, `alpha_t = 0`.
- Forward relaxes `q_cc, T_t, T_div, T_u, alpha_t` as `0.4 new + 0.6 old` from the second iteration. Inverse
  takes the direct `c_z` each iteration without relaxation.
- With a divertor neutral pressure constraint the density is updated `n_u <- n_u / (p_n,computed / p_n,required)`;
  starting guess `n_u = 0.265 p_n,required^0.31`. Inverse takes the new value directly; forward blends it
  (`0.4 old + 0.6 new`). A zero or non-finite pressure ratio holds the density and blocks convergence.
- Convergence: `np.allclose(current, previous, rtol=1e-6, atol=0)` over `[alpha_t, T_t, T_div, T_u, q_cc]`
  (forward), `[alpha_t, c_z, T_u]` (inverse), with `[p_n, p_n,required]` appended for the pdiv variants.
- Physicality gate: a converged inverse solve with `c_z < 0` or `Z_eff,u < 1`, or a converged forward solve with
  `T_t < 0`, is reported as not converged. A non-converged solve returns NaN for every physics output.

## Extensions relative to the paper

`b_pres`, `f_enrich` (and the separate divertor and upstream `Z_eff` in the conductivity), the Moulton
`t_B` term and the `fx_t`, `fx_xpt` factors in the ladder and target. All reduce to the paper's equations at
their default values of 1.
