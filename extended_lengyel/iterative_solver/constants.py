"""Physical constants, unit-conversion factors and fit coefficients shared by the extended Lengyel models."""

from typing import Any

import scipy.constants as sc

mu_0 = sc.mu_0  # meter * tesla / ampere
elementary_charge = sc.elementary_charge  # coulomb
amu_to_kg = sc.atomic_mass  # kilogram / amu
eV_to_J = sc.elementary_charge
boltzmann_constant = sc.Boltzmann  # joule/kelvin
electron_mass = sc.electron_mass  # kilogram
epsilon_0_per_eV = sc.epsilon_0 / sc.elementary_charge  # e^2 eV^-1 m^-1 (vacuum permittivity)
MA_to_A = 1.0e6  # MA / A
MW_to_W = 1.0e6  # MW / W
n20_to_m3 = 1.0e20  # 10^20 / m^3 to 1 / m^3

kappa_e0 = 2390.0  # W / (m * eV**3.5)

# Tolerances for the iterative solvers' convergence test (np.allclose kwargs)
CONVERGENCE: dict[str, Any] = dict(equal_nan=False, atol=0.0, rtol=1e-6)

# Coefficients for the convection-layer loss functions
# (equation 33 from Stangeby, 2018, PPCF 60 044022; see physics.temperature_fit_function)
MOMENTUM_LOSS_FIT = dict(amplitude=0.8858679172531956, width=3.8263045353064467, shape=0.8282347762381935)
POWER_LOSS_FIT = dict(amplitude=0.8532115334413933, width=5.195481324376164, shape=0.9642427916765323)
DENSITY_LOSS_FIT = dict(amplitude=0.5587910467003282, width=2.020427078509838, shape=0.9600157520406738)
