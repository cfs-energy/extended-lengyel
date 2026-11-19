"""Physical constants, unit-conversion factors and fit coefficients shared by the extended Lengyel models."""

mu_0 = 1.2566370621250601e-6  # meter * tesla / ampere
elementary_charge = 1.602176634e-19  # coulomb
MA_to_A = 1.0e6  # MA / A
amu_to_kg = 1.6605390666e-27  # amu / kilogram
MW_to_W = 1.0e6  # MW / W
eV_to_J = elementary_charge
boltzmann_constant = 1.380649e-23  # joule/kelvin
n20_to_m3 = 1.0e20  # 10^20 / m^3 to 1 / m^3

kappa_e0 = 2390.0  # W / (m * eV**3.5)

# Tolerances for the iterative solvers' convergence test (np.allclose kwargs)
CONVERGENCE = dict(equal_nan=False, atol=0.0, rtol=1e-6)

# Coefficients for the convection-layer loss functions
# (equation 33 from Stangeby, 2018, PPCF 60 044022; see physics.temperature_fit_function)
MOMENTUM_LOSS_FIT = dict(amplitude=0.8858679172531956, width=3.8263045353064467, shape=0.8282347762381935)
POWER_LOSS_FIT = dict(amplitude=0.8532115334413933, width=5.195481324376164, shape=0.9642427916765323)
DENSITY_LOSS_FIT = dict(amplitude=0.5587910467003282, width=2.020427078509838, shape=0.9600157520406738)
