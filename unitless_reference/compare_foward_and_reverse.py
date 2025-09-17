import numpy as np
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("error")

if __name__ == "__main__":
    from run_unitless_forward_extended_lengyel_model import run_forward_extended_lengyel_model
    from run_unitless_extended_lengyel_model import run_inverse_extended_lengyel_model

    species = "Neon"

    Tetar_1 = np.linspace(1, 40)
    conc_1 = np.zeros_like(Tetar_1)
    converged_1 = np.zeros_like(Tetar_1, dtype=bool)

    for i, Tetar in enumerate(Tetar_1):
        res = run_inverse_extended_lengyel_model(
            target_electron_temp=Tetar, fixed_impurity_concentrations=dict(), seed_impurity_weights={species: 1.0},
            iterations = 1000,
        )

        converged_1[i] = res["converged"]
        conc_1[i] = res[f"{species}_concentration"]

    conc_2 = np.linspace(0.0, 0.05)
    Tetar_2 = np.zeros_like(conc_2)
    converged_2 = np.zeros_like(Tetar_2, dtype=bool)

    for i, conc in enumerate(conc_2):
        try:
            res = run_forward_extended_lengyel_model(impurity_concentrations={species: conc},
                iterations = 1000,
            )

            converged_2[i] = res["converged"]
            Tetar_2[i] = res["target_electron_temp"]
        except AssertionError:
            Tetar_2[i] = np.nan
            converged_2[i] = False

    plt.plot(conc_2[converged_2] * 1e2, Tetar_2[converged_2], label="Forward")
    plt.plot(conc_1[converged_1] * 1e2, Tetar_1[converged_1], label="Inverse")
    plt.xlabel(f"{species} concentration required [%]")
    plt.ylabel("Target electron temp [eV]")

    plt.axvline(0.0, color="k")
    plt.axhline(0.0, color="k")

    plt.legend()
    plt.show()
