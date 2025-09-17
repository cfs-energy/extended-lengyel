import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from run_unitless_forward_extended_lengyel_model import run_forward_extended_lengyel_model
    from run_unitless_extended_lengyel_model import run_inverse_extended_lengyel_model

    Tetar_1 = np.linspace(1, 30)
    conc_1 = np.zeros_like(Tetar_1)

    for i, Tetar in enumerate(Tetar_1):
        res = run_inverse_extended_lengyel_model(
            target_electron_temp=Tetar, fixed_impurity_concentrations=dict(), seed_impurity_weights=dict(Nitrogen= 1.0),
            iterations = 50,
        )

        if res["converged"]:
            conc_1[i] = res["Nitrogen_concentration"]

    conc_2 = np.linspace(0.01, 0.05)
    Tetar_2 = np.zeros_like(conc_2)

    for i, conc in enumerate(conc_2):
        try:
            res = run_forward_extended_lengyel_model(impurity_concentrations=dict(Nitrogen=conc),
                iterations = 50,
            )

            if res["converged"]:
                Tetar_2[i] = res["target_electron_temp"]
        except AssertionError:
            Tetar_2[i] = np.nan

    Tetar_as_x = False

    if Tetar_as_x:
        plt.scatter(Tetar_2, conc_2, label="Forward")
        plt.scatter(Tetar_1, conc_1, label="Inverse")
        plt.xlabel("Target electron temp [eV]")
        plt.ylabel("Impurity concentration required")
    else:
        plt.scatter(conc_2, Tetar_2, label="Forward")
        plt.scatter(conc_1, Tetar_1, label="Inverse")
        plt.xlabel("Impurity concentration required")
        plt.ylabel("Target electron temp [eV]")

    plt.axvline(0.0, color="k")
    plt.axhline(0.0, color="k")

    plt.legend()
    plt.show()
