import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from run_unitless_forward_extended_lengyel_model import run_forward_extended_lengyel_model
    from run_unitless_extended_lengyel_model import run_inverse_extended_lengyel_model

    Tetar_1 = np.linspace(1, 20)
    conc_1 = np.zeros_like(Tetar_1)

    for i, Tetar in enumerate(Tetar_1):
        res = run_inverse_extended_lengyel_model(target_electron_temp=Tetar, fixed_impurity_concentrations=dict(), seed_impurity_weights=dict(Nitrogen= 1.0))

        conc_1[i] = res["Nitrogen_concentration"]

    plt.plot(Tetar_1, conc_1, label="Inverse")

    conc_2 = np.linspace(0.038, 0.052)
    Tetar_2 = np.zeros_like(conc_2)

    for i, conc in enumerate(conc_2):
        try:
            res = run_forward_extended_lengyel_model(impurity_concentrations=dict(Nitrogen=conc))
            Tetar_2[i] = res["target_electron_temp"]
        except AssertionError:
            Tetar_2[i] = np.nan

    plt.plot(Tetar_2, conc_2, label="Forward")

    plt.legend()
    plt.show()
