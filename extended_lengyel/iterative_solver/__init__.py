"""Iterative solvers for the extended Lengyel model.

Four model variants share the single iteration loop in :mod:`.solver`, and are
registered in the cfspopcon Algorithm registry under their function names:

* ``run_forward_extended_lengyel_model`` fixes the impurity concentration and
  solves for the target electron temperature, by relaxed fixed-point iteration.
* ``run_inverse_extended_lengyel_model`` fixes the target electron temperature
  and solves for the required impurity concentration directly each iteration.
* ``run_forward_extended_lengyel_model_pdiv`` and
  ``run_inverse_extended_lengyel_model_with_pdiv`` do the same with the
  separatrix density unknown, constrained by the divertor neutral pressure.

The loop implements Body, Kallenbach and Eich, NF 2025, using the reformulated
Lengyel solve that combines equations 40 and 43 of that paper, plus pressure
broadening, divertor enrichment, region-specific heat conductivity and the
Moulton 2021 flux-expansion term. :mod:`extended_lengyel.extended_lengyel_model` publishes an independent
implementation of the published inverse solve — see the note in :mod:`.models`.

A non-converged solve returns NaN for every physics output, so always check the
returned ``converged`` flag.
"""

from .models import (
    run_forward_extended_lengyel_model,
    run_forward_extended_lengyel_model_pdiv,
    run_inverse_extended_lengyel_model,
    run_inverse_extended_lengyel_model_with_pdiv,
)

__all__ = [
    "run_forward_extended_lengyel_model",
    "run_forward_extended_lengyel_model_pdiv",
    "run_inverse_extended_lengyel_model",
    "run_inverse_extended_lengyel_model_with_pdiv",
]
