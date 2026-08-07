"""Iterative solvers for the extended Lengyel model.

Two model variants share the single iteration loop in :mod:`.solver`, and are
registered in the cfspopcon Algorithm registry under their function names:

* ``run_forward_extended_lengyel_model`` fixes the impurity concentration and
  solves for the target electron temperature, by relaxed fixed-point iteration.
* ``run_inverse_extended_lengyel_model`` fixes the target electron temperature
  and solves for the required impurity concentration directly each iteration.

Both implement the model of Body, Kallenbach and Eich, NF 2025, using the
reformulated Lengyel solve that combines equations 40 and 43 of that paper.
:mod:`extended_lengyel.extended_lengyel_model` publishes a second, independent
implementation of the inverse solve built from the modular algorithms — see the
note in :mod:`.models`.

A non-converged solve returns NaN for every physics output, so always check the
returned ``converged`` flag.
"""

from .models import run_forward_extended_lengyel_model, run_inverse_extended_lengyel_model

__all__ = [
    "run_forward_extended_lengyel_model",
    "run_inverse_extended_lengyel_model",
]
