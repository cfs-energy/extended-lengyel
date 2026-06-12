"""Iterative extended Lengyel solvers (forward/inverse, with and without a divertor pressure constraint).

The four model variants share a single iteration loop (see :mod:`.solver`) and are
registered in the cfspopcon Algorithm registry when this subpackage is imported.

Note: the standalone ``extended_lengyel_validation`` package registers the same
algorithm names — do not import both in the same session.
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
