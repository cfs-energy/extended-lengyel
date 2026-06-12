"""Shared impurity-interpolator setup for the iterative extended Lengyel models.

Builds the CzLINT and mean-charge interpolators that the models take as inputs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from cfspopcon.formulas.atomic_data import read_atomic_data
from cfspopcon.helpers import get_item
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import Unitfull, magnitude_in_units, ureg

from ..config import setup_impurities
from ..directories import radas_dir as RADAS_DIR
from ..extended_lengyel_model.Lengyel_model_core import CzLINT_integrator, Mean_charge_interpolator
from ..mavrin_data import MavrinData

# Species used for the per-discharge seed impurities in the NF23 comparisons
NF23_SEED_SPECIES = (AtomicSpecies.Nitrogen, AtomicSpecies.Neon, AtomicSpecies.Argon)


def load_atomic_data(name: str = "radas", radas_dir: Path = RADAS_DIR) -> Any:
    """Load the atomic dataset ("radas" or "mavrin") used to build the interpolators."""
    if name == "radas":
        return read_atomic_data(radas_dir)[0]
    if name == "mavrin":
        return MavrinData()
    raise ValueError(f"Unknown atomic data name: {name} (expected 'radas' or 'mavrin')")


@dataclass(frozen=True)
class ImpurityInterpolators:
    """The species/weights of one impurity set, plus the two interpolators the models need."""

    species: xr.DataArray
    weights: xr.DataArray
    CzLINT: CzLINT_integrator
    mean_charge: Mean_charge_interpolator


def build_impurity_interpolators(
    species: Sequence[AtomicSpecies],
    weights: Sequence[float],
    atomic_data: Any,
    *,
    ne_tau: Unitfull = 0.5 * ureg.ms * ureg.n20,
    electron_density: Unitfull = 1.0 * ureg.n20,
    rtol_nearest: float = 1e-6,
) -> ImpurityInterpolators:
    """Build the CzLINT and mean-charge interpolators for one impurity set.

    Args:
        species: impurity species in the set.
        weights: relative concentration of each species.
        atomic_data: dataset from :func:`load_atomic_data`.
        ne_tau: non-coronal residence time parameter.
        electron_density: reference electron density for the atomic rates.
        rtol_nearest: relative tolerance for nearest-neighbour matching in the interpolators.
    """
    impurity_species, impurity_weights = setup_impurities(list(species), list(weights))
    impurity_kwargs = dict(ne_tau=ne_tau, electron_density=electron_density, rtol_nearest=rtol_nearest)

    return ImpurityInterpolators(
        species=impurity_species,
        weights=impurity_weights,
        CzLINT=CzLINT_integrator(impurity_species, impurity_weights, atomic_data, **impurity_kwargs),
        mean_charge=Mean_charge_interpolator(impurity_species, atomic_data, **impurity_kwargs),
    )


def model_interpolator_kwargs(seed: ImpurityInterpolators, fixed: ImpurityInterpolators) -> dict:
    """Map seed/fixed interpolator sets onto the four interpolator kwargs of the models."""
    return dict(
        CzLINT_for_seed_impurities=seed.CzLINT,
        mean_charge_for_seed_impurities=seed.mean_charge,
        CzLINT_for_fixed_impurities=fixed.CzLINT,
        mean_charge_for_fixed_impurities=fixed.mean_charge,
    )


def build_seed_impurities_from_dataset(
    ds: xr.Dataset,
    atomic_data: Any,
    *,
    error_factor: float = 0.0,
    zero_weight_fallback: bool = False,
    **interpolator_kwargs: Any,
) -> ImpurityInterpolators:
    """Build N/Ne/Ar seed interpolators with weights from the measured concentrations of one discharge.

    Args:
        ds: single-discharge dataset with ``{nitrogen,neon,argon}_concentration`` (and
            ``*_err`` variables when ``error_factor`` is non-zero).
        atomic_data: dataset from :func:`load_atomic_data`.
        error_factor: shift the weights by ``-error_factor * concentration_err`` (used to
            propagate the measured concentration uncertainty through the forward model).
        zero_weight_fallback: if all measured concentrations are zero, fall back to pure
            nitrogen weights ``[1, 0, 0]`` (used by the inverse model, where the weights
            only set the seed mix and the overall magnitude is solved for).
        **interpolator_kwargs: forwarded to :func:`build_impurity_interpolators`.
    """
    weights = np.array(
        [
            get_item(magnitude_in_units(ds[f"{species}_concentration"], ureg.dimensionless))
            for species in ("nitrogen", "neon", "argon")
        ]
    )

    if error_factor != 0.0:
        weight_errs = np.array(
            [
                get_item(magnitude_in_units(ds[f"{species}_concentration_err"], ureg.dimensionless))
                for species in ("nitrogen", "neon", "argon")
            ]
        )
        weights = weights - error_factor * weight_errs

    if zero_weight_fallback and np.allclose(weights, 0.0):
        weights = np.array([1.0, 0.0, 0.0])

    return build_impurity_interpolators(NF23_SEED_SPECIES, weights, atomic_data, **interpolator_kwargs)
