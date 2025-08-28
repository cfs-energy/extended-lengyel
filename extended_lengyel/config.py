"""Read in a config.yml file and convert it into a form that can be used to run raddivmom algorithms."""

import yaml
from pathlib import Path
from cfspopcon.unit_handling import Quantity, UndefinedUnitError
from cfspopcon.named_options import AtomicSpecies
from fractions import Fraction
from typing import Any, Optional, Callable
import xarray as xr
import numpy as np
import warnings


def test_convert(element: str, conversion: Callable[[str], Any]) -> Any | None:
    """Use the conversion routine to convert a string to another type. If this fails, return None."""
    try:
        return conversion(element)
    except (ValueError, UndefinedUnitError, KeyError):
        return None

def convert_elements(element): # noqa:PLR0911
    """Read the elements of the configuration and convert them to their underlying types."""
    if isinstance(element, dict):
        return {k: convert_elements(v) for k, v in element.items()}
    elif isinstance(element, list):
        return [convert_elements(v) for v in element]
    elif isinstance(element, (float, int)):
        return element
    elif isinstance(element, str):
        if (val:=test_convert(element, float)) is not None:
            return val
        if (val:=test_convert(element, lambda s: float(Fraction(s)))) is not None:
            return val
        elif (val:=test_convert(element, Quantity)) is not None:
            return val
        elif (val:=test_convert(element, lambda s: AtomicSpecies.__getitem__(str.capitalize(s)))) is not None:
            return val
        elif element.startswith("PATH:"):
            return Path(element.lstrip("PATH:")).absolute()

    raise NotImplementedError(f"Cannot handle {element} of type {type(element)}")

def read_config_from_yaml(filepath: Path):
    """Read a configuration YAML file."""
    with open(filepath) as file:
        return yaml.safe_load(file)

def read_config( # noqa:PLR0912
    filepath: Path,
    elements: Optional[list[str]] = None,
    keys: Optional[list[str]] = None,
    allowed_missing: Optional[list[str]] = None,
    warn_if_unused: bool = False,
    overrides: Optional[dict[str, Any]] = None,
    convert_overrides: bool = False,
):
    """Read configuration file and return as a dictionary.

    N.b. if multiple elements contain the same config keys, the key from the last element containing the key is used.
    """
    if overrides is None:
        overrides = {}
    if allowed_missing is None:
        allowed_missing = []
    if elements is None:
        elements = []
    if allowed_missing is None:
        allowed_missing = []
    if overrides is None:
        overrides = {}

    config = read_config_from_yaml(filepath)

    flattened_config = {}
    for element in elements:
        flattened_config.update(convert_elements(config[element]))

    for k, v in overrides.items():
        if convert_overrides:
            flattened_config[k] = convert_elements(v)
        else:
            flattened_config[k] = v

    flattened_config["seed_impurity_species"], flattened_config["seed_impurity_weights"] = \
        setup_impurities(flattened_config.get("seed_impurity_species", []), flattened_config.get("seed_impurity_weights", []))
    flattened_config["fixed_impurity_species"], flattened_config["fixed_impurity_weights"] = \
        setup_impurities(flattened_config.get("fixed_impurity_species", []), flattened_config.get("fixed_impurity_weights", []))

    if keys is None:
        return flattened_config
    else:
        if warn_if_unused:
            if len(unused_keys:=set(flattened_config.keys()) - set(keys)):
                warnings.warn(f"Not all keys in config were used. Unused keys were {unused_keys}.")
        selected_config = {}
        for k in keys:
            if k in flattened_config.keys():
                selected_config[k] = flattened_config[k]
            elif k in allowed_missing:
                continue
            else:
                raise KeyError(
                    f"Need key {k} but this is not in the selected config\nelements = {', '.join(elements)})\nkeys = {', '.join(flattened_config.keys())}"
                )

        return selected_config

def setup_impurities(impurity_species, impurity_weights) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert linked lists for seed impurity species into xarrays."""
    coords = {"dim_species": np.atleast_1d(impurity_species)}
    impurity_weights = xr.DataArray(np.atleast_1d(impurity_weights), coords=coords)
    impurity_species = xr.DataArray(np.atleast_1d(impurity_species), coords=coords)

    return impurity_species, impurity_weights

def promote_to_coordinate(array, units, dims):
    """Convert an array of values to a coordinate for performing scans over."""
    return xr.DataArray(array * units, coords={f"dim_{dims}": array})
