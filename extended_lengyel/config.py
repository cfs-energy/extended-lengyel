"""Read in a config.yml file and convert it into a form that can be used to run raddivmom algorithms."""

import yaml
from cfspopcon.unit_handling import Quantity, UndefinedUnitError, ureg
from cfspopcon.named_options import AtomicSpecies
from fractions import Fraction
from typing import Any, Optional, Callable
import xarray as xr

from extended_lengyel.directories import notebook_dir


def test_convert(element: str, conversion: Callable[[str], Any]) -> Any | None:
    """Use the conversion routine to convert a string to another type. If this fails, return None."""
    try:
        return conversion(element)
    except (ValueError, UndefinedUnitError):
        return None

def convert_elements(element):
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
    
    raise NotImplementedError(f"Cannot handle {element} of type {type(element)}")


def read_config(
    elements: Optional[list[str]] = None,
    filepath = notebook_dir / "config.yml",
    overrides: Optional[dict[str, Any]] = None,
    keys: Optional[list[str]] = None,
    allowed_missing: Optional[list[str]] = None,
):
    """Read configuration file and return as a dictionary.

    N.b. if multiple elements contain the same config keys, the key from the last element containing the key is used.
    """
    if elements is None:
        elements = []
    if allowed_missing is None:
        allowed_missing = []
    if overrides is None:
        overrides = {}

    with open(filepath) as file:
        config = yaml.safe_load(file)

    config = convert_elements(config)

    flattened_config = {}
    for element in elements:
        flattened_config.update(config[element])

    for k, v in overrides.items():
        flattened_config[k] = v

    if keys is None:
        return flattened_config
    else:
        selected_config = {}
        for k in keys:
            if k in flattened_config.keys():
                selected_config[k] = flattened_config[k]
            elif k in allowed_missing:
                continue
            else:
                raise KeyError(
                    f"Need key {k} but this is not in the selected config\nelements = {", ".join(elements)})\nkeys = {", ".join(flattened_config.keys())}"
                )

        return selected_config


def promote_to_coordinate(array, units, dims):
    """Convert an array of values to a coordinate for performing scans over."""
    return xr.DataArray(array * units, coords={f"dim_{dims}": array})
