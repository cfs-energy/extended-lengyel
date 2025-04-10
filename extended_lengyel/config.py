"""Read in a config.yml file and convert it into a form that can be used to run raddivmom algorithms."""

import yaml
from cfspopcon.unit_handling import Quantity, UndefinedUnitError, ureg
from cfspopcon import named_options
from typing import Any, Optional
import xarray as xr

from extended_lengyel.directories import notebook_dir


def convert_elements(config):
    """Recursive conversion for config elements."""
    for k, v in config.items():
        if isinstance(v, dict):
            config[k] = convert_elements(v)
        else:
            try:
                val = Quantity(v)

                if val.units == "":
                    val = val.to(ureg.dimensionless).magnitude

                    if val.size == 1:
                        config[k] = float(val)
                    else:
                        config[k] = val

                else:
                    config[k] = val

            except UndefinedUnitError:
                pass

        for enum in [named_options.AtomicSpecies]:
            try:
                config[k] = enum[v]
            except (KeyError, TypeError):
                pass

    return config


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
