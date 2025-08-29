"""Steady-state 1D detachment models."""

from pathlib import Path

import cfspopcon
import yaml
from cfspopcon.unit_handling.default_units import extend_default_units_map
from .config import read_config, promote_to_coordinate

from . import adas_data
from . import initialize, postprocess
from . import kallenbach_model
from . import extended_lengyel_model
from . import spatial_lengyel_model


def extend_units_dictionary():
    with open(Path(__file__).parent / "extended_units.yaml") as filepath:
        units_dictionary = yaml.safe_load(filepath)

    variable_keys = set(cfspopcon.unit_handling.default_units._DEFAULT_UNITS.keys())
    if variable_keys & set(units_dictionary.keys()):
        duplicated_list = "\n".join(variable_keys & set(units_dictionary.keys()))
        raise AssertionError(f"The following keys have been defined multiple times:\n{duplicated_list}")

    extend_default_units_map(units_dictionary)


extend_units_dictionary()


def check_extended_units_dictionary():
    algorithm_keys = set()
    for alg in cfspopcon.Algorithm.instances.values():
        algorithm_keys.update(alg.input_keys)
        algorithm_keys.update(alg.return_keys)

    variable_keys = set(cfspopcon.unit_handling.default_units._DEFAULT_UNITS.keys())

    if variable_keys - algorithm_keys:
        unused_list = "\n".join(variable_keys - algorithm_keys)
        print(f"The following keys are not used by any algorithm:\n{unused_list}")

    if algorithm_keys - variable_keys:
        missing_list = "\n".join(algorithm_keys - variable_keys)
        raise AssertionError(f"The following keys do not have defined default units:\n{missing_list}")


check_extended_units_dictionary()

__all__ = [
    "adas_data",
    "extended_lengyel_model",
    "initialize",
    "kallenbach_model",
    "postprocess",
    "promote_to_coordinate",
    "read_config",
    "spatial_lengyel_model",
]
