"""Steady-state 1D detachment models.

A cfspopcon plugin. Importing this package registers nothing; call
``cfspopcon.register_plugin("extended_lengyel")`` or list ``extended_lengyel``
in an input file's ``plugins`` section.
"""

from .config import read_config, promote_to_coordinate

from . import adas_data
from . import initialize, postprocess
from . import kallenbach_model
from . import extended_lengyel_model
from . import spatial_lengyel_model
from . import iterative_solver
from . import mavrin_data

__all__ = [
    "adas_data",
    "extended_lengyel_model",
    "initialize",
    "iterative_solver",
    "kallenbach_model",
    "mavrin_data",
    "postprocess",
    "promote_to_coordinate",
    "read_config",
    "spatial_lengyel_model",
]
