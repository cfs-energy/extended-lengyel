"""Steady-state 1D detachment models.

This is a cfspopcon plugin. Register it with ``cfspopcon.register_plugin("extended_lengyel")``; see
https://cfspopcon.readthedocs.io/en/latest/doc_sources/authoring_a_plugin.html for how plugins work.
"""

from .config import read_config, promote_to_coordinate

from . import adas_data
from . import initialize, postprocess
from . import kallenbach_model
from . import extended_lengyel_model
from . import spatial_lengyel_model
from . import mavrin_data

__all__ = [
    "adas_data",
    "extended_lengyel_model",
    "initialize",
    "kallenbach_model",
    "mavrin_data",
    "postprocess",
    "promote_to_coordinate",
    "read_config",
    "spatial_lengyel_model",
]
