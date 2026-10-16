"""Patches for gaps in cfspopcon's case-file reader.

``process_input_dictionary`` registers plugins, then applies grids, named
options, paths, and units. Three of those steps reject inputs this package
already accepts:

- ``set_default_units`` builds ``Quantity(value, unit)``. A string such as
  ``"0.5 m"`` or ``"4/3"`` is stored as text, and later arithmetic fails.
- A number whose declared unit is ``null`` (a weight, a tolerance, an
  iteration count) raises ``RuntimeError``.
- ``process_grid_values`` builds the linspace before units are attached, so
  a grid bound written as ``"0.4 m"`` never gets parsed.

``patched_cfspopcon_input`` replaces those two functions and puts the
originals back when the context exits.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cfspopcon import input_file_handling
from cfspopcon.unit_handling import Quantity, default_unit, magnitude_in_units
from cfspopcon.unit_handling import set_default_units as _original_set_default_units

_TRUE = {"true"}
_FALSE = {"false"}


def _coerce_string(value: str) -> Any:
    """Turn a Click or YAML scalar into a bool or number when it is one."""
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    if any(marker in value.lower() for marker in (".", "e")):
        return number
    return int(number)


def _coerce(value: Any) -> Any:
    """Coerce strings inside a value, leaving other objects untouched."""
    if isinstance(value, list):
        return [_coerce(item) for item in value]
    if isinstance(value, str):
        return _coerce_string(value)
    return value


def _prepare(value: Any, unit: str | None) -> Any:
    """Coerce literals and parse quantity strings into magnitudes.

    A ``null`` unit means the value is not a quantity, so a number or a
    species name is returned as-is. ``""`` is dimensionless and still parsed.
    """
    value = _coerce(value)
    if unit is None:
        return value
    if isinstance(value, str):
        return magnitude_in_units(Quantity(value), unit)
    if isinstance(value, list):
        return [_prepare(item, unit) for item in value]
    return value


def _set_default_units(value: Any, key: str) -> Any:
    """Attach units without rejecting null-unit numbers or quantity strings."""
    unit = default_unit(key)
    value = _prepare(value, unit)
    if unit is None:
        return value
    return _original_set_default_units(value, key)


def _process_grid_values(repr_d: dict[str, Any]) -> None:
    """Parse grid bounds, then build the grid with cfspopcon's own function."""
    grid = repr_d.get("grid") or {}
    for key, spec in grid.items():
        if not isinstance(spec, dict):
            continue
        unit = default_unit(key)
        for bound in ("min", "max"):
            if bound in spec:
                spec[bound] = _prepare(spec[bound], unit)
        if "num" in spec:
            spec["num"] = _coerce(spec["num"])
    return _original_process_grid_values(repr_d)


_original_process_grid_values = input_file_handling.process_grid_values


@contextmanager
def patched_cfspopcon_input() -> Iterator[None]:
    """Patch cfspopcon's input reader and restore it on exit.

    Yields:
        None. The patch is active for the duration of the context.
    """
    saved_units = input_file_handling.set_default_units
    saved_grid = input_file_handling.process_grid_values
    input_file_handling.set_default_units = _set_default_units
    input_file_handling.process_grid_values = _process_grid_values
    try:
        yield
    finally:
        input_file_handling.set_default_units = saved_units
        input_file_handling.process_grid_values = saved_grid
