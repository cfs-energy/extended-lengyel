"""Simple functions for use with xarray."""

import xarray as xr

def item(val):
    """Extract items from xr.DataArray wrappers."""
    if isinstance(val, xr.DataArray):
        return val.item()
    else:
        return val

def values(val):
    """Extract values from xr.DataArray wrappers."""
    if isinstance(val, xr.DataArray):
        return val.values
    else:
        return val
