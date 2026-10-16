"""Session fixtures."""

import pytest

import cfspopcon


@pytest.fixture(scope="session", autouse=True)
def registered_plugin_algorithms():
    """Register extended-lengyel once for the whole suite."""
    return cfspopcon.register_plugin("extended_lengyel")
