"""Session fixtures."""

import pytest

import cfspopcon


@pytest.fixture(scope="session", autouse=True)
def registered_plugin_algorithms():
    """Register extended-lengyel once for the whole suite, returning the names it registered."""
    names = cfspopcon.register_plugin("extended_lengyel")
    # An empty list means something registered the plugin before this fixture, and every test using it would pass vacuously.
    assert names, "extended_lengyel was already registered before the test session set it up."
    return names
