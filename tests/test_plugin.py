"""The package registers as a cfspopcon plugin."""

import cfspopcon
from cfspopcon.unit_handling import default_unit


def test_plugin_registers_its_algorithms(registered_plugin_algorithms):
    """Registration adds this package's algorithms, including a composite of them."""
    assert "run_basic_lengyel_model" in registered_plugin_algorithms
    assert "kallenbach_model" in registered_plugin_algorithms
    assert "run_forward_extended_lengyel_model" in registered_plugin_algorithms
    assert "run_forward_extended_lengyel_model_pdiv" in registered_plugin_algorithms
    assert "run_inverse_extended_lengyel_model_with_pdiv" in registered_plugin_algorithms
    assert cfspopcon.registry["kallenbach_model"].name == "kallenbach_model"


def test_plugin_algorithms_have_default_units(registered_plugin_algorithms):
    """Every input and output of a plugin algorithm has an entry in variables.yaml or cfspopcon's."""
    missing = []
    for name in registered_plugin_algorithms:
        algorithm = cfspopcon.registry[name]
        for key in (*algorithm.input_keys, *algorithm.return_keys):
            try:
                default_unit(key)
            except KeyError:
                missing.append(f"{name}: {key}")
    assert missing == []
