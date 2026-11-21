"""The package registers as a cfspopcon plugin."""

import subprocess
import sys
from pathlib import Path

import cfspopcon
import pytest
import yaml
from cfspopcon.unit_handling import default_unit

from extended_lengyel.cli import run_extended_lengyel

REPO = Path(__file__).parents[1]


def test_importing_registers_nothing():
    """A fresh interpreter that only imports the package sees none of its algorithms."""
    code = "import extended_lengyel, cfspopcon; print('run_basic_lengyel_model' in cfspopcon.registry)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_plugin_registers_its_algorithms(registered_plugin_algorithms):
    """Registration adds this package's algorithms, including a composite of them."""
    assert "run_basic_lengyel_model" in registered_plugin_algorithms
    assert "kallenbach_model" in registered_plugin_algorithms
    assert "run_extended_lengyel_model_with_S_Zeff_and_alphat_correction" in registered_plugin_algorithms
    assert "run_forward_extended_lengyel_model" in registered_plugin_algorithms
    assert "run_forward_extended_lengyel_model_with_pdiv" in registered_plugin_algorithms
    assert "run_inverse_extended_lengyel_model" in registered_plugin_algorithms
    assert "run_inverse_extended_lengyel_model_with_pdiv" in registered_plugin_algorithms
    assert cfspopcon.registry["kallenbach_model"].name == "kallenbach_model"


def test_registering_twice_changes_nothing():
    """A second registration returns no new names and leaves the registry as it was."""
    before = set(cfspopcon.registry)
    assert cfspopcon.register_plugin("extended_lengyel") == []
    assert set(cfspopcon.registry) == before


def test_unknown_plugin_raises_and_registers_nothing():
    """A misspelt plugin name raises and leaves the registry untouched."""
    before = set(cfspopcon.registry)
    with pytest.raises(ModuleNotFoundError):
        cfspopcon.register_plugin("extended_lengyel_does_not_exist")
    assert set(cfspopcon.registry) == before


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


def test_variables_yaml_has_no_unused_entries(registered_plugin_algorithms):
    """Every variable declared in variables.yaml is used by some registered algorithm."""
    declared = set(yaml.safe_load((REPO / "extended_lengyel" / "variables.yaml").read_text()))
    used = set()
    for name in cfspopcon.registry:
        used.update(cfspopcon.registry[name].input_keys, cfspopcon.registry[name].return_keys)
    assert declared - used == set()


def test_input_yml_example_runs(tmp_path):
    """The shipped input.yml lists the plugin and runs through the command line interface."""
    config = yaml.safe_load((REPO / "input.yml").read_text())
    assert config["plugins"] == ["extended_lengyel"]
    output_file = tmp_path / "output.yml"
    run_extended_lengyel(REPO / "input.yml", output_file)
    assert "impurity_fraction" in yaml.safe_load(output_file.read_text())


def test_cli_registers_extended_lengyel_without_a_plugins_section(tmp_path):
    """A case file with no `plugins` section still finds the extended-lengyel algorithms (fresh interpreter)."""
    config = yaml.safe_load((REPO / "input.yml").read_text())
    del config["plugins"]
    config_file = tmp_path / "input.yml"
    config_file.write_text(yaml.safe_dump(config))
    code = f"from extended_lengyel.cli import run_extended_lengyel; run_extended_lengyel({str(config_file)!r}, {str(tmp_path / 'output.yml')!r})"
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "impurity_fraction" in yaml.safe_load((tmp_path / "output.yml").read_text())
