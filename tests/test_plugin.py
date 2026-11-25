"""The package registers as a cfspopcon plugin."""

import subprocess
import sys
from pathlib import Path

import cfspopcon
import pytest
import yaml
from cfspopcon.unit_handling import default_unit

from extended_lengyel.cli import plugins_to_register, run_extended_lengyel

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


def test_variables_yaml_is_sorted_and_described():
    """Every entry of variables.yaml has a description, and the entries are in alphabetical order (ignoring case)."""
    variables = yaml.safe_load((REPO / "extended_lengyel" / "variables.yaml").read_text())
    assert list(variables) == sorted(variables, key=str.lower)
    assert [name for name, entry in variables.items() if not entry.get("description")] == []


@pytest.mark.parametrize(
    ("plugins", "expected"),
    [
        pytest.param(None, ["extended_lengyel"], id="absent-or-null"),
        pytest.param([], ["extended_lengyel"], id="empty"),
        pytest.param("", ["extended_lengyel"], id="empty-string"),
        pytest.param(["extended_lengyel"], ["extended_lengyel", "extended_lengyel"], id="listed"),
        pytest.param("other_plugin", ["extended_lengyel", "other_plugin"], id="bare-string"),
        pytest.param(["a", "b"], ["extended_lengyel", "a", "b"], id="others-keep-order"),
    ],
)
def test_plugins_to_register_always_starts_with_extended_lengyel(plugins, expected):
    """Whatever form the `plugins` section takes, extended_lengyel is registered first and the other names follow in order."""
    assert plugins_to_register(plugins) == expected


def test_cli_registers_extended_lengyel_in_a_fresh_interpreter(tmp_path):
    """An input file with no `plugins` section runs end to end, because the command line interface registers the plugin itself."""
    config = yaml.safe_load((REPO / "input.yml").read_text())
    del config["plugins"]
    config_file = tmp_path / "input.yml"
    config_file.write_text(yaml.safe_dump(config))
    output_file = tmp_path / "output.yml"
    code = f"from extended_lengyel.cli import run_extended_lengyel; run_extended_lengyel({str(config_file)!r}, {str(output_file)!r})"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "separatrix_electron_temp" in yaml.safe_load(output_file.read_text())


def test_cli_names_the_plugins_section_when_a_plugin_is_misspelt(tmp_path):
    """A plugin that cannot be imported is reported as a problem with the input file's `plugins` section."""
    config = yaml.safe_load((REPO / "input.yml").read_text())
    config["plugins"] = ["extended_lengyel_does_not_exist"]
    config_file = tmp_path / "input.yml"
    config_file.write_text(yaml.safe_dump(config))

    with pytest.raises(ModuleNotFoundError, match="'plugins' section"):
        run_extended_lengyel(config_file, tmp_path / "output.yml")
