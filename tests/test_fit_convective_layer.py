"""The convective-layer fit command line interface."""

import math

import yaml
from click.testing import CliRunner

from extended_lengyel import fit_convective_layer
from extended_lengyel.directories import notebook_dir
from extended_lengyel.fit_convective_layer import fit_convective_layer_cli

COEFFICIENTS = ["amplitude", "width", "shape"]


def test_fit_cli_writes_fit_coefficients_for_each_loss(tmp_path):
    """The command writes amplitude, width and shape, with their uncertainties, for each of the three fitted losses."""
    output_file = tmp_path / "curve_fit.yml"

    result = CliRunner().invoke(
        fit_convective_layer_cli,
        ["--config-file", str(notebook_dir / "config.yml"), "--output-file", str(output_file), "--info"],
    )

    assert result.exit_code == 0, result.output
    assert "Covariance matrix condition number" in result.output
    fitted = yaml.safe_load(output_file.read_text())
    assert set(fitted) == {"fmom_coeffs", "fpow_coeffs", "fdens_coeffs"}
    for coefficients in fitted.values():
        for name in COEFFICIENTS:
            assert math.isfinite(coefficients[name])
            assert math.isfinite(coefficients[f"{name}_uncertainty"])
            assert coefficients[f"{name}_uncertainty"] > 0


def test_fit_cli_without_default_config_asks_for_one(tmp_path, monkeypatch):
    """Outside a source checkout the default config is missing, so the command says to pass --config-file."""
    monkeypatch.setattr(fit_convective_layer, "DEFAULT_CONFIG_FILE", tmp_path / "missing_config.yml")

    result = CliRunner().invoke(fit_convective_layer_cli, ["--output-file", str(tmp_path / "curve_fit.yml")])

    assert result.exit_code != 0
    assert "Pass --config-file" in result.output
    assert not (tmp_path / "curve_fit.yml").exists()
