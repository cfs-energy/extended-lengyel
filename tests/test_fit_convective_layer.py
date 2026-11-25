"""The convective-layer fit command line interface."""

import math

import pytest
import xarray as xr
import yaml
from click.testing import CliRunner

from extended_lengyel import fit_convective_layer
from extended_lengyel.directories import notebook_dir
from extended_lengyel.extended_lengyel_model.convective_loss_fits import CURVE_FIT_FILE, FIT_NAMES, fit_coefficients_key
from extended_lengyel.fit_convective_layer import fit_convective_layer_cli, write_fit_coefficients

COEFFICIENTS = ["amplitude", "width", "shape"]


@pytest.fixture(scope="module")
def refitted_coefficients(tmp_path_factory):
    """Refit the curves from a fresh scan of the Kallenbach model, writing them to a temporary file."""
    output_file = tmp_path_factory.mktemp("fit") / "curve_fit.yml"
    result = CliRunner().invoke(
        fit_convective_layer_cli,
        ["--config-file", str(notebook_dir / "config.yml"), "--output-file", str(output_file), "--info"],
    )
    assert result.exit_code == 0, result.output
    assert "Covariance matrix condition number" in result.output
    return yaml.safe_load(output_file.read_text())


def test_fit_cli_writes_fit_coefficients_for_each_loss(refitted_coefficients):
    """The command writes amplitude, width and shape, with their uncertainties, for each of the three fitted losses."""
    assert set(refitted_coefficients) == {fit_coefficients_key(name) for name in FIT_NAMES}
    for coefficients in refitted_coefficients.values():
        for name in COEFFICIENTS:
            assert math.isfinite(coefficients[name])
            assert math.isfinite(coefficients[f"{name}_uncertainty"])
            assert coefficients[f"{name}_uncertainty"] > 0


def test_fit_cli_reproduces_the_shipped_coefficients(refitted_coefficients):
    """A fresh fit agrees with the curve_fit.yml shipped with the package, so the fit and the Kallenbach model have not drifted."""
    shipped = yaml.safe_load(CURVE_FIT_FILE.read_text())
    for group, coefficients in shipped.items():
        for name in COEFFICIENTS:
            assert refitted_coefficients[group][name] == pytest.approx(coefficients[name], rel=1e-3), f"{group}.{name}"


def test_fit_cli_without_default_config_asks_for_one(tmp_path, monkeypatch):
    """Outside a source checkout the default config is missing, so the command says to pass --config-file."""
    monkeypatch.setattr(fit_convective_layer, "DEFAULT_CONFIG_FILE", tmp_path / "missing_config.yml")

    result = CliRunner().invoke(fit_convective_layer_cli, ["--output-file", str(tmp_path / "curve_fit.yml")])

    assert result.exit_code != 0
    assert "Pass --config-file" in result.output
    assert not (tmp_path / "curve_fit.yml").exists()


def test_fit_cli_refuses_to_write_non_finite_coefficients(tmp_path, monkeypatch):
    """A fit which fails to converge gives non-finite uncertainties; the command says so and writes nothing."""
    scan = xr.Dataset({fit_convective_layer.FITTED_VARIABLES["fmom"]: ("x", [0.1, 0.2])})
    coefficients = {fit_coefficients_key(name): dict(amplitude=1.0, width=1.0, shape=float("inf")) for name in FIT_NAMES}
    monkeypatch.setattr(fit_convective_layer, "build_reference_scan_dataset", lambda config_file: scan)
    monkeypatch.setattr(fit_convective_layer, "determine_fmomc_and_fpowc_fit_functions", lambda ds, info=False: (coefficients, {}))
    output_file = tmp_path / "curve_fit.yml"

    result = CliRunner().invoke(
        fit_convective_layer_cli, ["--config-file", str(notebook_dir / "config.yml"), "--output-file", str(output_file)]
    )

    assert result.exit_code != 0
    assert "non-finite values for fmom_coeffs.shape" in result.output
    assert not output_file.exists()


def test_fit_cli_reports_scan_points_where_the_model_failed(tmp_path, monkeypatch):
    """Scan points where the Kallenbach model failed are left out of the fit, and the command says how many."""
    scan = xr.Dataset({fit_convective_layer.FITTED_VARIABLES["fmom"]: ("x", [0.1, float("nan"), 0.3, float("nan")])})
    coefficients = {fit_coefficients_key(name): dict(amplitude=1.0, width=1.0, shape=1.0) for name in FIT_NAMES}
    monkeypatch.setattr(fit_convective_layer, "build_reference_scan_dataset", lambda config_file: scan)
    monkeypatch.setattr(fit_convective_layer, "determine_fmomc_and_fpowc_fit_functions", lambda ds, info=False: (coefficients, {}))

    result = CliRunner().invoke(
        fit_convective_layer_cli,
        ["--config-file", str(notebook_dir / "config.yml"), "--output-file", str(tmp_path / "curve_fit.yml")],
    )

    assert result.exit_code == 0, result.output
    assert "failed at 2 of 4 scan points" in result.output


def test_write_fit_coefficients_keeps_the_existing_file_if_writing_fails(tmp_path, monkeypatch):
    """An error while writing leaves the file being replaced as it was, and no temporary file behind."""
    output_file = tmp_path / "curve_fit.yml"
    output_file.write_text("original: 1\n")

    def fail(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fit_convective_layer.yaml, "dump", fail)

    with pytest.raises(RuntimeError, match="disk full"):
        write_fit_coefficients({"new": 2}, output_file)

    assert output_file.read_text() == "original: 1\n"
    assert list(tmp_path.iterdir()) == [output_file]


def test_write_fit_coefficients_keeps_file_permissions(tmp_path):
    """Replacing a file keeps its permissions, rather than the private ones of the temporary file."""
    output_file = tmp_path / "curve_fit.yml"
    output_file.write_text("original: 1\n")
    output_file.chmod(0o644)

    write_fit_coefficients({"new": 2}, output_file)

    assert yaml.safe_load(output_file.read_text()) == {"new": 2}
    assert output_file.stat().st_mode & 0o777 == 0o644
