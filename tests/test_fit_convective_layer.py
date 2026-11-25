"""The convective-layer fit command line interface writes curve_fit.yml-style coefficients."""

import numpy as np
import yaml
from click.testing import CliRunner

from extended_lengyel.directories import module_directory
from extended_lengyel.fit_convective_layer import fit_convective_layer_cli


def test_fit_cli_writes_coefficients_and_leaves_shipped_file_alone(tmp_path):
    """The fit reproduces the shipped coefficients, and only writes where asked."""
    shipped_file = module_directory / "curve_fit.yml"
    shipped_before = shipped_file.read_bytes()
    output_file = tmp_path / "curve_fit.yml"

    result = CliRunner().invoke(fit_convective_layer_cli, ["--output-file", str(output_file)])

    assert result.exit_code == 0, result.output
    assert shipped_file.read_bytes() == shipped_before

    fitted = yaml.safe_load(output_file.read_text())
    shipped = yaml.safe_load(shipped_before)
    assert fitted.keys() == shipped.keys()
    for name, coefficients in shipped.items():
        assert fitted[name].keys() == coefficients.keys()
        for coefficient in ["amplitude", "width", "shape"]:
            assert np.isclose(fitted[name][coefficient], coefficients[coefficient], rtol=1e-2)
