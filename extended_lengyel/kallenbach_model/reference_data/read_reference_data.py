"""Methods for loading reference data."""

from pathlib import Path

import numpy as np
import yaml


def read_kallenbach_figure_4_reference():
    """Read in the values from Kallenbach et al., 2016, figure 4."""
    filepath = Path(__file__).parent / "kallenbach_figure_4.yaml"

    data = yaml.safe_load(filepath.read_text())

    for key, entry in data.items():
        values = np.array(entry)

        x_multiplier = 1e-2 if "left" in key else 1.0

        data[key] = {"x": values[:, 0] * x_multiplier, "y": values[:, 1]}

    return data
