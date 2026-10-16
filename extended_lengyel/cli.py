#!.venv/bin/python
# Run this script from the repository directory.
"""CLI for extended-lengyel."""

import warnings
from pathlib import Path
from typing import Any

import click
import xarray as xr
import yaml
from cfspopcon.file_io import ignored_keys, sanitize_variable
from cfspopcon.helpers import get_item
from cfspopcon.input_file_handling import process_input_dictionary
from cfspopcon.named_options import AtomicSpecies
from cfspopcon.unit_handling import UnitStrippedWarning, magnitude_in_units

from .config import setup_impurities
from .input_patches import patched_cfspopcon_input

# Not written to the YAML result. A local copy: cfspopcon.file_io.ignored_keys is a shared list.
_OUTPUT_SKIP = {
    *ignored_keys,
    "seed_impurity_species",
    "seed_impurity_weights",
    "CzLINT_for_seed_impurities",
    "mean_charge_for_seed_impurities",
    "fixed_impurity_species",
    "fixed_impurity_weights",
    "CzLINT_for_fixed_impurities",
    "mean_charge_for_fixed_impurities",
    "dim_species",
}


def _as_atomic_species(value: Any) -> list[Any]:
    """Turn species names into ``AtomicSpecies`` values.

    Args:
        value: a species name, a list of names, or an empty value.

    Returns:
        A list of species, with strings capitalized and looked up on ``AtomicSpecies``.
    """
    if not value:
        return []
    entries = value if isinstance(value, list | tuple) else [value]
    return [AtomicSpecies[entry.capitalize()] for entry in entries]


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(exists=False))
@click.option("--debug", is_flag=True, help="Enable the ipdb exception catcher. (Development helper)", hidden=True)
@click.option("--dict", "-d", "kwargs", type=(str, str), multiple=True, help="Command-line arguments, takes precedence over config.")
def run_extended_lengyel_cli(config_file: str, output_file: str, kwargs: tuple[tuple[str, str], ...], debug: bool = False) -> None:
    """Run the extended Lengyel model from the command line, using the Click command line.

    Args:
        config_file: path to the input YAML file.
        output_file: path to the YAML file to write.
        kwargs: ``-d key value`` overrides. Values are strings.
        debug: catch exceptions in ipdb when it is installed.
    """
    cli_args: dict[str, str] = dict(kwargs)

    if debug:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UnitStrippedWarning)
            try:
                # if ipdb is installed we use it to catch exceptions during development
                from ipdb import launch_ipdb_on_exception

                with launch_ipdb_on_exception():
                    run_extended_lengyel(config_file, output_file, cli_args)
            except ModuleNotFoundError:
                run_extended_lengyel(config_file, output_file, cli_args)
    else:
        run_extended_lengyel(config_file, output_file, cli_args)


def run_extended_lengyel(config_file: str | Path, output_file: str | Path, cli_args: dict[str, str] | None = None) -> None:
    """Run the extended Lengyel model from a YAML input file and write a YAML result.

    Variables may sit at the top level, as in a cfspopcon case file, or under an ``input``
    key. A missing ``plugins`` entry registers this package. ``cfspopcon`` then registers
    those plugins and applies grids, named options, paths, and units.

    Args:
        config_file: path to the input YAML file.
        output_file: path to the YAML file to write.
        cli_args: input overrides. Values are strings, as Click provides them.
    """
    if cli_args is None:
        cli_args = {}
    config_file = Path(config_file).absolute()
    assert config_file.exists(), f"{config_file} not found."
    assert config_file.suffix == ".yml", f"{config_file} is not a YAML file."

    output_file = Path(output_file).absolute()
    if output_file.exists():
        click.confirm(f"{output_file} already exists. Overwrite?", abort=True)
    assert output_file.suffix == ".yml", f"{output_file} is not a YAML file."

    raw = yaml.safe_load(config_file.read_text())
    inputs = raw.pop("input", {}) or {}
    inputs.update(cli_args)
    raw.update(inputs)
    raw.setdefault("plugins", ["extended_lengyel"])
    with patched_cfspopcon_input():
        params, algorithm, _points, _plots = process_input_dictionary(raw, config_file.parent)

    seed_species, seed_weights = setup_impurities(
        _as_atomic_species(params.get("seed_impurity_species", [])),
        params.get("seed_impurity_weights", []),
    )
    fixed_species, fixed_weights = setup_impurities(
        _as_atomic_species(params.get("fixed_impurity_species", [])),
        params.get("fixed_impurity_weights", []),
    )
    params["seed_impurity_species"] = seed_species
    params["seed_impurity_weights"] = seed_weights
    params["fixed_impurity_species"] = fixed_species
    params["fixed_impurity_weights"] = fixed_weights

    unused = set(params) - set(algorithm.input_keys)
    if unused:
        warnings.warn(f"Not all keys in config were used. Unused keys were {unused}.")
    missing = [key for key in algorithm.required_input_keys if key not in params]
    if missing:
        raise KeyError(f"Need keys {', '.join(missing)} but they are not in the selected config.")
    selected = {key: params[key] for key in algorithm.input_keys if key in params}

    ds = xr.Dataset(data_vars=selected)
    algorithm.validate_inputs(ds)
    ds = algorithm.update_dataset(ds)

    write_output_file(output_file, ds)

    print("Extended lengyel model ran successfully.")


def write_output_file(filepath: Path, ds: xr.Dataset) -> None:
    """Write the results from the extended Lengyel model to a YAML file.

    Args:
        filepath: YAML path to write.
        ds: dataset of inputs and algorithm outputs.
    """
    output_dict = {}

    for key in ds.keys():
        if key in _OUTPUT_SKIP:
            continue
        output_dict[key] = sanitize_variable(ds[key], key)

    for key in ds.coords:
        if key in _OUTPUT_SKIP:
            continue
        output_dict[key] = sanitize_variable(ds[key], key)

    impurity_fraction = magnitude_in_units(ds["impurity_fraction"], "")
    seed_impurity_concentration = impurity_fraction * ds["seed_impurity_weights"].dropna(dim="dim_species")
    fixed_impurity_concentration = ds["fixed_impurity_weights"].dropna(dim="dim_species")

    output_impurity_fraction: dict[str, dict[str, Any]] = {"seed_impurity": {}, "fixed_impurity": {}}
    for cz in seed_impurity_concentration:
        output_impurity_fraction["seed_impurity"][get_item(cz.dim_species).name] = get_item(cz)
    for cz in fixed_impurity_concentration:
        output_impurity_fraction["fixed_impurity"][get_item(cz.dim_species).name] = get_item(cz)

    for key, val in output_dict.items():
        units = getattr(val, "units", None)
        v = val.values.tolist()
        if units is not None:
            output_dict[key] = f"{v} {units}"
        else:
            output_dict[key] = v

    output_dict["impurity_fraction"] = output_impurity_fraction

    with open(filepath, "w") as f:
        f.write(yaml.dump(output_dict))


if __name__ == "__main__":
    run_extended_lengyel_cli()
