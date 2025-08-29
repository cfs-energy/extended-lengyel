#!.venv/bin/python
# Run this script from the repository directory.
"""CLI for extended-lengyel."""

import click
import warnings
from pathlib import Path
import xarray as xr
import yaml

from . import config
from .xr_helpers import item
import cfspopcon

from cfspopcon.unit_handling import UnitStrippedWarning

@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(exists=False))
@click.option("--debug", is_flag=True, help="Enable the ipdb exception catcher. (Development helper)", hidden=True)
@click.option("--dict", "-d", "kwargs", type=(str, str), multiple=True, help="Command-line arguments, takes precedence over config.")
def run_extended_lengyel_cli(config_file: str, output_file: str, kwargs: tuple[tuple[str, str]], debug=False):
    """Run the extended Lengyel model from the command line, using the Click command line."""
    cli_args: dict[str, str] = dict(kwargs)

    if debug:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UnitStrippedWarning)
            try:
                # if ipdb is installed we use it to catch exceptions during development
                from ipdb import launch_ipdb_on_exception  # noqa:PLC0415 type:ignore[import-untyped]

                with launch_ipdb_on_exception():
                    run_extended_lengyel(config_file, output_file, cli_args)
            except ModuleNotFoundError:
                run_extended_lengyel(config_file, output_file, cli_args)
    else:
        run_extended_lengyel(config_file, output_file, cli_args)

def run_extended_lengyel(config_file, output_file, cli_args=None) -> None:
    """Run the extended Lengyel model as a calculator."""
    if cli_args is None:
        cli_args = {}
    config_file = Path(config_file).absolute()
    assert config_file.exists(), f"{config_file} not found."
    assert config_file.suffix == ".yml", f"{config_file} is not a YAML file."

    output_file = Path(output_file).absolute()
    if output_file.exists():
        click.confirm(f"{output_file} already exists. Overwrite?", abort=True)
    assert output_file.suffix == ".yml", f"{output_file} is not a YAML file."

    algorithms = config.read_config_from_yaml(config_file)["algorithms"]

    algorithm = cfspopcon.CompositeAlgorithm.from_list(algorithms)

    data_vars = config.read_config(
        elements          = ["input"],
        filepath          = config_file,
        keys              = algorithm.input_keys,
        allowed_missing   = algorithm.default_keys,
        overrides         = cli_args,
        warn_if_unused    = True,
        convert_overrides = True,
    )

    ds = xr.Dataset(data_vars=data_vars)
    algorithm.validate_inputs(ds)
    ds = algorithm.update_dataset(ds)

    write_output_file(output_file, ds)

    print("Extended lengyel model ran successfully.")

def write_output_file(filepath: Path, ds: xr.Dataset):
    """Write the results from the extended Lengyel model to a YAML file."""
    from cfspopcon.file_io import sanitize_variable, ignored_keys # noqa:PLC0415
    ignored_keys += [
        "seed_impurity_species",
        "seed_impurity_weights",
        "CzLINT_for_seed_impurities",
        "mean_charge_for_seed_impurities",
        "fixed_impurity_species",
        "fixed_impurity_weights",
        "CzLINT_for_fixed_impurities",
        "mean_charge_for_fixed_impurities",
        "dim_species",
    ]
    output_dict = dict()

    for key in ds.keys():
        if key in ignored_keys:
            continue
        output_dict[key] = sanitize_variable(ds[key], key)

    for key in ds.coords:
        if key in ignored_keys:
            continue
        output_dict[key] = sanitize_variable(ds[key], key)

    impurity_fraction = cfspopcon.unit_handling.magnitude_in_units(ds["impurity_fraction"], "")
    seed_impurity_concentration = impurity_fraction * ds["seed_impurity_weights"].dropna(dim="dim_species")
    fixed_impurity_concentration = ds["fixed_impurity_weights"].dropna(dim="dim_species")

    output_impurity_fraction = dict(seed_impurity=dict(), fixed_impurity=dict())
    for cz in seed_impurity_concentration:
        output_impurity_fraction["seed_impurity"][item(cz.dim_species).name] = item(cz)
    for cz in fixed_impurity_concentration:
        output_impurity_fraction["fixed_impurity"][item(cz.dim_species).name] = item(cz)

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
