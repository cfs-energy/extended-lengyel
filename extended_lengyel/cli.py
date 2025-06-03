#!.venv/bin/python
# Run this script from the repository directory.
"""CLI for extended-lengyel."""

import click
import warnings
from pathlib import Path
import yaml

from . import config
from . import extended_lengyel_model
import cfspopcon

from cfspopcon.unit_handling import UnitStrippedWarning

@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable the ipdb exception catcher. (Development helper)", hidden=True)
@click.option("--dict", "-d", "kwargs", type=(str, str), multiple=True, help="Command-line arguments, takes precedence over config.")
def run_extended_lengyel_cli(config_file: str, kwargs: tuple[tuple[str, str]], debug=False):
    """Run the extended Lengyel model from the command line, using the Click command line."""
    cli_args: dict[str, str] = dict(kwargs)

    if debug:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UnitStrippedWarning)
            try:
                # if ipdb is installed we use it to catch exceptions during development
                from ipdb import launch_ipdb_on_exception  # type:ignore[import-untyped]

                with launch_ipdb_on_exception():
                    run_extended_lengyel(config_file, cli_args)
            except ModuleNotFoundError:
                run_extended_lengyel(config_file, cli_args)
    else:
        run_extended_lengyel(config_file, cli_args)

def run_extended_lengyel(config_file, cli_args) -> None:
    """Run the extended Lengyel model as a calculator."""
    config_file = Path(config_file).absolute()
    assert config_file.exists(), f"{config_file} not found."
    assert config_file.suffix == ".yml", f"{config_file} is not a YAML file."

    algorithm = cfspopcon.Algorithm.get_algorithm("extended_lengyel_for_experiment_inputs")

    data_vars = config.read_config(
        elements        = ["input"],
        filepath        = config_file,
        keys            = algorithm.input_keys,
        allowed_missing = algorithm.default_keys,
        overrides       = cli_args,
    )

    algorithm.validate_inputs(data_vars)

    

if __name__ == "__main__":
    run_extended_lengyel_cli()