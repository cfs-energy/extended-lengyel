#!.venv/bin/python
# Run this script from the repository directory.
"""CLI for extended-lengyel."""

import click
import warnings
from pathlib import Path
import xarray as xr
import yaml

from . import config
from . import extended_lengyel_model
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
                from ipdb import launch_ipdb_on_exception  # type:ignore[import-untyped]

                with launch_ipdb_on_exception():
                    run_extended_lengyel(config_file, output_file, cli_args)
            except ModuleNotFoundError:
                run_extended_lengyel(config_file, output_file, cli_args)
    else:
        run_extended_lengyel(config_file, output_file, cli_args)

def run_extended_lengyel(config_file, output_file, cli_args) -> None:
    """Run the extended Lengyel model as a calculator."""
    config_file = Path(config_file).absolute()
    assert config_file.exists(), f"{config_file} not found."
    assert config_file.suffix == ".yml", f"{config_file} is not a YAML file."

    output_file = Path(output_file).absolute()
    if output_file.exists():
        click.confirm(f"{output_file} already exists. Overwrite?", abort=True)
    assert output_file.suffix == ".yml", f"{output_file} is not a YAML file."

    algorithm = cfspopcon.CompositeAlgorithm.from_list([
            "calc_magnetic_field_and_safety_factor",
            "calc_fieldline_pitch_at_omp",
            "set_radas_dir",
            "read_atomic_data",
            "calc_kappa_e0",
            "build_CzLINT_for_seed_impurities",
            "build_mean_charge_for_seed_impurities",
            "build_CzLINT_for_fixed_impurities",
            "build_mean_charge_for_fixed_impurities",
            "calc_momentum_loss_from_cc_fit",
            "calc_power_loss_from_cc_fit",
            "calc_electron_temp_from_cc_fit",
            "run_extended_lengyel_model_with_S_Zeff_and_alphat_correction",
            "calc_sound_speed_at_target",
            "calc_target_density",
            "calc_flux_density_to_pascals_factor",
            "calc_parallel_to_perp_factor",
            "calc_ion_flux_to_target",
            "calc_divertor_neutral_pressure",
            "calc_heat_flux_perp_to_target"
    ])

    data_vars = config.read_config(
        elements        = ["input"],
        filepath        = config_file,
        keys            = algorithm.input_keys,
        allowed_missing = algorithm.default_keys,
        overrides       = cli_args,
        warn_if_unused  = True,
    )

    ds = xr.Dataset(data_vars=data_vars)
    algorithm.validate_inputs(ds)
    ds = algorithm.update_dataset(ds)

    write_output_file(output_file, ds)

    print("Extended lengyel model ran successfully.")

def write_output_file(filepath: Path, ds: xr.Dataset):
    """"""
    from cfspopcon.file_io import sanitize_variable, ignored_keys
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
        if key in ignored_keys: continue
        output_dict[key] = sanitize_variable(ds[key], key)

    for key in ds.coords:
        if key in ignored_keys: continue
        output_dict[key] = sanitize_variable(ds[key], key)

    impurity_fraction = cfspopcon.unit_handling.magnitude_in_units(ds["impurity_fraction"], "")
    seed_impurity_concentration = impurity_fraction * ds["seed_impurity_weights"].dropna(dim="dim_species")
    fixed_impurity_concentration = ds["fixed_impurity_weights"].dropna(dim="dim_species")

    output_impurity_fraction = dict(seed_impurity=dict(), fixed_impurity=dict())
    for cz in seed_impurity_concentration:
        output_impurity_fraction["seed_impurity"][item(cz.dim_species).name] = item(cz)
    for cz in fixed_impurity_concentration:
        output_impurity_fraction["fixed_impurity"][item(cz.dim_species).name] = item(cz)

    for k, v in output_dict.items():
        units = getattr(v, "units", None)
        v = v.values.tolist()
        if units is not None:
            output_dict[k] = f"{v} {units}"
        else:
            output_dict[k] = v
        
    output_dict["impurity_fraction"] = output_impurity_fraction

    with open(filepath, "w") as f:
        f.write(yaml.dump(output_dict))


if __name__ == "__main__":
    run_extended_lengyel_cli()