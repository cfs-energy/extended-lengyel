"""Loaders for the experimental datasets used in the comparison notebooks."""

import pandas as pd
import xarray as xr
import yaml
from scipy.io import readsav
from cfspopcon.unit_handling import ureg

from ..directories import notebook_dir

DATA_DIRECTORY = notebook_dir / "data"

def check_keys(dataset_prefix: str, config: dict, raw_keys: set, ignored: set, machine_prefix: str=""):
    """Check that every raw dataset key is either mapped in the config or explicitly ignored."""
    mapped_keys = set()
    for remap in config.values():
        if machine_prefix + "name" in remap:
            mapped_keys.add(remap[machine_prefix + "name"])
            if machine_prefix + "err" in remap:
                mapped_keys.add(remap[machine_prefix + "err"])

    missing = raw_keys - mapped_keys - ignored
    if missing:
        raise ValueError(f"{dataset_prefix} dataset contains unmapped keys: {missing}")

    defined_ignored = mapped_keys & ignored
    if defined_ignored:
        raise ValueError(f"Definition provided for {defined_ignored} but these keys are marked as ignored.")

def load_nf23_data(
    yaml_path=DATA_DIRECTORY / "renaming.yaml",
    nf23_txt_path=DATA_DIRECTORY / "divertor_data.txt",
) -> xr.Dataset:
    """Loads Henderson et al. 2023 ASDEX Upgrade divertor data.

    This dataset includes parameters such as target temperature (TDIV),
    separatrix power (PSEP), and impurity concentrations (CN, CNE, CAR)
    measured during mixed seeding experiments.
    """
    with open(yaml_path, encoding="UTF-8") as f:
        config = yaml.safe_load(f)["NF23"]

    df = pd.read_csv(nf23_txt_path, delimiter=" ")
    raw_keys = set(df.columns)
    ignored = set(config.pop("ignored", []))

    check_keys("NF23", config, raw_keys, ignored)

    ds = xr.Dataset()

    for name, remap in config.items():

        units = remap.get("units", "")
        # Quantify against cfspopcon's registry explicitly: other packages (e.g. radas) replace
        # the pint application registry on import, and only cfspopcon's defines n19/n20.
        ds[name] = df[remap["name"]].to_xarray().pint.quantify(units, unit_registry=ureg)
        if "err" in remap:
            ds[f"{name}_err"] = df[remap["err"]].to_xarray().pint.quantify(units, unit_registry=ureg)

    return ds

def load_nme21_data(
    yaml_path=DATA_DIRECTORY / "renaming.yaml",
    nme21_aug_sav_path=DATA_DIRECTORY / "database_scaling_aug.sav",
    nme21_jet_sav_path=DATA_DIRECTORY / "database_jet.sav",
) -> dict[str, xr.Dataset]:
    """Loads Henderson et al. 2021 JET and ASDEX Upgrade nitrogen scaling data.

    This dataset assesses parameter dependencies for detachment thresholds,
    comparing nitrogen concentrations (cn_mean) against separatrix density
    and power flows.
    """
    with open(yaml_path, encoding="UTF-8") as f:
        config = yaml.safe_load(f)["NME21"]

    AUG_ignored = set(config.pop("AUG_ignored", []))
    JET_ignored = set(config.pop("JET_ignored", []))

    def _process_machine_data(sav_path, machine_prefix, ignored):
        sav_data = readsav(sav_path)
        raw_keys = {k for k in sav_data.keys() if not k.startswith("_")}

        check_keys(f"NME21_{machine_prefix}", config, raw_keys, ignored, machine_prefix)

        ds = xr.Dataset()

        for name, remap in config.items():

            units = remap.get(machine_prefix + "units", "")

            if machine_prefix + "name" in remap:
                ds[name] = xr.DataArray(sav_data(remap[machine_prefix + "name"])).pint.quantify(units, unit_registry=ureg)
                if machine_prefix + "err" in remap:
                    ds[f"{name}_err"] = xr.DataArray(sav_data(remap[machine_prefix + "err"])).pint.quantify(units, unit_registry=ureg)

        return ds

    return {
        "JET": _process_machine_data(nme21_jet_sav_path, "JET_", JET_ignored),
        "AUG": _process_machine_data(nme21_aug_sav_path, "AUG_", AUG_ignored),
    }

if __name__ == "__main__":
    ds_nf23 = load_nf23_data()
    nme21_datasets = load_nme21_data()

    print("NF23 Nitrogen concentration (percent):", ds_nf23.nitrogen_concentration.values[:5])
    print("NME21 JET Magnetic Field (T):", nme21_datasets["JET"].magnetic_field_on_axis.values[:5])
