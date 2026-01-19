#!/usr/bin/env python

import os
import yaml
import pickle
import numpy as np
os.environ["GCR_CONFIG_SOURCE"] = "files"
import GCRCatalogs as GCRCat
import sys
sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils/utils.py")
import utils as ut

# Project-specific imports
# Assumes these already exist in your environment
# -----------------------------------------------
# from some_module import  dataColumns, 
def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def getColumnsFromFile(fname):
    """Read columns from column file path"""
    with open(fname) as f:
        a = f.read().splitlines()
    return a

def save_data_products(
    output_path,
    results,
    data,
    limiting_mags,
    hp_band_dict,
    save_format="npz",
):
    """
    Save core data products to disk.

    Parameters
    ----------
    output_path : str
        Base path (no extension).
    save_format : str
        'npz' or 'pickle'
    """
    data.to_csv(os.path.join(output_path,"data.csv"))
    if save_format == "npz":
        np.savez(
            os.path.join(output_path,"results"),
            results=results,
            limiting_mags=limiting_mags,
            hp_band_dict=hp_band_dict,
        )
    elif save_format == "pickle":
        with open(os.path.join(output_path,"results.pkl"), "wb") as f:
            pickle.dump(
                {
                    "results": results,
                    "limiting_mags": limiting_mags,
                    "hp_band_dict": hp_band_dict,
                },
                f,
            )
    else:
        raise ValueError(f"Unknown save format: {save_format}")

galaxySizes = ["small","medium","large"]

def getCatalogFromSize(string):
    """
    Map a human-readable galaxy catalog size label to the corresponding
    SkySim catalog name.

    Parameters
    ----------
    string : str
        Catalog size specifier. Must be one of:
        {"small", "medium", "large"} (case-insensitive).

    Returns
    -------
    str
        The SkySim catalog identifier corresponding to the requested size:
        - "small"  -> "skysim5000_v1.2_small"
        - "medium" -> "skysim5000_v1.2_image"
        - "large"  -> "skysim5000_v1.2"

    Raises
    ------
    ValueError
        If the provided size string does not match one of the accepted
        catalog sizes.
    """
    if not (string.lower() in galaxySizes):
        raise ValueError(f"Galaxy catalog size provided ({string.lower()} is not one of the accepted sizes ({galaxySizes}), exiting.")
        return -1
    if string.lower()=='small':
        return "skysim5000_v1.2_small"
    elif string.lower()=='medium':
        return "skysim5000_v1.2_image"
    elif string.lower()=='large':
        return "skysim5000_v1.2"
    else:
        raise ValueError(f"Something went wrong in `getCatalogFromSize` function, provided the parameter: {string}")
        return -1

def getPlotParams(isProd):
    """
    Helper function to provide plot parameters based on draft or production level styling.
    """
    if type(isProd)!=type(True):
        raise ValueError(f"Parameter provided to `getPlotParams` function is not of type bool ({isProd}, type {type(isProd)}), exiting.")
        return -1
    elif isProd:
        return {"z_step_pz":0.01,
              "z_step_lf":0.025,
              "brightMag":-28,
              "faintMag":-11,
              "p0":(1e-3, -22.0, -1.1),
              "maxfev":50000,
              "NSIDE":NSIDE,
              "fit_schecter":True,
              "delta_mag_schecter":0.025,
              "vmax":True,
              "z_min":0.1,} # Prod dict
    else:
        return {"z_step_pz":0.01,
               "z_step_lf":0.1,
               "brightMag":-28,
               "faintMag":-11,
               "p0":(1e-3, -22.0, -1.1),
               "maxfev":50000,
               "NSIDE":NSIDE,
               "fit_schecter":True,
               "delta_mag_schecter":0.1,
               "vmax":True,
               "z_min":0.1,} # Draft dict


def main(config_path):
    # ------------------------
    # Load configuration
    # ------------------------
    cfg = load_config(config_path)

    survey = cfg["survey"]
    diagnostics = cfg["diagnostics"]
    io_cfg = cfg["io"]
    runtime = cfg["runtime"]

    verbose = runtime.get("verbose", False)

    # Derived quantities
    NSIDE = survey["nside"]
    
    size = survey["sim_size"]
    simulatedCatalog = GCRCat.load_catalog(getCatalogFromSize(size))

    plotParams = getPlotParams(diagnostics["prod"])
    
    mags_deeper = survey["mags_deeper_factor"] * survey["uniformity"]

    dataColumns = getColumnsFromFile(io_cfg["column_file"])

    # ------------------------
    # Prepare output dirs
    # ------------------------
    prefix = io_cfg["path_prefix"]
    runPath = os.path.join(io_cfg["output_dir"],prefix)
    ensure_dir(runPath)
    
    figPath = os.path.join(runPath,io_cfg["figure_dir"])
    ensure_dir(figPath)

    if verbose:
        print("Running LSST simulated observations pipeline with configuration:")
        print(cfg)

    # ------------------------
    # Data query
    # ------------------------
    data, limiting_mags, hp_band_dict = ut.apply_lsst_depth_and_uniformity(
        simulatedCatalog,
        survey["year"],
        dataColumns,
        survey["airmass"],
        survey["z_max"],
        NSIDE,
        z_min=survey["z_min"],
        mags_deeper=mags_deeper,
        uniformity=survey["uniformity"],
        modeled=survey["modeled"],
        spectroscopic=survey["spectroscopic"],
        verbose=verbose,
    )

    # ------------------------
    # Diagnostics & plots
    # ------------------------
    results, mag_lim_check_result, figs, axes = ut.run_survey_diagnostics(
        data,
        hp_band_dict,
        simulatedCatalog,
        survey["year"],
        survey["z_max"],
        limiting_mags,
        **plotParams,
        modeled=survey["modeled"],
        hi_mag=diagnostics["hi_mag"],
        low_mag=diagnostics["low_mag"],
        spectroscopic=survey["spectroscopic"],
        verbose=verbose,
    )

    # ------------------------
    # Save figures
    # ------------------------
    for i, fig in enumerate(figs):
        fig_path = os.path.join(
            io_cfg["figure_dir"], f"{prefix}_fig{i}.jpg"
        )
        fig.savefig(fig_path)
        if verbose:
            print(f"Saved figure: {fig_path}")

    # ------------------------
    # Save data products
    # ------------------------

    save_data_products(
        runPath,
        results,
        data,
        limiting_mags,
        hp_band_dict,
        save_format=io_cfg.get("save_format", "npz"),
    )

    if verbose:
        print("Saved data products:")
        print(f"  {runPath}.{io_cfg.get('save_format', 'npz')}")

    if verbose:
        print("Pipeline complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LSST survey diagnostics from a config file"
    )
    parser.add_argument(
        "config",
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()
    main(args.config)