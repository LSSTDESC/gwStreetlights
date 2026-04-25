#!/usr/bin/env python

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ["GCR_CONFIG_SOURCE"] = "files"
import GCRCatalogs as GCRCat
import sys

sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut


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
    output_path, results, data, limiting_mags, hp_band_dict, save_format="npz",kmeans=False,kmean_it=0,
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

    if kmeans:
        append_to = f"kmeans_{kmean_it}_"
    else:
        append_to = ""
    
    # data.to_csv(os.path.join(output_path, "data.csv"), index=False) # Removing this for now
    if save_format == "npz":
        np.savez(
            os.path.join(output_path, f"{append_to}combined_results"),
            results=results,
            limiting_mags=limiting_mags,
            hp_band_dict=hp_band_dict,
        )
    elif save_format == "pickle":
        with open(os.path.join(output_path, f"{append_to}combined_results.pkl"), "wb") as f:
            pickle.dump(
                {
                    "results": results,
                    "limiting_mags": limiting_mags,
                    "hp_band_dict": hp_band_dict,
                },
                f,
            )
    else:
        raise ValueError("Unknown save format:", save_format)


galaxySizes = ["small", "medium", "large"]


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
        raise ValueError(
            "Galaxy catalog size provided ({} is not one of the accepted sizes ({}), exiting.".format(
                string.lower(), galaxySizes
            )
        )
        return -1
    if string.lower() == "small":
        return "skysim5000_v1.2_small"
    elif string.lower() == "medium":
        return "skysim5000_v1.2_image"
    elif string.lower() == "large":
        return "skysim5000_v1.2"
    else:
        raise ValueError(
            f"Something went wrong in `getCatalogFromSize` function, provided the parameter: {string}"
        )
        return -1


def getPlotParams(isProd, nside=256):
    """
    Helper function to provide plot parameters based on draft or production level styling.
    """
    if type(isProd) != type(True):
        raise ValueError(
            f"Parameter provided to `getPlotParams` function is not of type bool ({isProd}, type {type(isProd)}), exiting."
        )
        return -1
    elif isProd:
        return {
            "z_step_pz": 0.01,
            "z_step_lf": 0.025,
            "brightMag": -28,
            "faintMag": -11,
            "p0": (1e-3, -22.0, -1.1),
            "maxfev": 50000,
            "NSIDE": nside,
            "fit_schecter": True,
            "delta_mag_schecter": 0.025,
            "vmax": True,
            "z_min": 0.1,
        }  # Prod dict
    else:
        return {
            "z_step_pz": 0.01,
            "z_step_lf": 0.1,
            "brightMag": -28,
            "faintMag": -11,
            "p0": (1e-3, -22.0, -1.1),
            "maxfev": 50000,
            "NSIDE": nside,
            "fit_schecter": True,
            "delta_mag_schecter": 0.1,
            "vmax": True,
            "z_min": 0.1,
        }  # Draft dict


def main(config_path, data_path, doKmeans, kmean_label):
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
    nside = survey["nside"]

    size = survey["sim_size"]
    simulatedCatalog = GCRCat.load_catalog(getCatalogFromSize(size))

    plotParams = getPlotParams(diagnostics["prod"], nside=nside)

    mags_deeper = survey["mags_deeper_factor"] * survey["uniformity"]

    dataColumns = getColumnsFromFile(io_cfg["column_file"])

    h_val = survey.get("h", None)

    # ------------------------
    # Read in the combined data csv
    # ------------------------
    logger.info("Loading the combined data from {}".format(data_path))
    data = pd.read_csv(data_path)

    # Split up the dataframe for the kmeans case
    if doKmeans:
        loaded = np.load(os.path.join(os.path.split(data_path)[0], "km_labels_50.npy"))
        subData = data.loc[[x == kmean_label for x in loaded]]
        del data
        data = subData
        del subData
        figString = f"kmean_{kmean_label}_"
    else:
        figString = "all"

    # ------------------------
    # Regenerate the hp_band_dict
    # ------------------------

    logger.info("Regenerating the limiting magnitude dictionary")
    hp_band_dict = dict()
    indiv_mag_dict = ut.get_limiting_mag_dict(
        data,
        np.unique(data["hp_ind_nside128"]),
        ut.LSST_bands,
    )
    # Concatenate to hp_band_dict
    for k in indiv_mag_dict.keys():
        hp_band_dict[k] = indiv_mag_dict[k]

    # ------------------------
    # Regenerate the limiting_mags
    # ------------------------

    limiting_mags = ut.GCR_mag_filter_from_year(
        survey["year"], ut.LSST_bands, ut.eTime_dict, ut.visits_dict
    )

    # ------------------------
    # Diagnostics & plots
    # ------------------------
    results, mag_lim_check_result, figs, axes, phi_dictionary = ut.run_survey_diagnostics(
        data,
        hp_band_dict,
        simulatedCatalog,
        survey["year"],
        survey["z_max"],
        limiting_mags,
        **plotParams,
        modeled=bool(survey["modeled"]),
        hi_mag=float(diagnostics["hi_mag"]),
        low_mag=float(diagnostics["low_mag"]),
        spectroscopic=bool(survey["spectroscopic"]),
        verbose=verbose,
        div=50 if doKmeans else 1,
    )

    # ------------------------
    # Save data products
    # ------------------------

    runPath = os.path.split(data_path)[0]

    save_data_products(
        runPath,
        results,
        data,
        limiting_mags,
        hp_band_dict,
        phi_dictionary,
        save_format=io_cfg.get("save_format", "npz"),
        kmeans = doKmeans,
        kmean_it = kmean_label
    )

    # ------------------------
    # Save figures
    # ------------------------
    if not doKmeans:
        figPath = os.path.join(runPath, "figures")
    
        for i, fig in enumerate(figs):
            fig_path = os.path.join(figPath, f"{figString}_fig{i}.jpg")
            fig.savefig(fig_path)
            if verbose:
                logger.info(f"Saved figure: {fig_path}")

    if verbose:
        logger.info(f"Saved data products to {runPath}")

    if verbose:
        logger.info("Pipeline complete, exiting.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LSST survey diagnostics AFTER the GCR catalog query. Best used on the combined dataset"
    )

    # parser.add_argument(
    #     "-d",
    #     "--dataDir",
    #     help="Path to the cfs directory containing the combined dataset.",
    # )

    parser.add_argument(
        "-k",
        "--doKmeans",
        action="store_true",
        help="If set, will perform kmeans calculation.",
    )

    parser.add_argument(
        "--kmean_label",
        type=int,
        required=False,
        help="The kmeans label to use for this iteration.",
    )

    args = parser.parse_args()

    if args.doKmeans:
        rem = args.kmean_label%50 # This becomes the kmean label 
        quot = args.kmean_label//50 
        dirTextFile = ("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/prod_directories.txt")
        dataDir = Path(dirTextFile).read_text().splitlines()[quot]
        
    combinedCSVPath = os.path.join(dataDir, "combined.csv")
    runLabel = dataDir.split("/")[-2]

    configDir = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config"

    config = os.path.join(configDir, runLabel + ".yaml")

    main(config, combinedCSVPath, args.doKmeans, rem)
