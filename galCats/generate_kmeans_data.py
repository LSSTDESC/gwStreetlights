#!/usr/bin/env python

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid, simpson, cumulative_trapezoid
from scipy.interpolate import CubicSpline
import kmeans_radec as km
import os, h5py
import argparse
import logging
import glob

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# This file calculates kmeans ra/dec centers and labels
# File owner: Sean MacBride
# Based on work started by Isaac McMahon

logger.info("Initializing functions and parser...")

parser = argparse.ArgumentParser()
# parser.add_argument("-c", "--catalog", type=str, help="The path to the .csv")
parser.add_argument(
    "-i",
    "--iteration",
    type=int,
    help="The iterator of the specific combined .csv to run",
)
args = parser.parse_args()

# Number of kmeans centers, by which to divide the survey footprint
ncen = 50
# Number of absolute magnitude bins
nbins = 500

# This is a file that is just the full catalog as one hdf5
# Needs columns for mags, z, zerr, ra, dec

# catalog_path = args.catalog
# parentPath = os.path.dirname(catalog_path)

catalog_path = np.sort(
    glob.glob(
        "/global/cfs/cdirs/lsst/groups/MCP/standardSirens/mockGalCats/prod*/combined*.csv"
    )
)[args.iteration]
parentPath = os.path.dirname(catalog_path)

# I have manual cosmology interpolation for speed
# Should be accurate to a difference of 1e-7 or so to astropy
H0 = 100
O_m = 0.3065
O_l = 1 - O_m


def getColorRange(band):
    if band == "g" or band == "r":
        return [-0.1, 1.9]
    elif band == "i":
        return [0, 3]
    elif band == "z":
        return [-0.1, 1.9]
    else:
        raise ValueError("Provided band ({}) not one of g, r, i, z.".format(band))


def getKCorrFunction(band):
    if band == "g":
        return calc_kcorr_chil_g
    elif band == "r":
        return calc_kcorr_chil_r
    elif band == "i":
        return calc_kcorr_chil_i
    elif band == "z":
        return calc_kcorr_chil_z
    else:
        raise ValueError(f"Provided band ({band}) not one of g, r, i, z.")


def getColors(band):
    """
    This function returns columns used in the color calculation, in the form of [band0,band1]
    They are used in this script as band0 - band1
    """
    if band == "g" or band == "r":
        return ["g", "r"]
    elif band == "i":
        return ["g", "i"]
    elif band == "z":
        return ["r", "z"]
    else:
        raise ValueError(f"Provided band ({band}) not one of g, r, i, z.")


def comoving_distance(z_array, n_points=10):
    x = z_array[:, None] * np.linspace(0, 1, n_points)
    dx = z_array * 1.0 / (n_points - 1)
    y = (O_m * ((1 + x) ** 3) + O_l) ** -0.5
    avg_y = (y[:, :-1] + y[:, 1:]) / 2
    return np.sum(dx[:, np.newaxis] * avg_y, axis=1) * 299792.458 / H0


def luminosity_distance(z, n_points=10):
    return (1 + z) * comoving_distance(z, n_points)


def comoving_volume(z, n_points=10):
    return 4.18879 * (comoving_distance(z, n_points) ** 3)


def calc_kcorr_chil_g(z, gr_color):
    x, y = z, gr_color
    (
        p10,
        p11,
        p12,
        p13,
        p20,
        p21,
        p22,
        p23,
        p30,
        p31,
        p32,
        p33,
        p40,
        p41,
        p42,
        p43,
        p50,
        p51,
        p52,
        p60,
        p61,
        p70,
    ) = [
        -2.45204,
        4.10188,
        10.5258,
        -13.5889,
        56.7969,
        -140.913,
        144.572,
        57.2155,
        -466.949,
        222.789,
        -917.46,
        -78.0591,
        2906.77,
        1500.8,
        1689.97,
        30.889,
        -10453.7,
        -4419.56,
        -1011.01,
        17568,
        3236.68,
        -10820.7,
    ]
    return (
        p10 * x
        + p11 * x * y
        + p12 * x * y**2
        + p13 * x * y**3
        + p20 * x**2
        + p21 * y * x**2
        + p22 * (x**2) * (y**2)
        + p23 * (x**2) * (y**3)
        + p30 * x**3
        + p31 * y * x**3
        + p32 * (x**3) * (y**2)
        + p33 * (x**3) * (y**3)
        + p40 * x**4
        + p41 * y * x**4
        + p42 * (x**4) * (y**2)
        + p43 * (x**4) * (y**3)
        + p50 * x**5
        + p51 * y * x**5
        + p52 * (x**5) * (y**2)
        + p60 * x**6
        + p61 * y * x**6
        + p70 * x**7
    )


def calc_kcorr_chil_r(z, gr_color):
    # https://arxiv.org/abs/1002.2360
    x, y = z, gr_color
    p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p40, p41, p50 = [
        1.83285,
        -2.71446,
        4.97336,
        -3.66864,
        -19.7595,
        10.5033,
        18.8196,
        6.07785,
        33.6059,
        -120.713,
        -49.299,
        144.371,
        216.453,
        -295.39,
    ]
    return (
        p10 * x
        + p11 * x * y
        + p12 * x * y**2
        + p13 * x * y**3
        + p20 * x**2
        + p21 * y * x**2
        + p22 * (x**2) * (y**2)
        + p23 * (x**2) * (y**3)
        + p30 * x**3
        + p31 * y * x**3
        + p32 * (x**3) * (y**2)
        + p40 * x**4
        + p41 * y * x**4
        + p50 * x**5
    )


def calc_kcorr_chil_z(z, rz_color):
    x, y = z, rz_color
    p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p40, p41, p50 = [
        0.669031,
        -3.08016,
        9.87081,
        -7.07135,
        -18.6165,
        8.24314,
        -14.2716,
        13.8663,
        94.1113,
        11.2971,
        -11.9588,
        -225.428,
        -17.8509,
        197.505,
    ]
    return (
        p10 * x
        + p11 * x * y
        + p12 * x * y**2
        + p13 * x * y**3
        + p20 * x**2
        + p21 * y * x**2
        + p22 * (x**2) * (y**2)
        + p23 * (x**2) * (y**3)
        + p30 * x**3
        + p31 * y * x**3
        + p32 * (x**3) * (y**2)
        + p40 * x**4
        + p41 * y * x**4
        + p50 * x**5
    )


def calc_kcorr_chil_i(z, gi_color):
    # https://arxiv.org/abs/1002.2360
    x, y = z, gi_color
    p10, p11, p12, p13, p20, p21, p22, p23, p30, p31, p32, p40, p41, p50 = [
        -2.21853,
        3.94007,
        0.678402,
        -1.24751,
        -15.7929,
        -19.3587,
        15.0137,
        2.27779,
        118.791,
        -40.0709,
        -30.6727,
        -134.571,
        125.799,
        -55.4483,
    ]
    return (
        p10 * x
        + p11 * x * y
        + p12 * x * y**2
        + p13 * x * y**3
        + p20 * x**2
        + p21 * y * x**2
        + p22 * (x**2) * (y**2)
        + p23 * (x**2) * (y**3)
        + p30 * x**3
        + p31 * y * x**3
        + p32 * (x**3) * (y**2)
        + p40 * x**4
        + p41 * y * x**4
        + p50 * x**5
    )


def schechter_magnitude(M, phi_star, M_star, alpha):
    return (
        phi_star
        * (10 ** (0.4 * (M_star - M))) ** (alpha + 1)
        * np.exp(-(10 ** (0.4 * (M_star - M))))
    )


logger.info("Making splines and defining k_corr functions.")

# Don't forget to adjust the range of this z interpolation if you need further redshifts
z_table = np.linspace(0, 3, 100000)
dl_table = luminosity_distance(z_table, 1000)
dc_table = comoving_distance(z_table, 1000)
vc_table = 4.18879 * (dc_table**3)  # Just a sphere
dl_to_z = CubicSpline(dl_table, z_table)
z_to_dc = CubicSpline(z_table, dc_table)
z_to_dl = CubicSpline(z_table, dl_table)
z_to_vc = CubicSpline(z_table, vc_table)

logger.info("Done making splines.")
logger.info("Loading data...")


def get_entry_indices(filename, search_list):
    try:
        with open(filename, "r") as f:
            first_line = f.readline().strip()

            if not first_line:
                return "The file is empty."

            # Split by comma and clean whitespace
            all_entries = [item.strip() for item in first_line.split(",")]

            # Create a dictionary to store the results
            # We use a list comprehension to find all indices for each target item
            results = {}
            for target in search_list:
                indices = [int(i) for i, val in enumerate(all_entries) if val == target]
                if indices:
                    results[target] = indices

            return results

    except FileNotFoundError:
        return "File not found."


# kmeans_dir = "/home/smacbr/delve-o4c-preparation/los_prior/kmeans"
# if not os.path.exists(kmeans_dir):
#     os.makedirs(kmeans_dir)

# kmeans_radec from https://github.com/esheldon/kmeans_radec
# Precalculates a certain number of nodes which defines separate areas/labels to assign galaxies
# Saves the precalculation to a file. If you don't change ncen or the catalog then it's reusable
# I use these labels later to separate the galaxies into different sky areas to account for variation across the footprint
if os.path.exists(os.path.join(parentPath, f"km_labels_{ncen}.npy")):
    labels_unmasked = np.load(
        os.path.join(parentPath, f"km_labels_{ncen}.npy"), allow_pickle=True
    )
else:
    ra_dec_dict = get_entry_indices(catalog_path, ["ra_true", "dec_true"])

    df = pd.read_csv(
        catalog_path,
        usecols=[int(ra_dec_dict["ra_true"][0]), int(ra_dec_dict["dec_true"][0])],
    )

    ra_unmasked = np.array(df["ra_true"])
    dec_unmasked = np.array(df["dec_true"])
    pos_unmasked = np.array([ra_unmasked, dec_unmasked]).T
    if os.path.exists(os.path.join(parentPath, f"km_centers_{ncen}.npy")):
        centers = np.load(
            os.path.join(parentPath, f"km_centers_{ncen}.npy"), allow_pickle=True
        )
        km_kernel = km.KMeans(centers)
    else:
        km_kernel = km.kmeans_sample(
            pos_unmasked[
                np.random.choice(
                    np.arange(0, len(pos_unmasked) - 1),
                    len(pos_unmasked) // 100,
                    replace=False,
                )
            ],
            ncen,
            maxiter=100,
            tol=1.0e-5,
        )
        np.save(
            os.path.join(parentPath, f"km_centers_{ncen}.npy"),
            km_kernel.centers,
            allow_pickle=True,
        )
    labels_unmasked = []
    for i in tqdm(range(1 + (len(pos_unmasked) // 1000000))):
        if i < len(pos_unmasked) // 1000000:
            labels_unmasked = np.append(
                labels_unmasked,
                km_kernel.find_nearest(pos_unmasked[1000000 * i : 1000000 * (i + 1)]),
            )
        else:
            labels_unmasked = np.append(
                labels_unmasked, km_kernel.find_nearest(pos_unmasked[1000000 * i :])
            )
    np.save(
        os.path.join(parentPath, f"km_labels_{ncen}"),
        labels_unmasked.astype("int"),
        allow_pickle=True,
    )
    del pos_unmasked
    del ra_unmasked
    del dec_unmasked

logger.info("Done generating/loading k_means centers.")
