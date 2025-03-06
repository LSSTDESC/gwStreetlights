import numpy as np
import bilby as bb
from astropy.time import Time
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def O4DutyCycles(detector):
    """
    A function to return the O4b duty cycles, based on a detector supplied
    Duty cycles taken from https://observing.docs.ligo.org/plan/
    """
    
    if detector=="H1":
        return 0.65
    elif detector=="L1":
        return 0.8
    elif detector=="V1":
        return 0.7
    else:
        raise ValueError("Detector {} is not one of 'H1', 'L1', or 'V1'".format(detector))
        return

def detectorListing(obsRun="O4"):
    """
    A function to return a list of interferometers, based on their duty cycles
    """

    if obsRun=="O4":
        dutyFunc=O4DutyCycles
    else:
        raise ValueError("Observing run {} not yet supported for computing duty cycles.".format(obsRun))
        return
    
    detList = []
    for detector in ["H1","L1","V1"]:
        if np.random.random()<dutyFunc(detector):
            detList.append(detector)
    return detList

def get_source_model(source_model_name="BBH"):
    """
    Returns a source model from bilby.gw.source based on the provided model name.

    Parameters:
        source_model_name (str): The name of the source model to retrieve.
                                 Options include "BinaryBlackHole", "BinaryNeutronStar", etc.

    Returns:
        class: The corresponding source model class (e.g., BinaryBlackHole).

    Raises:
        ValueError: If the source model name is not recognized.
    """
    source_models = {
        "BBH": bb.gw.source.lal_binary_black_hole,   # Binary Black Hole model
        "BNS": bb.gw.source.lal_binary_neutron_star,   # Binary Neutron Star model
        # "NSBH": bb.gw.source.NeutronStarBlackHole  # Neutron Star-Black Hole model
    }

    if source_model_name not in source_models:
        raise ValueError(f"Invalid source model '{source_model_name}'. Choose from {list(source_models.keys())}.")

    return source_models[source_model_name]

def get_next_available_dir(base_dir):
    """
    Checks if a directory exists, and if it does, finds the next available numbered version,
    starting from base_dir_0.

    Parameters:
        base_dir (str): The base directory name (e.g., "run").

    Returns:
        str: The next available directory name.
    """
    # Pattern to find existing numbered directories (e.g., run_0, run_1)
    base_pattern = re.escape(base_dir) + r"_(\d+)$"
    existing_numbers = []

    # Check all directories in the parent folder
    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    for entry in os.listdir(parent_dir):
        match = re.match(base_pattern, entry)
        if match:
            existing_numbers.append(int(match.group(1)))

    # Determine the next available number, starting from 0
    next_number = max(existing_numbers, default=-1) + 1
    return f"{base_dir}_{next_number}"

def fakeGeoCentTime():
    """
    A function to generate a fake geocentric time between 2020-2022 (inclusive)
    returns the geocentric time in unix form
    The time is between the start-end time of O4b
    """
    times = ['2024-04-03T00:00:00', '2025-06-04T00:00:00']
    tArr = Time(times, format='isot', scale='tcg')
    return bb.core.prior.Uniform(tArr[0],tArr[1]).sample(1).unix[0]

def makeGeoCentTime(time):
    return Time(time, format='isot', scale='tcg')

def constraintToUniform(priorDict):
    """
    A function to convert any priors in the prior dictionary from bb.core.prior.base.Constraint types to bb.core.prior.prior.Uniform types.
    """
    # Turning entries in the prior dictionary from Constraint type to Uniform over the constraint range
    for i,k in zip(priorDict.keys(),priorDict.values()):
        if type(k) ==bb.core.prior.base.Constraint:
            priorDict[i] = bb.core.prior.Uniform(k.minimum,k.maximum)
            
    return priorDict

def get_merger_prior(merger_type="BBH"):
    """
    Returns a prior dictionary for a given gravitational wave merger type.

    Parameters:
        merger_type (str): The type of merger prior to load.
                          Options include "BBH" (binary black hole),
                          "BNS" (binary neutron star), and "NSBH" (neutron star-black hole).

    Returns:
        bilby.gw.prior.BBHPriorDict or related prior dictionary.

    Raises:
        ValueError: If the merger type is not recognized.
    """
    merger_priors = {
        "BBH": bb.gw.prior.BBHPriorDict,   # Binary Black Hole
        "BNS": bb.gw.prior.BNSPriorDict,   # Binary Neutron Star
        # "NSBH": bb.gw.prior.NSBHPriorDict  # Neutron Star-Black Hole
    }

    if merger_type not in merger_priors:
        raise ValueError(f"Invalid merger type '{merger_type}'. Choose from {list(merger_priors.keys())}.")

    return merger_priors[merger_type]()

    