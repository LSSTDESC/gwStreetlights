import json
import h5py
import os

def rewriteJson(path):
    """
    A function to take a json file and convert it to an h5 file
    
    Inputs
    ------
    path: Full path to json file

    Outputs
    -------
    Full path to h5 file after being written
    """

    # Load JSON file
    with open(path, "r") as f:
        data = json.load(f)

    # Isolate filename
    fname = path.split("/")[-1].split(".")[0]

    # Isolate base path
    basePath = "/"
    for part in path.split("/")[:-1]:
        basePath = os.path.join(basePath,part)

    h5Path = os.path.join(basePath,fname+".h5")
    # Create HDF5 file
    with h5py.File(h5Path, "w") as h5file:
        # Previously, recursively_save_dict_contents was defined here
    
        recursively_save_dict_contents(h5file, data)
    return h5Path

def recursively_save_dict_contents(h5group, dict_data):
    for key, value in dict_data.items():
        if isinstance(value, dict):
            subgroup = h5group.create_group(key)
            recursively_save_dict_contents(subgroup, value)
        else:
            h5group.create_dataset(key, data=value)