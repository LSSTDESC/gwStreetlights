# Creating CBC host catalogs

Here we have the code for simulating CBC host catalogs. It follows the following five steps:

-  Evaluate the parent EM catalog, and draw CBC samples based on some weighting criteria.
    - For the unaltered SkySim5000 catalog, we use `SkySim_validation_uniformParent.ipynb`. This notebook evaluates different SkySim5000 distributions, and creates two cbc sample csv's, one for uniform weighting, and one for stellar mass weighting.
    - `SkySim_hosts_prob.py` used a model described in [this paper](https://arxiv.org/abs/2405.07904).
-  Load the mock CBC catalog, and sample CBC's based on a given prior file.
    - `MakeBilbySamples.py` is the script to do this. The luminosity distance is determined using a cosmology, so you need to assume a cosmology here (relevant for different CBC catalog generation).
-  Computes the network and individual SNR of the drawn CBC's.
    -  `SepBilbyBySNR.py` is the scriptto do this.
    -  `MakeBilbySamples.py`, and adds that information to a new `.csv`
-  Downselects the CBC host galaxies to make up the CBC catalogs used in the forthcoming analysis.
    -  `Selecting catalogs.ipynb` is the script that does this. The notebook also includes some quick visualizations to ensure that the the input sample is not overtly biased.
-  Reads a base `.ini` file associated with a `bilby_pipe` submission, and appends the necessary lines to the file for job submission based on the CBC samples.
    - `iniLineMaker.ipynb` is the script to do this. 

The simulated catalogs are saved in the `~/data/` folder, with a readme detailing the specifics of each catalog
