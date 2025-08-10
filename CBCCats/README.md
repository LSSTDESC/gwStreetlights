# Creating CBC host catalogs

Here we have the code for simulating CBC host catalogs.

-  `SkySim_hosts_prob.py` is used to simulate a host catalog over the whole sky using the small SkySim5000 catalog (~50 deg^2)
    -  This used a model described in [this paper](https://arxiv.org/abs/2405.07904)
-  `MakeBilbySamples.py` loads the mock CBC catalog that was made in `SkySim_hosts_prob.py`, and samples a new CBC based on a given prior file. For the nominal CBC data set, the luminosity distance is determined using the SkySim5000 cosmology, and the `redshiftHubble` parameter of the associated mock CBC.
-  `SepBilbyBySNR.py` computes the network and individual SNR of the CBC based on the injection parameters that were sampled in `MakeBilbySamples.py`, and adds that information to a new `.csv`
-  `Selecting catalogs.ipynb` downselects the CBC host galaxies to make up the CBC catalogs used in the forthcoming analysis. The notebook also includes some quick visualizations to ensure that the the input sample is not overtly biased.
-  `iniLineMaker.ipynb` reads a base `.ini` file associated with a `bilby_pipe` submission, and appends the necessary lines to the file for job submission. 

The simulated catalogs are saved in the `~/data/` folder, with a readme detailing the specifics of each catalog