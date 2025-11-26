# Creating CBC host catalogs

Here we have the code for simulating CBC host catalogs in advance of `bilby_pipe` parameter estimation. It follows the following four steps:

1.  Evaluate the parent EM catalog, and draw CBC samples based on some weighting criteria.
    - For the unaltered SkySim5000 catalog, we use `SkySim_validation_uniformParent.ipynb`. This notebook evaluates different SkySim5000 distributions, and creates two cbc sample csv's, one for uniform weighting, and one for stellar mass weighting.
    - `SkySim_hosts_prob.py` used a model described in [this paper](https://arxiv.org/abs/2405.07904).
2.  Load the mock CBC catalog, and sample CBC's based on a given prior file.
    - `MakeBilbySamples.py` is the script to do this. The luminosity distance is determined using a cosmology, so you need to assume a cosmology here (relevant for different CBC catalog generation).
    - You can use `generate_MakeBilbySamples_prompts.py` to generate the prompts for this script quickly.
3.  Compute the network and individual SNR of the drawn CBC's, and downselects the CBC host galaxies to make up the CBC catalogs used in the forthcoming analysis.
    -  Now, I use `SNR_sep_downselect.py` and `generate_SNR_sep_downsample_prompts.py`, which is much faster
    -  Previously, I used two scripts to do this:
        -  `SepBilbyBySNR.py`
        -  `Selecting catalogs.ipynb`    
4.  Read a base `.ini` file associated with a `bilby_pipe` submission, and appends the necessary lines to the file for job submission based on the CBC samples.
    - `iniLineMaker.py` is the script to do this. 

The simulated catalogs are saved in the `~/data/` folder, with a readme detailing the specifics of each catalog
