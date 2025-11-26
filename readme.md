# DESC Project 411: Quantifying the impact of galaxy catalog conditions on dark sirens

We aim to quantify the constraining power of dark sirens under different observational conditions, using SkySim5000 as the host galaxy catalog, and simulated Bilby GW events as the GW host catalog. 

We will test different galaxy catalog conditions, including
1. LSST photometric limits on SkySim5000 catalog
1. Photo-z errors impact on H0 PE
1. Galaxy completeness effects

We also will test GW host catalog constraints, including different CBC weighting schemes that have an impact on the underlying P(z) distribution.

This project will have four deliverables
1. Deliver CBC host catalogs for testing methods.
    1. Scripts used in the production of these catalogs are in `./CBCCats/`. After creating the `bilby_pipe` `.ini` files, you simply need to run `bilby_pipe path/to/file.ini --submit` to submit the `bilby_pipe` job.
2. Validate dark siren statistical methods with simulated catalogs.
    1. Scripts for this goal can be found in `./initialValidation/`. The focus is to ensure that the CBC catalog is representative, and that the dark siren method can recover the relevant cosmology in the fiducial case.
3. Test different galaxy and GW host conditions.
    1. Scripts for this goal can be found in `./hostConditions/`
4. Deliver the first pipeline for performing a dark siren measurement with LSST.
