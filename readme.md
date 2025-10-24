# DESC Project 411: Quantifying the impact of galaxy catalog conditions on dark sirens

We aim to quantify the constraining power of dark sirens under different observational conditions, using SkySim5000 as the host galaxy catalog, and simulated Bilby GW events as the GW host catalog. 

We will test three different galaxy catalog conditions
1. LSST photometric limits on SkySim5000 catalog
1. Photo-z errors impact on H0 PE
1. Galaxy completeness effects

We also will test GW host catalog constraints, including
1. GW event localization completely/partially enclosed in galaxy survey footprint
2. Different CBC weighting schemes that have an impact on the underlying P(z) distribution
3. GW host galaxy inside/outside of galaxy survey footprint.

This project will have four deliverables
1. Deliver CBC host catalogs for testing methods.
    1. Information on this goal can be found in `./CBCCats/`
2. Validate dark siren statistical methods with simulated catalogs.
    1. Information on this goal can be found in `./initialValidation/`
3. Test different galaxy and GW host conditions.
    1. Information on this goal can be found in `./hostConditions/`
4. Deliver the first pipeline for integrating GW-cosmology to LSST catalogs.
