import numpy as np
import bilby as bb
from astropy.time import Time
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import healpy as hp
from astropy.cosmology.units import redshift_distance
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import astropy.cosmology.units as cu

def getHp_band_dict(hp_ids,lsst_bands,sigma,nside):
    """
    Construct per-Healpix, per-band limiting magnitude dictionaries.

    This function generates spatially varying limiting magnitudes across the sky
    by perturbing fiducial limiting magnitudes with a median-centered, negative
    log-normal random field, and then associates the resulting per-pixel limits
    with the Healpix indices actually present in a catalog.

    Parameters
    ----------
    hp_ids : array_like
        Array of Healpix pixel indices corresponding to objects in the catalog.
    lsst_bands : iterable of str
        LSST band identifiers (e.g., ``["u", "g", "r", "i", "z", "y"]``).
    sigma : float
        Log-normal scatter amplitude used to generate depth perturbations.
    nside : int
        Healpix NSIDE parameter defining sky map resolution.

    Returns
    -------
    dict
        Dictionary mapping each unique Healpix index to a dictionary of per-band
        limiting magnitudes, e.g. ``{hp_ind: {band: m_lim_band, ...}, ...}``.

    Notes
    -----
    This function assumes that a global dictionary ``limiting_mags`` exists in
    scope and provides the fiducial limiting magnitude for each band.
    """
    realistic_mags_hp_matched = {} # Instantiate the dictionary
    uniq_hp_ids = np.sort(np.unique(hp_ids) # Get the unique hp indices
    for band in lsst_bands: # iterate over bands
        realistic_mags = limiting_mags[band]+getMagLimitAdjustment(sigma,nside) # 
        realistic_mags_hp_matched[band] = realistic_mags[uniq_hp_ids]
    hp_band_dict = {} # Instantiate the dictionary
    count = 0 # Start count at zero
    for uniq_hp_id,count in zip(uniq_hp_ids,range(len(uniq_hp_ids))):
        band_mag_dict = {} # Instantiate the dictionary
        for band in lsst_bands: # iterate over bands
            band_mag_dict[band] = realistic_mags_hp_matched[band][count] # populate dict
        hp_band_dict[uniq_hp_id] = band_mag_dict # populate dict
        count+=1 # Iterate count
    return hp_band_dict # Return populated dictionary

def dropFaintGalaxies(data,uniq_hp_indices,lsst_bands,hp_band_dictionary,hp_ind_label="hp_ind_nside128"):
    """
    Apply spatially varying magnitude cuts to a galaxy catalog.

    This function removes objects from a catalog that are fainter than the
    local (Healpix-dependent) limiting magnitude in any LSST band. It enforces
    depth variations across the sky by comparing each object's true magnitude
    to the per-pixel limiting magnitudes stored in ``hp_band_dictionary``.

    Parameters
    ----------
    data : pandas.DataFrame
        Catalog containing true LSST-band magnitudes and Healpix pixel indices.
    uniq_hp_indices : array_like
        Unique Healpix indices to be processed.
    lsst_bands : iterable of str
        LSST band identifiers.
    hp_band_dictionary : dict
        Dictionary mapping Healpix indices to per-band limiting magnitudes,
        as produced by ``getHp_band_dict``.
    hp_ind_label : str, optional
        Column name in ``data`` containing Healpix indices. Default is
        ``"hp_ind_nside128"``.

    Returns
    -------
    pandas.DataFrame
        Filtered catalog with all objects fainter than the local depth limit
        removed.
    """
    for ind in uniq_hp_indices: # For each healpix index
        hp_data = data[data[hp_ind_label]==ind] # Subset of data, here for the healpix specified
        dropIndices = np.array([]) # Instantiate dropIndex array
        for band in lsst_bands: # Iterate over bands
            bool_index = hp_data[f"mag_true_{band}_lsst_no_host_extinction"] > hp_band_dictionary[ind][band] # Set the magnitude limit
            dropIndices = np.append(dropIndices,data[data[hp_ind_label]==ind][f"mag_true_{band}_lsst_no_host_extinction"][bool_index].index) # Append qualifying indices to array
        data = data.drop(dropIndices,axis=0) # Drop rows in dataframe
    return data

def RaDecToIndex(RA,decl,nside):
    """
    Convert equatorial coordinates to Healpix pixel indices.

    Parameters
    ----------
    RA : array_like
        Right ascension in degrees.
    decl : array_like
        Declination in degrees.
    nside : int
        Healpix NSIDE parameter.

    Returns
    -------
    ndarray
        Healpix pixel indices corresponding to the input coordinates.

    Notes
    -----
    This uses the HEALPix ``ang2pix`` convention with colatitude defined as
    ``theta = -decl + 90°`` and longitude ``phi = 360° - RA``.
    """
    return hp.ang2pix(nside,np.radians(-decl+90.),np.radians(360.-RA))

def getMagLimArr(nside,lim):
    """
    Create a full-sky Healpix map of constant limiting magnitude.

    Parameters
    ----------
    nside : int
        Healpix NSIDE parameter.
    lim : float
        Limiting magnitude to assign to every Healpix pixel.

    Returns
    -------
    ndarray
        Array of length ``hp.nside2npix(nside)`` filled with ``lim``.
    """
    return np.full(hp.nside2npix(nside),lim)

def getMagLimitAdjustment(sig,nside):
    """
    Generate a median-centered, negative log-normal magnitude perturbation map.

    This function produces a Healpix-resolution random field that models
    spatial variations in survey depth. The distribution is skewed toward
    negative values, representing localized depth degradation.

    Parameters
    ----------
    sig : float
        Log-normal scatter amplitude.
    nside : int
        Healpix NSIDE parameter.

    Returns
    -------
    ndarray
        Array of length ``hp.nside2npix(nside)`` containing per-pixel magnitude
        adjustments whose median is zero.
    """
    counts = -np.random.lognormal(mean=0,sigma=sig,size=hp.nside2npix(nside))
    counts -= np.median(counts)
    return counts

def transmute_redshift(inputRedshift,inputCosmology,alternate_h=0.5,z_max=5):
    """
    Re-map redshifts under an alternative Hubble constant via luminosity-distance
    invariance.

    This function transforms an array of input redshifts defined in an original
    cosmology into their corresponding redshifts in a new, flat ΛCDM cosmology
    with a modified Hubble constant. The transformation preserves luminosity
    distance: objects are assigned the redshift they would have in the new
    cosmology such that their luminosity distance remains unchanged.

    Parameters
    ----------
    inputRedshift : array_like
        Array of redshifts defined in the original cosmology.
    inputCosmology : astropy.cosmology.Cosmology
        The reference cosmology used to define the original redshift–distance
        relation.
    alternate_h : float, optional
        Dimensionless Hubble parameter ``h`` of the alternate cosmology,
        where ``H0 = 100 * h km s⁻¹ Mpc⁻¹``. Default is 0.5.

    Returns
    -------
    ndarray
        Array of redshifts in the alternate cosmology corresponding to the same
        luminosity distances as the input redshifts in the original cosmology.

    Notes
    -----
    Internally, the function computes luminosity distances in the alternate
    cosmology and then converts them back into redshifts using an equivalency
    based on ``cu.redshift_distance``. This effectively performs an implicit
    inversion of the distance–redshift relation while holding luminosity
    distance fixed.
    """
    inputRedshift_units = inputRedshift * cu.redshift
    d_L = inputCosmology.luminosity_distance(inputRedshift_units)
    
    newCosmologyParams = dict(inputCosmology.parameters)
    newCosmologyParams["H0"] = alternate_h* 100 * u.km/(u.Mpc * u.s)
    newCosmology = FlatLambdaCDM(name="skySimCopy",**newCosmologyParams)

    # newLuminosityDistances = newCosmology.luminosity_distance(inputRedshift_units)
    
    return d_L.to(cu.redshift, cu.redshift_distance(newCosmology, kind="luminosity",zmax=zmax))
    
    # newLuminosityDistances.to(cu.redshift, cu.redshift_distance(newCosmology, kind="luminosity", zmax=5))

def mag_log_normal_dist(sigma,num_pix):
    """
    Generate a median-centered negative log-normal random field.

    This function draws ``num_pix`` samples from a log-normal distribution
    with logarithmic standard deviation ``sigma``, negates the values, and
    shifts the distribution such that its median is exactly zero. The result
    is a skewed, strictly non-positive random field with a well-defined
    central reference level.

    Such distributions are useful for modeling magnitude-like perturbations,
    extinction fluctuations, or other asymmetric noise processes in which
    downward deviations are more probable than upward ones.

    Parameters
    ----------
    sigma : float
        Logarithmic standard deviation of the underlying log-normal
        distribution.
    num_pix : int
        Number of random samples (pixels) to generate.

    Returns
    -------
    counts : ndarray
        Array of length ``num_pix`` containing the median-centered,
        negative log-normal samples.
    min_val : float
        Minimum value of ``counts``.
    max_val : float
        Maximum value of ``counts``.

    Notes
    -----
    The distribution is constructed as:

    1. ``X ~ LogNormal(mean=0, sigma)``
    2. ``counts = -X``
    3. ``counts = counts - median(counts)``

    which guarantees that ``median(counts) = 0`` and that the distribution
    remains skewed toward negative values.
    """
    counts = -np.random.lognormal(mean=0,sigma=sigma,size=num_pix)
    counts -= np.median(counts)
    return counts,np.min(counts),np.max(counts)

def perBand_mag_distribution(LSST_bands,limitingMags,sigma,nside):
    """
    Generate per-band, per-pixel limiting magnitude maps with log-normal scatter.

    This function constructs HEALPix-resolution maps of limiting magnitude
    perturbations for each LSST band by drawing a median-centered, negative
    log-normal random field and adding it to a fiducial limiting magnitude.
    The result is a set of spatially varying depth maps that mimic realistic,
    skewed variations in survey depth across the sky.

    Parameters
    ----------
    LSST_bands : iterable of str
        Iterable of LSST band identifiers (e.g., ``["u", "g", "r", "i", "z", "y"]``).
    limitingMags : float or ndarray
        Fiducial limiting magnitude(s) to which the stochastic perturbations
        are added. If an array is provided, it must be broadcastable to the
        HEALPix pixel count corresponding to ``nside``.
    sigma : float
        Logarithmic standard deviation of the underlying log-normal distribution
        used to generate per-pixel magnitude perturbations.
    nside : int
        HEALPix NSIDE parameter defining the map resolution. The number of
        pixels is computed as ``hp.nside2npix(nside)``.

    Returns
    -------
    band_limit_arrs : dict
        Dictionary mapping each band to a NumPy array of length
        ``hp.nside2npix(nside)`` containing the per-pixel limiting magnitudes.
    band_limit_min_arrs : dict
        Dictionary mapping each band to the minimum perturbation applied in
        that band (i.e., the minimum value drawn from the log-normal field).
    band_limit_max_arrs : dict
        Dictionary mapping each band to the maximum perturbation applied in
        that band (i.e., the maximum value drawn from the log-normal field).

    Notes
    -----
    The per-pixel perturbations are generated by ``mag_log_normal_dist`` and
    are median-centered such that the median limiting magnitude in each band
    remains equal to the fiducial ``limitingMags``.
    """
    band_limit_arrs = {}
    band_limit_min_arrs = {}
    band_limit_max_arrs = {}
    num_pix = hp.nside2npix(nside)
    for band in LSST_bands:
        counts,min_counts,max_counts = mag_log_normal_dist(sigma,num_pix)
        band_limit_arrs[band] = limitingMags[band]+counts
        band_limit_min_arrs[band] = min_counts
        band_limit_max_arrs[band] = max_counts
    return band_limit_arrs,band_limit_min_arrs,band_limit_max_arrs

def trueZ_to_specZ(true_z,year):
    """
    Convert true redshifts to spectroscopic redshift realizations.

    This function is intended to generate mock spectroscopic redshift
    measurements by perturbing true redshifts with a Gaussian error model
    whose width decreases with survey duration. The scatter is assumed to
    scale as (1 + z) and improves with total observing time following a
    sqrt(time) relation.

    Parameters
    ----------
    true_z : array_like
        Array of true (cosmological) redshifts.
    year : float
        Effective survey duration in years used to scale the redshift
        uncertainty.

    Returns
    -------
    ndarray
        Array of spectroscopic redshift realizations.

    Notes
    -----
    This function is currently **not implemented** and will raise a
    ``ValueError`` when called. It exists as a placeholder for future
    spectroscopic error modeling.
    """
    time_term = np.sqrt(10/year)
    z_adjust = np.random.normal(loc=true_z,scale=prefactor*(1+true_z)*time_term,size=len(true_z))
    raise ValueError("Not yet implemented")
    return None

def trueZ_to_photoZ(true_z,year,modeled=False):
    """
    Convert true redshifts to photometric redshift realizations.

    This function generates mock photometric redshifts by applying Gaussian
    scatter to the true redshift distribution. The scatter scales with
    (1 + z) and improves with survey duration as sqrt(10 / year). Two noise
    regimes are supported: a nominal photometric error model and an idealized
    "modeled" case with reduced scatter.

    Parameters
    ----------
    true_z : array_like
        Array of true (cosmological) redshifts.
    year : float
        Effective survey duration in years.
    modeled : bool, optional
        If True, use a reduced photometric error model (prefactor = 0.01).
        If False, use a nominal LSST-like photometric scatter (prefactor = 0.04).

    Returns
    -------
    ndarray
        Array of photometric redshift realizations.
    """
    if modeled:
        prefactor=0.01
    else:
        prefactor=0.04
    time_term = np.sqrt(10/year)
    z_adjust = np.random.normal(loc=true_z,scale=prefactor*(1+true_z)*time_term,size=len(true_z))
    return z_adjust

def GCR_filter_overlord(year, LSST_bands,e_dict,v_dict,airmass=1.2,
                        z_max=1.2,z_min=None,mag_scatter=0.15,nside=128):
    """
    Construct a complete set of magnitude and redshift selection filters for
    GCRCatalog-style mock survey queries.

    This function orchestrates the construction of spatially varying magnitude
    depth maps, applies band-dependent limiting magnitude cuts, and appends
    global redshift constraints. The resulting filters can be directly passed
    to GCRCatalog query methods to emulate LSST-like survey selection effects.

    Parameters
    ----------
    year : float
        Effective survey duration in years.
    LSST_bands : iterable of str
        LSST band identifiers (e.g., ``["u", "g", "r", "i", "z", "y"]``).
    e_dict : dict
        Per-band exposure scaling factors.
    v_dict : dict
        Per-band visit or cadence scaling factors.
    airmass : float, optional
        Assumed observing airmass for depth calculations.
    z_max : float, optional
        Maximum allowed true redshift.
    z_min : float, optional
        Optional minimum allowed true redshift.
    mag_scatter : float, optional
        Log-normal scatter amplitude used to model spatial depth variations.
    nside : int, optional
        HEALPix NSIDE parameter defining sky map resolution.

    Returns
    -------
    filters : list of str
        List of GCR-compatible filter expressions.
    band_limit_dict : dict
        Dictionary mapping each band to its per-pixel limiting magnitude map.
    """
    
    # The magnitude limiting cut
    limiting_mags = GCR_mag_filter_from_year(year, LSST_bands,e_dict,v_dict,airmass)

    band_limit_dict,band_limit_min_dict,band_limit_max_dict = perBand_mag_distribution(LSST_bands,limiting_mags,mag_scatter,nside)
    
    # Add the magnitude limiting filter
    filters = []
    for band in LSST_bands:
        filters.append(f"mag_true_{band}_lsst_no_host_extinction<{limiting_mags[band]+band_limit_max_dict[band]}") # limiting_mags has been swapped for band_limit_max_dict here

    # Add the redshift filter
    filters = GCR_redshift_filter(filters,z_max,z_min)

    return filters,band_limit_dict

def GCR_redshift_filter(filt,z_max,z_min=None):
    """
    Append redshift selection cuts to an existing filter list.

    Parameters
    ----------
    filt : list of str
        Existing list of GCR-compatible filter expressions.
    z_max : float
        Maximum allowed true redshift.
    z_min : float, optional
        Minimum allowed true redshift.

    Returns
    -------
    list of str
        Updated filter list including the redshift cuts.
    """
    filt.append(f"redshift_true<{z_max}")
    if z_min!=None:
        filt.append(f"redshift_true>{z_min}")
    return filt

def GCR_mag_filter_from_year(year, LSST_bands,e_dict,v_dict,airmass=1.2):
    """
    Compute per-band limiting magnitudes for an LSST-like survey realization.

    This function calculates band-dependent limiting magnitudes using the
    LSST photometric depth prescription, scaling the effective exposure time
    by survey duration and per-band cadence/exposure factors.

    Parameters
    ----------
    year : float
        Effective survey duration in years.
    LSST_bands : iterable of str
        Iterable of LSST band identifiers.
    e_dict : dict
        Dictionary mapping each band to its exposure scaling factor.
    v_dict : dict
        Dictionary mapping each band to its visit or cadence scaling factor.
    airmass : float, optional
        Assumed observing airmass.

    Returns
    -------
    dict
        Dictionary mapping each band to its limiting magnitude.
    """
    limiting_mags = {}
    for band in LSST_bands:
        C_m,m_sky,theta_eff,k_m = getLSSTBandParameters(band)
        limiting_mags[band] = LSST_mag_lim(C_m,m_sky,theta_eff,e_dict[band]*v_dict[band]*year,k_m,airmass)
    
    return limiting_mags

def LSST_mag_lim(C_m,m_sky,theta_eff,t_vis,k_m,X):
    '''
    C_m is the band dependent parameter
    m_sky is the sky brightness (AB mag arcsec−2)
    theta_eff is the seeing (in arcseconds)
    t_vis is the exposure time (seconds)
    k_m is the atmospheric extinction coefficient
    X is air mass
    '''
    return C_m + 0.5 * (m_sky-21) + 2.5*np.log10(0.7/theta_eff)+1.25*np.log10(t_vis/30)-k_m*(X-1)

def getLSSTBandParameters(band):
    # Band dict in the form of band: [C_m,m_sky,theta_eff,k_m]
    # From eq. 6 of Ivecic 2019
    bandDict = {"u":[23.09,22.99,0.92,0.491],
                "g":[24.42,22.26,0.87,0.213],
                "r":[24.44,21.20,0.83,0.126],
                "i":[24.32,20.48,0.80,0.096],
                "z":[24.16,19.60,0.78,0.069],
                "y":[23.73,18.61,0.76,0.170],
               }
    return bandDict[band.lower()]

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
        raise ValueError("Invalid source model '{}'. Choose from {}.".format(source_model_name,list(source_models.keys())))

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
    return "{}_{}".format(base_dir,next_number)

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
        raise ValueError("Invalid merger type '{}'. Choose from {}.".format(merger_type,list(merger_priors.keys())))

    return merger_priors[merger_type]()

    