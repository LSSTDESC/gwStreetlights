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
from scipy.optimize import curve_fit
from astropy.modeling.models import Schechter1D
from scipy.interpolate import CubicSpline
import datetime

LSST_bands = ["u", "g", "r", "i", "z", "Y"]
visits_per_yr = (
    np.array([52, 71, 174, 178, 154, 149]) / 10
)  # visits per year in u-g-r-i-z-y, in the WFD area, from the latest opsim run (https://usdf-maf.slac.stanford.edu/allMetricResults?runId=2#Nvisits%20Maps_WFD)
bandColors = ["violet", "blue", "forestgreen", "lime", "orange", "red"]
band_color = {}
for band, col in zip(LSST_bands, bandColors):
    band_color[band] = col

saturation = [14.7, 15.7, 15.8, 15.8, 15.3, 13.9]  # CCD saturation in mags, per band
band_saturation = {}
for band, sat in zip(LSST_bands, saturation):
    band_saturation[band] = sat

visits_dict = {}
for band, vis in zip(LSST_bands, visits_per_yr):
    visits_dict[band] = vis

expTimes = [38, 30, 30, 30, 30, 30]

eTime_dict = {}
for band, eTime in zip(LSST_bands, expTimes):
    eTime_dict[band] = eTime


def schecterEvolutionPlot(res, tight=True, save=False, fname=None):
    """
    Plot the redshift evolution of Schechter luminosity–function parameters.

    This function visualizes the evolution of the three Schechter parameters
    (phi*, M*, and alpha) as a function of redshift for each LSST band. The input
    is expected to be the output dictionary produced by the luminosity–function
    fitting routine, with entries indexed by redshift bin and band.

    For each LSST band, the median redshift of each fitted redshift bin is used
    as the x–coordinate, and the corresponding fitted Schechter parameters are
    plotted as connected points.

    Parameters
    ----------
    res : dict
        Dictionary of fitted Schechter parameters. Keys must be tuples of the
        form ``(z_min, z_max, band)``, and values must be dictionaries containing
        the entries ``"phi_star"``, ``"M_star"``, and ``"alpha"``.

    tight : bool, optional
        If True (default), apply ``fig.tight_layout()`` to optimize subplot
        spacing.

    save : bool, optional
        If True, save the figure to disk using ``fname``. Default is False.

    fname : str or None, optional
        Output filename used when ``save=True``. Ignored if ``save=False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plots.

    axs : ndarray of matplotlib.axes.Axes
        Array of Axes objects with shape (3,), corresponding to the plots of
        ``phi*``, ``M*``, and ``alpha`` as functions of redshift.

    Notes
    -----
    The three panels show:

    1. Top: normalization ``phi*`` as a function of redshift.
    2. Middle: characteristic magnitude ``M*`` as a function of redshift.
    3. Bottom: faint–end slope ``alpha`` as a function of redshift.

    Colors and labels are assigned per LSST band using the global
    ``band_color`` mapping.
    """
    fig, axs = plt.subplots(3, figsize=[6, 4 * 3])
    for b in LSST_bands:
        alpha_arr, M_star_arr, phi_star_arr, redshift_arr = [], [], [], []
        for k in res.keys():
            lum_bin = np.median([k[0], k[1]])
            band = k[2]
            if k[2] == b:
                phi_star_arr.append(res[k]["phi_star"])
                M_star_arr.append(res[k]["M_star"])
                alpha_arr.append(res[k]["alpha"])
                redshift_arr.append(np.median([k[0], k[1]]))

        myKwargs = {"label": f"{b} band", "marker": "o", "color": band_color[b]}

        axs[0].plot(redshift_arr, phi_star_arr, **myKwargs)
        axs[1].plot(redshift_arr, M_star_arr, **myKwargs)
        axs[2].plot(redshift_arr, alpha_arr, **myKwargs)

    axs[-1].set_xlabel("$z$")
    axs[0].legend(ncols=2)
    axs[0].set_ylabel(r"$\phi* \times 10^{-2} [\text{Mpc}^{-3} \text{ h}^3]$")
    axs[1].set_ylabel(r"$M* - 5 \log(\text{h}) [\text{mag}]$")
    axs[2].set_ylabel(r"$\alpha$")

    if tight:
        fig.tight_layout()
    if save:
        fig.savefig(fname)
    return fig, axs


def run_survey_diagnostics(
    data,
    hp_band_dict,
    skysim_catalog,
    year,
    z_max,
    limiting_mags,
    z_min=0,
    z_step_pz=0.01,
    z_step_lf=0.2,
    brightMag=-26,
    faintMag=-15,
    fit_schecter=False,
    p0=(1e-3, -22.0, -1.1),
    maxfev=50000,
    modeled=True,
    hi_mag=27.3,
    low_mag=24.5,
    NSIDE=None,
    delta_mag_schecter=0.2,
    vmax=True,
    spectroscopic=False,
    verbose=True,
):
    """
    Run the standard diagnostics suite on a processed SkySim catalog.

    This function generates the core set of validation plots used to assess
    redshift performance, luminosity function recovery, and spatial magnitude
    non–uniformity for a simulated LSST galaxy sample.

    The diagnostics include:

      1. The redshift probability distribution P(z).
      2. Schechter luminosity function fits in LSST bands over redshift slices.
      3. Photometric redshift precision relative to model expectations.
      4. Healpix–based magnitude non–uniformity maps.
      5. A boolean sanity check verifying that each healpix pixel satisfies
         the imposed limiting magnitude cuts.

    Parameters
    ----------
    data : pandas.DataFrame
        LSST–like galaxy catalog produced by the forward–modeling pipeline.
        Must include photometric redshifts, true magnitudes, and healpix indices.

    hp_band_dict : dict
        Spatial depth–uniformity map produced by the survey selection stage.

    skysim_catalog : GCRCatalogs.cosmodc2.SkySim5000GalaxyCatalog
        GCRCatalog instance associated with the input data.

    year : int
        LSST survey year used for the photo-z model.

    z_max : float
        Maximum redshift used for binning and plotting.

    z_step : float, optional
        Redshift bin width for P(z) and luminosity–function plots. Default is 0.2.

    brightMag, faintMag : float, optional
        Magnitude range used for luminosity–function binning.

    p0 : tuple, optional
        Initial guess for Schechter parameters (phi*, M*, alpha).

    maxfev : int, optional
        Maximum number of function evaluations in LF fitting.

    modeled : bool, optional
        Whether to overlay modeled redshift precision expectations.

    hi_mag, low_mag : float, optional
        Color scale bounds for magnitude non–uniformity healpix maps.

    NSIDE : int, optional
        Healpix NSIDE used for magnitude–limit consistency checking.

    verbose : bool, optional
        Determine if verbose logging during plot generation is used

    Returns
    -------
    results : dict
        Dictionary of fitted Schechter parameters indexed by (z_low, z_high, band).

    mag_lim_check_result : dict
        Output of ``mag_lim_checker`` verifying per–pixel magnitude–limit compliance.
    """

    def log(msg):
        if verbose:
            print(
                f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - survey-diagnostics] {msg}"
            )

    log("Starting survey diagnostics")
    log(f"Catalog size: {len(data):,} objects")
    log(f"Redshift range: z = [{z_min}, {z_max}]")
    log(f"Luminosity-function binning: Δz = {z_step_lf}")
    log(f"P(z) binning: Δz = {z_step_pz}")
    log(f"Using Vmax method: {vmax}")
    log(f"Fitting Schechter function: {fit_schecter}")

    # ------------------------------------------------------------------
    # Luminosity functions
    # ------------------------------------------------------------------
    log("Computing luminosity functions")

    fig_lf, axs_lf, results = luminosityFunction(
        data,
        skysim_catalog,
        limiting_mags,
        p0=p0,
        z_min=z_min,
        z_max=z_max,
        z_step=z_step_lf,
        brightMag=brightMag,
        faintMag=faintMag,
        maxfev=maxfev,
        delta_mag=delta_mag_schecter,
        fit_schecter=fit_schecter,
        use_vmax=vmax,
    )

    log(f"Luminosity function computed for {len(results)} (z-bin, band) entries")

    log("Generating Schechter parameter evolution plot")
    fig_schec_evo, axs_schec_evo = schecterEvolutionPlot(results)

    # ------------------------------------------------------------------
    # P(z)
    # ------------------------------------------------------------------
    log("Computing redshift probability distribution P(z)")
    fig_pz, ax_pz = p_z_distribution(data, z_max=z_max, z_step=z_step_pz)

    # ------------------------------------------------------------------
    # Redshift precision
    # ------------------------------------------------------------------
    log("Computing photometric redshift precision")
    log(f"Modeled comparison enabled: {modeled}")
    log(f"Spectroscopic mode: {spectroscopic}")

    fig_zprec, ax_zprec = redshiftPrecisionPlot(
        data,
        year,
        modeled=modeled,
        spec=spectroscopic,
    )

    # ------------------------------------------------------------------
    # Magnitude non–uniformity
    # ------------------------------------------------------------------
    log("Computing magnitude non-uniformity maps")
    log(f"Color scale limits: [{low_mag}, {hi_mag}]")

    fig_uni, ax_uni = mag_uniformity_plot(
        hp_band_dict,
        limiting_mags,
        hi_mag=hi_mag,
        low_mag=low_mag,
    )

    # ------------------------------------------------------------------
    # Magnitude–limit consistency check
    # ------------------------------------------------------------------
    mag_lim_check_result = None
    if NSIDE is not None:
        log(f"Running magnitude-limit consistency check (NSIDE={NSIDE})")
        mag_lim_check_result = mag_lim_checker(data, hp_band_dict, NSIDE)
        log("Magnitude-limit consistency check complete")
    else:
        log("Skipping magnitude-limit consistency check (NSIDE=None)")

    figs = [
        fig_lf,
        fig_schec_evo,
        fig_pz,
        fig_zprec,
        fig_uni,
    ]
    axes = [
        axs_lf,
        axs_schec_evo,
        ax_pz,
        ax_zprec,
        ax_uni,
    ]

    log("Survey diagnostics complete")

    return results, mag_lim_check_result, figs, axes


def apply_lsst_depth_and_uniformity(
    skysimCat,
    year,
    dataColumns,
    airmass,
    z_max,
    NSIDE,
    z_min=0,
    mags_deeper=1,
    uniformity=0.1,
    modeled=False,
    spectroscopic=False,
    alternate_h=None,
    verbose=False,
):
    """
    Apply LSST survey depth, spatial uniformity, and photometric–redshift effects
    to a SkySim catalog, returning a depth–limited, non-uniform, photo-z–perturbed
    galaxy sample.

    This routine performs the full observational selection pipeline used in LSST–
    like forward modeling:

      1. Computes band–dependent limiting magnitudes for the specified survey year.
      2. Optionally deepens the survey depth by a uniform magnitude offset
         (e.g. to emulate deeper–than–nominal coadds).
      3. Constructs sky–position–dependent depth filters using GCR catalogs.
      4. Queries the SkySim catalog under those selection filters.
      5. Assigns each object to a HEALPix pixel at the specified NSIDE.
      6. Applies a spatial uniformity map that enforces non-uniform limiting depths.
      7. Converts true redshifts into measured photometric redshifts.

    The output catalog therefore represents a realistic, non-uniform LSST
    observational sample suitable for luminosity function, clustering, and
    selection–function analyses.

    Parameters
    ----------
    skysimCat : GCRCatalog
        SkySim catalog object supporting ``get_quantities`` queries.

    year : int
        Survey year (e.g. 1–10) used to determine LSST depth and photo-z performance.

    dataColumns : sequence of str
        Columns to request from the SkySim catalog.

    airmass : float
        Airmass used when computing limiting magnitudes.

    z_max : float
        Maximum redshift used in the GCR depth filters.

    NSIDE : int
        HEALPix NSIDE used for spatial depth uniformity modeling.

    mags_deeper : float, optional
        Uniform magnitude offset applied to all limiting magnitudes
        (positive values make the survey deeper). Default is 0.

    uniformity : float, optional
        Amplitude of the spatial non-uniformity model. Default is 0 (uniform survey).

    modeled : bool, optional
        Whether or not the photo-z's are based on modeled objects in the spectroscopic sample

    verbose : bool, optional
        If true, will add verbose logging during data generation.

    Returns
    -------
    data : pandas.DataFrame
        Catalog of galaxies after applying depth cuts, spatial uniformity,
        and photometric redshift perturbations. Includes an added column:

        ``hp_ind_nside{NSIDE}`` — HEALPix pixel index for each object
        ``redshift_measured`` — photometric redshift

    limiting_mags : dict
        Dictionary mapping LSST band → limiting magnitude after applying
        ``mags_deeper``.

    hp_band_dict : dict
        Spatial uniformity map describing band-dependent depth variations per
        HEALPix pixel.
    """

    def log(msg):
        if verbose:
            print(
                f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - lsst-depth-uniformity] {msg}"
            )

    log("Starting LSST depth + uniformity application")
    log(f"Survey year: {year}")
    log(f"Redshift range: z ∈ [{z_min}, {z_max}]")
    log(f"NSIDE: {NSIDE}")
    log(f"Airmass: {airmass}")
    log(f"Uniformity amplitude: {uniformity}")
    log(f"Magnitude deepening: {mags_deeper} mag")
    log(f"Spectroscopic mode: {spectroscopic}")
    log(f"Modeled photo-zs: {modeled}")

    # ------------------------------------------------------------------
    # Compute limiting magnitudes
    # ------------------------------------------------------------------
    log("Computing band-dependent limiting magnitudes")

    limiting_mags = GCR_mag_filter_from_year(
        year, LSST_bands, eTime_dict, visits_dict, airmass=airmass
    )

    log(
        "Survey limiting magnitudes: "
        + ", ".join(f"{b}={limiting_mags[b]:.2f}" for b in LSST_bands)
    )

    deeper_mags = {}
    for b in LSST_bands:
        deeper_mags[b] = limiting_mags[b] + mags_deeper

    log(
        "Limiting magnitudes after deepening: "
        + ", ".join(f"{b}={deeper_mags[b]:.2f}" for b in LSST_bands)
    )

    # ------------------------------------------------------------------
    # Build GCR depth filters
    # ------------------------------------------------------------------
    log("Constructing GCR depth and spatial-uniformity filters")

    filters, band_limit_dict = GCR_filter_overlord(
        year,
        LSST_bands,
        # eTime_dict,
        # visits_dict,
        deeper_mags,
        airmass=airmass,
        z_max=z_max,
        z_min=z_min,
        mag_scatter=uniformity,
        nside=NSIDE,
    )

    log(f"Number of GCR filters constructed: {len(filters)}")
    log(f"GCR filters constructed: {filters}")

    # ------------------------------------------------------------------
    # Query SkySim catalog
    # ------------------------------------------------------------------
    log("Querying SkySim catalog")

    data = pd.DataFrame(
        skysimCat.get_quantities(list(dataColumns), filters=tuple(filters))
    )

    log(f"Catalog size after GCR query: {len(data):,} objects")

    # ------------------------------------------------------------------
    # Assign HEALPix indices
    # ------------------------------------------------------------------
    hp_col = f"hp_ind_nside{NSIDE}"
    log(f"Assigning HEALPix indices ({hp_col})")

    data[hp_col] = RaDecToIndex(data["ra_true"], data["dec_true"], NSIDE).astype(
        np.int32
    )

    hp_uniq_ids = np.unique(data[hp_col])
    log(f"Number of populated HEALPix pixels: {len(hp_uniq_ids)}")

    # ------------------------------------------------------------------
    # Build spatial uniformity map
    # ------------------------------------------------------------------
    log("Building spatial uniformity depth map")

    hp_band_dict = getHp_band_dict(
        hp_uniq_ids, LSST_bands, uniformity, NSIDE, limiting_mags
    )

    # ------------------------------------------------------------------
    # Apply non-uniform depth cuts
    # ------------------------------------------------------------------
    n_before = len(data)
    log("Applying non-uniform depth cuts")

    data = dropFaintGalaxies(
        data, hp_uniq_ids, LSST_bands, hp_band_dict, hp_ind_label=hp_col
    )

    log(
        f"Depth cuts applied: {n_before:,} → {len(data):,} objects "
        f"({len(data)/n_before:.2%} retained)"
    )

    # ------------------------------------------------------------------
    # Optional cosmology rescaling
    # ------------------------------------------------------------------
    if alternate_h is not None:
        log(
            f"Rescaling true redshifts for alternate h: "
            f"{alternate_h} (SkySim h={skysimCat.cosmology.h})"
        )
        data["redshift_true"] *= alternate_h / skysimCat.cosmology.h

    # ------------------------------------------------------------------
    # Apply redshift model
    # ------------------------------------------------------------------
    log("Applying redshift measurement model")

    if spectroscopic:
        log("Using spectroscopic redshift model")
        data["redshift_measured"] = trueZ_to_specZ(data["redshift_true"], year)
    else:
        log("Using photometric redshift model")
        data["redshift_measured"] = trueZ_to_photoZ(
            data["redshift_true"], year, modeled=modeled
        )

    log("LSST depth + uniformity application complete")

    return data, limiting_mags, hp_band_dict


def mag_uniformity_plot(
    hp_band_dictionary,
    hp_lim_mags,
    nside=128,
    low_mag=24.5,
    hi_mag=27.8,
    mag_step=0.05,
    tight=True,
    save=False,
    fname="magnitudeUniformity.jpg",
):
    """
    Plot the distribution of limiting magnitudes across healpix sky pixels for
    each LSST photometric band.

    This function visualizes the spatial uniformity of survey depth by producing
    histograms of per–pixel limiting magnitudes stored in a healpix-indexed
    dictionary. A separate histogram is overlaid for each LSST band.

    Parameters
    ----------
    hp_band_dictionary : dict
        Dictionary keyed by healpix pixel index, where each value is a mapping
        ``{band: limiting_magnitude}`` for LSST bands (u, g, r, i, z, y).

    nside : int, optional
        Healpix NSIDE resolution parameter used to define the sky pixelization.
        Default is 128.

    low_mag : float, optional
        Lower bound of the magnitude histogram range. Default is 24.5.

    hi_mag : float, optional
        Upper bound of the magnitude histogram range. Default is 27.8.

    mag_step : float, optional
        Bin width of the magnitude histograms. Default is 0.05.

    tight : bool, optional
        If True, apply ``fig.tight_layout()`` before displaying or saving the
        figure. Default is True.

    save : bool, optional
        If True, save the figure to disk using ``fname``. Default is False.

    fname : str, optional
        Output filename for the saved figure. Default is
        ``"magnitudeUniformity.jpg"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib Figure object.

    ax : matplotlib.axes.Axes
        The Axes object containing the histogram plot.
    """
    fig, ax = plt.subplots()
    for band in LSST_bands:
        limits = []
        for ind in hp_band_dictionary.keys():
            limits.append(hp_band_dictionary[ind][band])
        ax.hist(
            limits,
            histtype="step",
            density=False,
            bins=np.arange(low_mag, hi_mag, step=mag_step),
            label=f"{band}",
            color=band_color[band],
        )
        ax.axvline(np.median(limits), alpha=0.5, color=band_color[band], ls="--")
        ax.axvline(hp_lim_mags[band], alpha=0.5, color=band_color[band], ls="-.")
    ax.legend()

    ax.set_xlabel(f"Limiting magnitude in each NSIDE={nside} healpix pixel")
    ax.set_ylabel("Counts / bin")
    if tight:
        fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(os.path.join(os.getcwd(), fname), dpi=200)

    return fig, ax


def mag_lim_checker(inputData, hp_band_dictionary, nside):
    """
    Validate that all objects in a catalog satisfy healpix–dependent magnitude
    limits in each LSST band.

    For each healpix pixel, this routine checks whether any object in the
    corresponding subset of ``inputData`` exceeds the limiting magnitude defined
    in ``hp_band_dictionary``. If any violation is found, a ``ValueError`` is
    raised.

    Parameters
    ----------
    inputData : pandas.DataFrame
        Catalog containing object magnitudes and a healpix index column named
        ``"hp_ind_nside{nside}"``.

    hp_band_dictionary : dict
        Dictionary keyed by healpix pixel index. Each value is a mapping
        ``{band: limiting_magnitude}`` for LSST bands.

    nside : int
        Healpix NSIDE parameter defining the sky pixelization.

    Returns
    -------
    int
        Returns 0 if all objects satisfy the magnitude limits.

    Raises
    ------
    ValueError
        If any object violates the magnitude limit in any band for its healpix
        pixel.
    """
    for hp, band_lim in zip(hp_band_dictionary.keys(), hp_band_dictionary.values()):
        subData = inputData[inputData[f"hp_ind_nside{nside}"] == hp]
        for band, lim in zip(band_lim.keys(), band_lim.values()):
            msk = subData[f"mag_true_{band}_lsst_no_host_extinction"] > lim
            if msk.any():
                raise ValueError(
                    f"Healpix index {hp} fails magnitude limit for band {band}"
                )
                return -1
    print("Dataframe passes magnitude limit criteria")
    return 0


def schechter_M(M, phi_star, M_star, alpha):
    """
    Evaluate the Schechter luminosity function in absolute–magnitude form.

    Implements:

    .. math::

        \\Phi(M) = 0.4\\ln(10) \\, \\phi_* \\, 10^{0.4(\\alpha+1)(M_* - M)}
        \\, \\exp\\left[-10^{0.4(M_* - M)}\\right]

    Parameters
    ----------
    M : array_like
        Absolute magnitudes.

    phi_star : float
        Normalization parameter :math:`\\phi_*` (Mpc⁻³).

    M_star : float
        Characteristic magnitude :math:`M_*`.

    alpha : float
        Faint–end slope parameter :math:`\\alpha`.

    Returns
    -------
    array_like
        Value of the Schechter luminosity function :math:`\\Phi(M)` in units of
        Mpc⁻³ mag⁻¹.
    """
    model = Schechter1D(phi_star, M_star, alpha)
    return model(M)


def p_z_distribution(
    data, z_step=0.05, z_max=1.8, tight=True, save=False, fname="P_Z_distribution.jpg"
):
    """
    Plot histograms of true and measured redshift distributions.

    This routine overlays histograms of the true and measured redshifts from a
    galaxy catalog, allowing direct visual comparison of photometric redshift
    performance.

    Parameters
    ----------
    data : pandas.DataFrame
        Catalog containing ``"redshift_true"`` and ``"redshift_measured"`` columns.

    z_step : float, optional
        Width of the redshift bins. Default is 0.05.

    z_max : float, optional
        Maximum redshift shown in the histogram. Default is 1.8.

    tight : bool, optional
        If True, apply ``fig.tight_layout()``. Default is True.

    save : bool, optional
        If True, save the figure to disk. Default is False.

    fname : str, optional
        Output filename for the saved figure. Default is
        ``"P_Z_distribution.jpg"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib Figure.

    ax : matplotlib.axes.Axes
        The Axes object containing the histograms.
    """
    fig, ax = plt.subplots()

    # Plot the true redshift
    ax.hist(
        data["redshift_true"],
        histtype="step",
        bins=np.arange(0, z_max + z_step, step=z_step),
        label="True redshift",
    )

    # Plot the measured redshift
    ax.hist(
        data["redshift_measured"],
        histtype="step",
        bins=np.arange(0, z_max + z_step, step=z_step),
        label="Measured redshift",
    )

    ax.set_xlabel("$z$")
    ax.set_ylabel("$N_{gals}$ / bin")
    ax.legend()
    if tight:
        fig.tight_layout()
    if save:
        fig.savefig(os.path.join(os.getcwd(), fname), dpi=200)

    return fig, ax


def redshiftPrecisionPlot(
    dat,
    yr,
    zmin=0,
    zmax=1.5,
    z_step=0.05,
    spec=False,
    modeled=True,
    tight=True,
    save=False,
    fname="RedshiftPrecision.jpg",
):
    """
    Plot predicted and measured photometric redshift precision as a function of redshift.

    This routine compares an analytic model for photometric redshift precision
    with the measured scatter in ``|z_measured - z_true|`` computed from a galaxy
    catalog, binned in redshift.

    Parameters
    ----------
    dat : pandas.DataFrame
        Catalog containing ``"redshift_true"`` and ``"redshift_measured"`` columns.

    yr : int or float
        Survey year used to scale the modeled redshift precision.

    zmin : float, optional
        Minimum redshift to include. Default is 0.

    zmax : float, optional
        Maximum redshift to include. Default is 1.5.

    z_step : float, optional
        Width of the redshift bins. Default is 0.05.

    spec : bool, optional
        If True, indicate spectroscopic precision. Currently not implemented.
        Default is False.

    modeled : bool, optional
        If True, use the modeled photometric precision prefactor; otherwise use
        a conservative empirical prefactor. Default is True.

    tight : bool, optional
        If True, apply ``fig.tight_layout()``. Default is True.

    save : bool, optional
        If True, save the figure to disk. Default is False.

    fname : str, optional
        Output filename for the saved figure. Default is
        ``"RedshiftPrecision.jpg"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib Figure.

    ax : matplotlib.axes.Axes
        The Axes object containing the precision curves.

    Raises
    ------
    ValueError
        If ``spec=True`` is requested (not yet implemented).
    """

    def getPrefactor(modeled=False):
        if modeled:
            return 0.01
        else:
            return 0.04

    def yearAdjustment(yr):
        return np.sqrt(10 / yr)

    fig, ax = plt.subplots()

    yr_adjust = yearAdjustment(yr)

    prefactor = getPrefactor(modeled)
    if spec:
        prefactor = 0.0004
        yr_adjust = 1  # Time independent version

    z_arr = np.arange(zmin, zmax + z_step, step=z_step)

    predicted = (1 + z_arr) * yr_adjust * prefactor

    measured, _95, _5 = [], [], []

    for z in z_arr:
        msk = np.logical_and(
            dat["redshift_true"] < z + z_step, dat["redshift_true"] > z
        )
        difference = dat[msk]["redshift_measured"] - dat[msk]["redshift_true"]
        measured_val = np.std(difference)
        # _95_val, _5_val = np.nanpercentile(difference, (68, 32))

        measured.append(measured_val)
        # _95.append(_95_val)
        # _5.append(_5_val)

    ax.step(z_arr, predicted, where="post", color="red")
    ax.scatter(
        z_arr + z_step / 2,
        predicted,
        marker="o",
        color="red",
        label="Predicted precision",
    )

    ax.step(z_arr, measured, where="post", color="blue")
    ax.scatter(
        z_arr + z_step / 2,
        measured,
        marker="o",
        color="blue",
        label="Measured precision",
    )

    ax.legend()
    ax.set_xlabel("$z$")
    ax.set_ylabel("$z_{measured} - z_{true}$")

    if tight:
        fig.tight_layout()
    if save:
        fig.savefig(fname, dpi=300)

    return fig, ax


def luminosity_distance_from_mags(appMag, absMag, out_unit=u.pc):
    """
    Compute luminosity distance from apparent and absolute magnitudes.

    Uses the standard distance–modulus relation

    .. math::
        m - M = 5 \\log_{10}(d_L / 10\\,\\mathrm{pc})

    to convert an apparent magnitude ``appMag`` and absolute magnitude
    ``absMag`` into a luminosity distance.

    Parameters
    ----------
    appMag : float or array-like
        Apparent magnitude(s) of the object(s).

    absMag : float or array-like
        Absolute magnitude(s) of the object(s).

    Returns
    -------
    dL : astropy.units.Quantity
        Luminosity distance(s) in parsecs.
    """
    distance_modulus = appMag - absMag
    lum_dist = 10 ** (distance_modulus / 5 + 1)

    return lum_dist * u.pc.to(out_unit)

    # return (
    #     (10 * (w1 / 1000000)) * 10**((appMag - absMag) / 5)
    # )  # To convert to Mpc


def phi_VMax_eales(
    omega_area,
    splines,
    absolute_mag,
    zmin,
    zmax,
    dimMag,
    brightMag,
    lum_bin_edges,
):
    """
    Compute the binned luminosity function using the V_max estimator.

    This routine applies the Eales V_max method to estimate the luminosity
    function

    .. math::
        \\phi(M) = \\sum_i \\frac{1}{V_{\\max,i}}

    within bins of absolute magnitude.

    Parameters
    ----------
    omega_area : float
        Survey solid angle in steradians.

    splines : list of scipy.interpolate.CubicSpline
        Spline interpolants returned by ``prep_vmax_quantities``, in the order:
        ``[dl_to_z, z_to_dc, z_to_dl, z_to_vc]``.

    absolute_mag : array-like
        Absolute magnitudes of galaxies in the sample.

    zmin, zmax : float
        Redshift bounds of the survey or redshift slice.

    dimMag : float
        Faint apparent–magnitude limit of the survey.

    brightMag : float
        Bright apparent–magnitude limit of the survey.

    lum_bin_edges : dict
        Dictionary mapping ``(M_low, M_high)`` bins to galaxy counts.

    Returns
    -------
    phi_array : numpy.ndarray
        Estimated luminosity function values for each magnitude bin,
        in units of Mpc⁻³ mag⁻¹ (up to factors of ``h``).
    """
    dl_to_z, z_to_dc, z_to_dl, z_to_vc = splines

    vmax_arr, msk = VMax_eales(
        absolute_mag, dimMag, brightMag, dl_to_z, z_to_vc, omega_area, zmin, zmax
    )

    abs_mag_masked = absolute_mag[msk]
    vmax_masked = vmax_arr[msk]

    phi_array = []

    for v in lum_bin_edges.keys():
        mag_mask = (abs_mag_masked >= v[0]) & (abs_mag_masked < v[1])

        binned_vmax = vmax_masked[mag_mask]
        if np.shape([]) == np.shape(
            binned_vmax
        ):  # If there are no galaxies in that luminosity bin
            phi_array.append(0)  # Append zero to the phi array
        else:
            phi_array.append(np.sum(np.reciprocal(binned_vmax)) / (v[1] - v[0]))

    return phi_array


def VMax_eales(
    abs_mag,
    app_mag_faint,
    app_mag_bright,
    dl_to_z,
    z_to_vc,
    omega_area,
    z_1,
    z_2,
    n_centers=1,
):
    """
    Compute the accessible comoving volume V_max for galaxies using the
    Eales (1993) formalism.

    For each galaxy of absolute magnitude ``abs_mag``, this function computes
    the maximum and minimum redshifts at which the object would still be
    observable within the survey apparent–magnitude limits, and evaluates
    the corresponding accessible comoving volume:

    .. math::
        V_{\\max} = \\frac{\\Omega}{n_{\\rm centers}}
        \\left[V_c(z_{\\rm max}) - V_c(z_{\\rm min})\\right]

    where :math:`\\Omega` is the survey solid angle.

    Parameters
    ----------
    abs_mag : array-like
        Absolute magnitudes of the galaxies.

    app_mag_faint : float
        Faint apparent–magnitude limit of the survey.

    app_mag_bright : float
        Bright apparent–magnitude limit of the survey.

    dl_to_z : scipy.interpolate.CubicSpline
        Spline mapping luminosity distance → redshift.

    z_to_vc : scipy.interpolate.CubicSpline
        Spline mapping redshift → enclosed comoving volume.

    omega_area : float
        Survey solid angle in steradians.

    n_centers : int, optional
        Number of survey centers used to tile the sky. Default is ``1``.

    Returns
    -------
    vmax : numpy.ndarray
        Accessible comoving volume for each galaxy, in Mpc³.
    """
    dl_for_faint = luminosity_distance_from_mags(
        app_mag_bright, abs_mag, out_unit=u.Mpc
    )
    z_faints = dl_to_z(
        dl_for_faint
    )  # The z limits for the faint galaxy to be placed at the bright end limit

    dl_for_bright = luminosity_distance_from_mags(
        app_mag_faint, abs_mag, out_unit=u.Mpc
    )
    z_brights = dl_to_z(
        dl_for_bright
    )  # The z limits for the bright galaxy to be placed at the faint end limit

    msk1_bright = (
        z_brights > z_2
    )  # If bright galaxy limit is outside of the redshift bin
    z_brights[msk1_bright] = (
        z_2  # Set the maximum accessible volume to the upper edge of the redshift bin
    )

    msk1_faint = z_faints < z_1  # If faint galaxy limit is outside of the redshift bin
    z_faints[msk1_faint] = (
        z_1  # Set the maximum accessible volume to the lower edge of the redshift bin
    )

    msk2_bright = np.logical_or(
        z_brights < z_1, z_brights > z_2
    )  # If the faint galaxy redshifts are outside of the redshift bin

    msk2_faint = np.logical_or(
        z_faints < z_1, z_faints > z_2
    )  # If the faint galaxy redshifts are outside of the redshift bin

    composite_mask = np.logical_or(
        msk2_bright, msk2_faint
    )  # This is the rejection mask
    composite_mask_accept = ~composite_mask  # This is the acceptance mask

    assert (
        z_faints[composite_mask_accept] < z_brights[composite_mask_accept]
    ).all(), f"Condition z_faints>z_brights not satisfied for {z_1}<z<{z_2} bin"

    VMax = (
        (z_to_vc(z_brights) - z_to_vc(z_faints)) * omega_area / n_centers
    )  # Compute vmax

    return VMax, composite_mask_accept


def printStats(inputArr, name):
    """
    Print basic summary statistics for an array.

    Outputs the minimum, median, maximum, and standard deviation of
    ``inputArr``, labeled with ``name``. Intended for quick debugging
    and sanity checks.
    """
    print(
        f"min, median, max, std of {name}: "
        f"{np.min(inputArr):0.2f}, {np.median(inputArr):0.2f}, "
        f"{np.max(inputArr):0.2f}, {np.std(inputArr):0.2f}"
    )
    return


def printContents(inputArr, name):
    """
    Print unique values and their counts for an array.

    Displays the unique elements of ``inputArr`` and the number of times
    each occurs, labeled with ``name``. Useful for inspecting categorical
    or discretized data during debugging.
    """
    print(f"contents of {name}: {np.unique(inputArr, return_counts=True)}")
    return


def prep_vmax_quantities(zmin, zmax, cosmo):
    """
    Precompute redshift–distance–volume relations and spline interpolants
    for efficient V_max calculations.

    This routine constructs finely sampled arrays of redshift, luminosity
    distance, comoving distance, and enclosed comoving volume over a specified
    redshift interval, and builds cubic–spline interpolants that map between
    these quantities.  These splines are intended for fast evaluation of
    redshift limits, luminosity–distance limits, and accessible comoving
    volumes when performing V_max and luminosity–function calculations.

    Specifically, the function produces:

      * Cubic–spline interpolants for converting between these quantities.

    The redshift grid is sampled densely to ensure smooth and accurate spline
    interpolation across the entire redshift range.

    Parameters
    ----------
    zmin : float
        Lower bound of the redshift interval.

    zmax : float
        Upper bound of the redshift interval.

    cosmo : astropy.cosmology.Cosmology
        Cosmology object used to compute distance–redshift relations.

    Returns
    -------

    splines : list of scipy.interpolate.CubicSpline
        A list of cubic–spline interpolants, in order:

        - ``dl_to_z`` : maps luminosity distance → redshift
        - ``z_to_dc`` : maps redshift → comoving distance
        - ``z_to_dl`` : maps redshift → luminosity distance
        - ``z_to_vc`` : maps redshift → enclosed comoving volume

    Notes
    -----
    These interpolants are designed to accelerate repeated conversions during
    luminosity–function and V_max calculations by avoiding repeated calls to
    cosmological distance routines.  The dense sampling ensures that spline
    interpolation errors are negligible compared to typical astrophysical
    uncertainties.
    """
    z_array = (
        np.linspace(zmin, zmax, int(50000 * (zmax - zmin))) * cu.redshift
    )  # Array of redshifts
    dl_array = (z_array).to(
        u.Mpc, cu.redshift_distance(cosmo, kind="luminosity")
    )  # Array of luminosity distances
    dc_array = (z_array).to(
        u.Mpc, cu.redshift_distance(cosmo, kind="comoving")
    )  # Array of comoving distances
    vc_array = 4 * np.pi / 3 * (dc_array**3)  # Array of comoving volumes
    dl_to_z = CubicSpline(
        dl_array, z_array
    )  # CubicSpline of luminosity distance to redshift
    z_to_dc = CubicSpline(
        z_array, dc_array
    )  # CubicSpline of redshift to comoving volume
    z_to_dl = CubicSpline(
        z_array, dl_array
    )  # CubicSpline of redshift to luminosity distance
    z_to_vc = CubicSpline(z_array, vc_array)  # CubicSpline of z to comoving volume

    splines = [dl_to_z, z_to_dc, z_to_dl, z_to_vc]

    return splines


def luminosityFunction(
    inputData,
    inputCatalog,
    limiting_mags,
    z_step=0.5,
    delta_mag=0.2,
    z_max=3,
    brightMag=-22.5,
    faintMag=-15.4,
    fname="SkySimSchecter.jpg",
    tight=True,
    save=False,
    p0=[1e-4, -10, -1.2],
    maxfev=10000,
    fit_schecter=True,
    use_vmax=True,
    z_min=0,
):
    """
    Compute, fit, and plot rest–frame galaxy luminosity functions in LSST bands
    across a sequence of redshift slices.

    This routine partitions an input galaxy catalog into contiguous redshift
    bins of width ``z_step`` between ``z_min`` and ``z_max``.  For each redshift
    slice and for each LSST band (u, g, r, i, z, y), it constructs binned
    luminosity functions in absolute magnitude and optionally estimates the
    differential luminosity function using the V_max estimator.  When enabled,
    a Schechter function is fitted to the binned luminosity function within
    adaptive magnitude limits.

    The resulting luminosity functions and fits are visualized in a multi–panel
    figure with one row per redshift slice and one column per LSST band.

    Parameters
    ----------
    inputData : pandas.DataFrame
        Galaxy catalog containing at minimum a ``"redshift_measured"`` column
        and the rest–frame absolute magnitude columns:

        - ``"Mag_true_u_lsst_z0_no_host_extinction"``
        - ``"Mag_true_g_lsst_z0_no_host_extinction"``
        - ``"Mag_true_r_lsst_z0_no_host_extinction"``
        - ``"Mag_true_i_lsst_z0_no_host_extinction"``
        - ``"Mag_true_z_lsst_z0_no_host_extinction"``
        - ``"Mag_true_Y_lsst_z0_no_host_extinction"``

    inputCatalog : GCRCatalogs.cosmodc2.SkySim5000GalaxyCatalog
        Catalog object providing cosmology and survey geometry information.

    limiting_mags : dict
        Dictionary mapping each LSST band to its faint apparent–magnitude limit.
        Used to compute V_max redshift bounds.

    z_step : float, optional
        Width of the redshift slices. Default is ``0.5``.

    delta_mag : float, optional
        Width of the absolute–magnitude bins. Default is ``0.2``.

    z_max : float, optional
        Maximum redshift to consider. Default is ``3``.

    brightMag : float, optional
        Bright (more negative) absolute–magnitude bound for binning.
        Default is ``-22.5``.

    faintMag : float, optional
        Faint absolute–magnitude bound for binning.
        Default is ``-15.4``.

    fname : str, optional
        Filename for saving the output figure.

    tight : bool, optional
        Apply ``tight_layout`` to the figure before saving/showing.

    save : bool, optional
        Save the figure to disk if True.

    p0 : sequence, optional
        Initial guess ``[phi_star, M_star, alpha]`` for Schechter–function fitting.

    maxfev : int, optional
        Maximum number of function evaluations for the non–linear fit.

    fit_schecter : bool, optional
        If True, fit a Schechter function to each luminosity function.

    use_vmax : bool, optional
        If True, estimate the luminosity function using the V_max estimator.
        Otherwise, uses a simple ``phi = N / (V_eff h^3)`` estimator.

    z_min : float, optional
        Minimum redshift of the analysis volume.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated multi–panel luminosity–function figure.

    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of Axes with shape ``(N_z_bins, 6)``.

    results : dict
        Dictionary keyed by ``(z_low, z_high, band)`` containing fitted
        Schechter parameters and covariance matrices:
        ``{'phi_star', 'M_star', 'alpha', 'cov'}``.

    Notes
    -----
    - When ``use_vmax=True``, luminosity functions are computed via the
      Eales (1993) V_max formalism and normalized by ``h^3``.
    - Schechter fits are performed only over magnitude ranges where the
      binned luminosity function is nonzero and maximized in the faintest bin.
    - All magnitudes are assumed to be rest–frame z=0 absolute magnitudes.
    """
    omega_area = inputCatalog.sky_area / (
        4 * np.pi * (180 / np.pi) ** 2
    )  # The solid angle subtended by the galaxy catalog

    fig, axs = plt.subplots(
        round((z_max - z_min) / z_step),
        len(LSST_bands),
        figsize=(4 * len(LSST_bands), 4 * round((z_max - z_min) / z_step)),
        sharex=True,
    )

    if use_vmax:
        splines = prep_vmax_quantities(z_min, z_max, inputCatalog.cosmology)

    rowIter = 0
    lum_bin_edges = np.arange(
        faintMag, brightMag + delta_mag, step=delta_mag
    )  # The luminosity bin edges in absolute mags

    results = {}  # The fitted results

    for z_lower in np.arange(z_min, z_max, step=z_step):
        z1, z2 = z_lower, z_lower + z_step
        V = (
            inputCatalog.cosmology.comoving_volume(z2)
            - inputCatalog.cosmology.comoving_volume(z1)
        ).value  # Mpc^3

        V_eff = V * omega_area

        data = inputData[
            (inputData["redshift_measured"] > z1)
            & (inputData["redshift_measured"] <= z2)
        ]
        # Could add an is_central filter here, to check for robustness...

        colIter = 0
        for columnName, band in zip(
            [
                "Mag_true_u_lsst_z0_no_host_extinction",
                "Mag_true_g_lsst_z0_no_host_extinction",
                "Mag_true_r_lsst_z0_no_host_extinction",
                "Mag_true_i_lsst_z0_no_host_extinction",
                "Mag_true_z_lsst_z0_no_host_extinction",
                "Mag_true_Y_lsst_z0_no_host_extinction",
            ],
            LSST_bands,
        ):

            ax = axs[rowIter, colIter]

            bin_num = {}
            for mag_low in np.arange(brightMag, faintMag, step=delta_mag):
                msk = np.logical_and(
                    data[columnName] > mag_low,
                    data[columnName] <= mag_low + delta_mag,
                )
                n_gals = len(data[msk])

                bin_num[(mag_low, mag_low + delta_mag)] = n_gals

            bad_keys = [k for k, v in bin_num.items() if v <= 0]
            for k in bad_keys:
                bin_num.pop(k)

            k, v = (
                bin_num.keys(),
                bin_num.values(),
            )  # k are the bin edges, v is the number of galaxies inside the bin

            if fit_schecter:

                M_centers = np.median(np.array(list(k)), axis=1)
                # Get the dimmest absolute magnitude bin where the galaxy count is maximized
                mag_dim_fit = max(max(bin_num, key=bin_num.get))

                # Get the brightest absolute magnitude bin with more than one galaxy inside of it
                mag_bright_fit, bright_number = min(
                    ((k, v) for k, v in bin_num.items() if v > 0),
                    key=lambda item: item[0],
                )
                mag_bright_fit = min(mag_bright_fit)
                assert (
                    bright_number > 0
                ), f"Low value ({low_value}) not greater than zero."

                mask = np.logical_and(
                    np.array(M_centers >= mag_bright_fit),
                    np.array(M_centers <= mag_dim_fit),
                )

                # Do the phi computation
                if use_vmax:
                    # Use the VMAX method
                    phi = phi_VMax_eales(
                        omega_area,
                        splines,
                        data[columnName],
                        z1,
                        z2,
                        limiting_mags[band],
                        band_saturation[band],
                        bin_num,
                    )

                else:
                    # Do a crude computation of phi=N/(V * h^3)
                    phi = np.array(list(v)) / (V_eff * inputCatalog.cosmology.h**3)

                k_centers = np.median(
                    np.array(list(k)), axis=1
                )  # Compute absolute mag bin centers

                mask = np.array([x.all() for x in mask])

                popt, pcov = curve_fit(
                    schechter_M,
                    M_centers[mask],
                    np.array(phi)[mask],
                    p0=p0,
                    maxfev=maxfev,
                )

                M_plot = np.linspace(mag_bright_fit, mag_dim_fit, 400)

                # Now do the rescaling
                h_cubed = np.power(inputCatalog.cosmology.h, 3)

                ax.plot(M_plot, schechter_M(M_plot, *popt))
                ax.plot(k_centers, np.array(phi), "-o", color=band_color[band])

                ax.axvline(
                    mag_bright_fit, color="red", alpha=0.6
                )  # Plot the magnitude limits for fitting purposes
                ax.axvline(mag_dim_fit, color="red", alpha=0.6)

                # Reporting values, formatting
                popt[0] = popt[0] / h_cubed
                fitTxt = ""
                if colIter == 0:
                    fitTxt += f"{z_lower:0.2f}<z<{z_lower+z_step:0.2f}"
                    fitTxt += "\n"
                labels = [
                    rf"\phi^*",
                    r"M^* - 5 \text{log(h)}",
                    rf"\alpha",
                    rf"M_1",
                    rf"M_2",
                ]
                units = [
                    r"\times 10^{-2} \text{ Mpc}^{-3} \text{ h}^3",
                    r"\text{mag}",
                    None,
                    r"\text{mag}",
                    r"\text{mag}",
                ]
                quantities = [
                    popt[0] * 10**2,
                    popt[1] - 5 * np.log10(inputCatalog.cosmology.h),
                    popt[2],
                    mag_dim_fit,
                    mag_bright_fit,
                ]

                for t, label, unt in zip(quantities, labels, units):
                    # if unt=='mag':
                    if unt != None:
                        fitTxt += rf"${label}$ = {t:.2f} ${unt}$"
                        # fitTxt += rf"${label}$ [{unt}] = {t:.2f}"
                    else:
                        fitTxt += rf"${label}$ = {t:.2f}"
                    if label != labels[-1]:
                        fitTxt += "\n"

                ax.text(
                    0.34,
                    0.19,
                    fitTxt,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=1),
                )

                results[(z1, z2, band)] = dict(
                    phi_star=popt[0] * 10**2,
                    M_star=popt[1] - 5 * np.log10(inputCatalog.cosmology.h),
                    alpha=popt[2],
                    cov=pcov,
                )  # TODO rescale the covariance and variances

            else:
                k_centers = np.median(np.array(list(k)), axis=1)
                ax.plot(k_centers, v, "-o")  # Plot raw number counts

            colIter += 1
        rowIter += 1

    # Formatting each plot
    for a, band in zip(axs[-1, :], LSST_bands):
        a.set_xlabel("$M_{}$".format(band))

    for a in axs[:, 0]:
        if fit_schecter:
            # a.set_ylabel("$\phi [h^3 Mpc^{-3]}]$")
            a.set_ylabel("$\phi [Mpc^{-3]}]$")
        else:
            a.set_ylabel("$N_{gals}$")

    for a in axs.flatten():
        a.grid(ls="--", which="major")
        a.grid(ls="-.", which="minor", alpha=0.3)
        a.semilogy()
        # a.set_ylim(10**-9,10**0)

    if tight:
        fig.tight_layout()
    if save:
        fig.savefig(os.path.join(os.getcwd(), fname), dpi=200)
    plt.show()
    return fig, axs, results


def getHp_band_dict(hp_ids, lsst_bands, sigma, nside, limiting_mags):
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
    limiting_mags : dict
        Dictionary with keys ("u", "g", "r", "i", "z", "y"), and values of limiting mags for each band

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
    realistic_mags_hp_matched = {}  # Instantiate the dictionary
    uniq_hp_ids = np.sort(np.unique(hp_ids))  # Get the unique hp indices
    for band in lsst_bands:  # iterate over bands
        realistic_mags = limiting_mags[band] + getMagLimitAdjustment(sigma, nside)  #
        realistic_mags_hp_matched[band] = realistic_mags[uniq_hp_ids]
    hp_band_dict = {}  # Instantiate the dictionary
    count = 0  # Start count at zero
    for uniq_hp_id, count in zip(uniq_hp_ids, range(len(uniq_hp_ids))):
        band_mag_dict = {}  # Instantiate the dictionary
        for band in lsst_bands:  # iterate over bands
            band_mag_dict[band] = realistic_mags_hp_matched[band][
                count
            ]  # populate dict
        hp_band_dict[uniq_hp_id] = band_mag_dict  # populate dict
        count += 1  # Iterate count
    return hp_band_dict  # Return populated dictionary


def dropFaintGalaxies(
    data,
    uniq_hp_indices,
    lsst_bands,
    hp_band_dictionary,
    hp_ind_label="hp_ind_nside128",
):
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
    for ind in uniq_hp_indices:  # For each healpix index
        hp_data = data[
            data[hp_ind_label] == ind
        ]  # Subset of data, here for the healpix specified
        dropIndices = np.array([])  # Instantiate dropIndex array
        for band in lsst_bands:  # Iterate over bands
            bool_index = (
                hp_data[f"mag_true_{band}_lsst_no_host_extinction"]
                > hp_band_dictionary[ind][band]
            )  # Set the magnitude limit
            dropIndices = np.append(
                dropIndices,
                data[data[hp_ind_label] == ind][
                    f"mag_true_{band}_lsst_no_host_extinction"
                ][bool_index].index,
            )  # Append qualifying indices to array
        data = data.drop(dropIndices, axis=0)  # Drop rows in dataframe
    return data


def RaDecToIndex(RA, decl, nside):
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
    return hp.ang2pix(nside, np.radians(-decl + 90.0), np.radians(360.0 - RA))


def getMagLimArr(nside, lim):
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
    return np.full(hp.nside2npix(nside), lim)


def getMagLimitAdjustment(sig, nside):
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
    counts = -np.random.lognormal(mean=0, sigma=sig, size=hp.nside2npix(nside))
    counts -= np.median(counts)
    return counts


def transmute_redshift(inputRedshift, inputCosmology, alternate_h=0.5, zmax=5, zmin=0):
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
    newCosmologyParams["H0"] = alternate_h * 100 * u.km / (u.Mpc * u.s)
    newCosmology = FlatLambdaCDM(name="skySimCopy", **newCosmologyParams)

    # newLuminosityDistances = newCosmology.luminosity_distance(inputRedshift_units)

    return d_L.to(
        cu.redshift,
        cu.redshift_distance(newCosmology, kind="luminosity", zmax=zmax, zmin=zmin),
    )

    # newLuminosityDistances.to(cu.redshift, cu.redshift_distance(newCosmology, kind="luminosity", zmax=5))


def mag_log_normal_dist(sigma, num_pix):
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
    counts = -np.random.lognormal(mean=0, sigma=sigma, size=num_pix)
    counts -= np.median(counts)
    return counts, np.min(counts), np.max(counts)


def perBand_mag_distribution(LSST_bands, limitingMags, sigma, nside):
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
        counts, min_counts, max_counts = mag_log_normal_dist(sigma, num_pix)
        band_limit_arrs[band] = limitingMags[band] + counts
        band_limit_min_arrs[band] = min_counts
        band_limit_max_arrs[band] = max_counts
    return band_limit_arrs, band_limit_min_arrs, band_limit_max_arrs


def trueZ_to_specZ(true_z, year):
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

    """
    z_adjust = np.random.normal(
        loc=true_z, scale=0.0004 * (1 + true_z), size=len(true_z)
    )
    return z_adjust


def trueZ_to_photoZ(true_z, year, modeled=False):
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
        prefactor = 0.01
    else:
        prefactor = 0.04
    time_term = np.sqrt(10 / year)
    z_adjust = np.random.normal(
        loc=true_z, scale=prefactor * time_term * (1 + true_z), size=len(true_z)
    )
    return z_adjust


def GCR_filter_overlord(
    year,
    LSST_bands,
    # e_dict,
    # v_dict,
    limiting_mags,
    airmass=1.2,
    z_max=1.2,
    z_min=None,
    mag_scatter=0.15,
    nside=128,
):
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
    limiting_mags : dict
        Per-band limiting magnitudes.
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
    # limiting_mags = GCR_mag_filter_from_year(year, LSST_bands, e_dict, v_dict, airmass)

    band_limit_dict, band_limit_min_dict, band_limit_max_dict = (
        perBand_mag_distribution(LSST_bands, limiting_mags, mag_scatter, nside)
    )

    # Add the magnitude limiting filter
    filters = []
    for band in LSST_bands:
        filters.append(
            f"mag_true_{band}_lsst_no_host_extinction<{limiting_mags[band]+band_limit_max_dict[band]}"
        )  # limiting_mags has been swapped for band_limit_max_dict here
        filters.append(
            f"mag_true_{band}_lsst_no_host_extinction>{band_saturation[band]}"
        )  # Bright limit

    # Add the redshift filter
    filters = GCR_redshift_filter(filters, z_max, z_min)

    return filters, band_limit_dict


def GCR_redshift_filter(filt, z_max, z_min=None):
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
    if z_min != None:
        filt.append(f"redshift_true>{z_min}")
    return filt


def GCR_mag_filter_from_year(year, LSST_bands, e_dict, v_dict, airmass=1.2):
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
        C_m, m_sky, theta_eff, k_m = getLSSTBandParameters(band)
        limiting_mags[band] = LSST_mag_lim(
            C_m, m_sky, theta_eff, e_dict[band] * v_dict[band] * year, k_m, airmass
        )

    return limiting_mags


def LSST_mag_lim(C_m, m_sky, theta_eff, t_vis, k_m, X):
    """
    C_m is the band dependent parameter
    m_sky is the sky brightness (AB mag arcsec−2)
    theta_eff is the seeing (in arcseconds)
    t_vis is the exposure time (seconds)
    k_m is the atmospheric extinction coefficient
    X is air mass
    """
    return (
        C_m
        + 0.5 * (m_sky - 21)
        + 2.5 * np.log10(0.7 / theta_eff)
        + 1.25 * np.log10(t_vis / 30)
        - k_m * (X - 1)
    )


def getLSSTBandParameters(band):
    # Band dict in the form of band: [C_m,m_sky,theta_eff,k_m]
    # From eq. 6 of Ivecic 2019
    bandDict = {
        "u": [23.09, 22.99, 0.92, 0.491],
        "g": [24.42, 22.26, 0.87, 0.213],
        "r": [24.44, 21.20, 0.83, 0.126],
        "i": [24.32, 20.48, 0.80, 0.096],
        "z": [24.16, 19.60, 0.78, 0.069],
        "y": [23.73, 18.61, 0.76, 0.170],
    }
    return bandDict[band.lower()]


def O4DutyCycles(detector):
    """
    A function to return the O4b duty cycles, based on a detector supplied
    Duty cycles taken from https://observing.docs.ligo.org/plan/
    """

    if detector == "H1":
        return 0.65
    elif detector == "L1":
        return 0.8
    elif detector == "V1":
        return 0.7
    else:
        raise ValueError(
            "Detector {} is not one of 'H1', 'L1', or 'V1'".format(detector)
        )
        return


def detectorListing(obsRun="O4"):
    """
    A function to return a list of interferometers, based on their duty cycles
    """

    if obsRun == "O4":
        dutyFunc = O4DutyCycles
    else:
        raise ValueError(
            "Observing run {} not yet supported for computing duty cycles.".format(
                obsRun
            )
        )
        return

    detList = []
    for detector in ["H1", "L1", "V1"]:
        if np.random.random() < dutyFunc(detector):
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
        "BBH": bb.gw.source.lal_binary_black_hole,  # Binary Black Hole model
        "BNS": bb.gw.source.lal_binary_neutron_star,  # Binary Neutron Star model
        # "NSBH": bb.gw.source.NeutronStarBlackHole  # Neutron Star-Black Hole model
    }

    if source_model_name not in source_models:
        raise ValueError(
            "Invalid source model '{}'. Choose from {}.".format(
                source_model_name, list(source_models.keys())
            )
        )

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
    return "{}_{}".format(base_dir, next_number)


def fakeGeoCentTime():
    """
    A function to generate a fake geocentric time between 2020-2022 (inclusive)
    returns the geocentric time in unix form
    The time is between the start-end time of O4b
    """
    times = ["2024-04-03T00:00:00", "2025-06-04T00:00:00"]
    tArr = Time(times, format="isot", scale="tcg")
    return bb.core.prior.Uniform(tArr[0], tArr[1]).sample(1).unix[0]


def makeGeoCentTime(time):
    return Time(time, format="isot", scale="tcg")


def constraintToUniform(priorDict):
    """
    A function to convert any priors in the prior dictionary from bb.core.prior.base.Constraint types to bb.core.prior.prior.Uniform types.
    """
    # Turning entries in the prior dictionary from Constraint type to Uniform over the constraint range
    for i, k in zip(priorDict.keys(), priorDict.values()):
        if type(k) == bb.core.prior.base.Constraint:
            priorDict[i] = bb.core.prior.Uniform(k.minimum, k.maximum)

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
        "BBH": bb.gw.prior.BBHPriorDict,  # Binary Black Hole
        "BNS": bb.gw.prior.BNSPriorDict,  # Binary Neutron Star
        # "NSBH": bb.gw.prior.NSBHPriorDict  # Neutron Star-Black Hole
    }

    if merger_type not in merger_priors:
        raise ValueError(
            "Invalid merger type '{}'. Choose from {}.".format(
                merger_type, list(merger_priors.keys())
            )
        )

    return merger_priors[merger_type]()
