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

LSST_bands = ["u", "g", "r", "i", "z", "Y"]
visits_per_yr = (
    np.array([56, 74, 184, 187, 166, 171]) / 10
)  # visits per year in u-g-r-i-z-y

visits_dict = {}
for band, vis in zip(LSST_bands, visits_per_yr):
    visits_dict[band] = vis

expTimes = [38, 30, 30, 30, 30, 30]

eTime_dict = {}
for band, eTime in zip(LSST_bands, expTimes):
    eTime_dict[band] = eTime


def run_survey_diagnostics(
    data,
    hp_band_dict,
    skysim_catalog,
    year,
    z_max,
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

    Returns
    -------
    results : dict
        Dictionary of fitted Schechter parameters indexed by (z_low, z_high, band).

    mag_lim_check_result : dict
        Output of ``mag_lim_checker`` verifying per–pixel magnitude–limit compliance.
    """

    # P(z)
    fig_pz, ax_pz = p_z_distribution(data, z_max=z_max, z_step=z_step_pz)

    # Luminosity functions
    fig_lf, axs_lf, results = luminosityFunction(
        data,
        skysim_catalog,
        p0=p0,
        z_max=z_max,
        z_step=z_step_lf,
        brightMag=brightMag,
        faintMag=faintMag,
        maxfev=maxfev,
        delta_mag=delta_mag_schecter,
        fit_schecter=fit_schecter,
    )

    # Redshift precision
    fig_zprec, ax_zprec = redshiftPrecisionPlot(data, year, modeled=modeled)

    # Magnitude non–uniformity
    fig_uni, ax_uni = mag_uniformity_plot(hp_band_dict, hi_mag=hi_mag, low_mag=low_mag)
    # TODO: Add a healpix map, and direct comparison to the imposed parameters

    # Magnitude–limit consistency check
    mag_lim_check_result = None
    if NSIDE is not None:
        mag_lim_check_result = mag_lim_checker(data, hp_band_dict, NSIDE)

    figs = [fig_pz, fig_lf, fig_zprec, fig_uni]
    axes = [ax_pz, axs_lf, ax_zprec, ax_uni]

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

    # Compute limiting magnitudes
    limiting_mags = GCR_mag_filter_from_year(
        year, LSST_bands, eTime_dict, visits_dict, airmass=airmass
    )

    # Apply uniform magnitude deepening
    for b in LSST_bands:
        limiting_mags[b] += mags_deeper

    # Build GCR depth filters
    filters, band_limit_dict = GCR_filter_overlord(
        year,
        LSST_bands,
        eTime_dict,
        visits_dict,
        airmass=airmass,
        z_max=z_max,
        z_min=z_min,
        mag_scatter=uniformity,
        nside=NSIDE,
    )

    # Query SkySim catalog
    data = pd.DataFrame(
        skysimCat.get_quantities(list(dataColumns), filters=tuple(filters))
    )

    # Assign HEALPix indices
    hp_col = f"hp_ind_nside{NSIDE}"
    data[hp_col] = RaDecToIndex(data["ra_true"], data["dec_true"], NSIDE).astype(
        np.int32
    )

    # Build spatial uniformity map
    hp_uniq_ids = np.unique(data[hp_col])
    hp_band_dict = getHp_band_dict(
        hp_uniq_ids, LSST_bands, uniformity, NSIDE, limiting_mags
    )

    # Apply non-uniform depth cuts
    data = dropFaintGalaxies(
        data, hp_uniq_ids, LSST_bands, hp_band_dict, hp_ind_label=hp_col
    )

    # Apply redshift adjustment
    if alternate_h != None:
        data["redshift_true"] = data["redshift_true"] * (
            alternate_h / skysimCat.cosmology.h
        )

    # Apply redshift model
    if spectroscopic:
        data["redshift_measured"] = trueZ_to_specZ(data["redshift_true"], year)
    else:
        data["redshift_measured"] = trueZ_to_photoZ(
            data["redshift_true"], year, modeled=modeled
        )

    return data, limiting_mags, hp_band_dict


def mag_uniformity_plot(
    hp_band_dictionary,
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
        )
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
            dat["redshift_measured"] < z + z_step, dat["redshift_measured"] > z
        )
        difference = dat[msk]["redshift_measured"] - dat[msk]["redshift_true"]
        difference_abs = np.absolute(difference)
        _95_val, _5_val = np.nanpercentile(difference_abs, (95, 5))
        measured_val = np.std(difference)
        measured.append(measured_val)
        _95.append(_95_val)
        _5.append(_5_val)
        # diffs.append(difference)

    ax.step(z_arr, predicted, where="post", color="red")
    ax.scatter(
        z_arr + z_step / 2,
        predicted,
        marker="o",
        color="red",
        label="Predicted precision",
    )

    _5, _95 = [0], [0]
    ax.step(z_arr, measured, where="post", color="blue")
    ax.errorbar(
        z_arr + z_step / 2,
        measured,
        yerr=[_5, _95],
        marker="o",
        color="blue",
        label="Measured precision",
        elinewidth=1,
        capsize=2,
        linestyle="",
    )

    ax.legend()
    ax.set_xlabel("$z$")
    ax.set_ylabel("$z_{measured} - z_{true}$")

    if tight:
        fig.tight_layout()
    if save:
        fig.savefig(fname, dpi=300)

    return fig, ax


def luminosityFunction(
    inputData,
    inputCatalog,
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
):
    """
    Compute and plot binned rest–frame galaxy luminosity functions in LSST bands
    over a sequence of redshift slices.

    This routine partitions an input galaxy catalog into contiguous redshift
    bins of width ``z_step`` between :math:`z=0` and ``z_max``. For each redshift
    slice, it computes a simple binned luminosity function in rest–frame absolute
    magnitude for each of the six LSST bands (u, g, r, i, z, y), and produces a
    multi–panel figure showing the number of galaxies per magnitude bin.

    The luminosity functions are plotted as :math:`N(M)` (counts per bin) on a
    logarithmic y–axis, with the magnitude axis inverted so that brighter objects
    appear to the left, following standard astronomical convention.

    Parameters
    ----------
    inputData : pandas.DataFrame
        Catalog of galaxies containing at minimum a ``"redshift"`` column and
        the following rest–frame magnitude columns:
        - ``"Mag_true_u_lsst_z0_no_host_extinction"``
        - ``"Mag_true_g_lsst_z0_no_host_extinction"``
        - ``"Mag_true_r_lsst_z0_no_host_extinction"``
        - ``"Mag_true_i_lsst_z0_no_host_extinction"``
        - ``"Mag_true_z_lsst_z0_no_host_extinction"``
        - ``"Mag_true_Y_lsst_z0_no_host_extinction"``

    inputCatalog : GCRCatalogs.cosmodc2.SkySim5000GalaxyCatalog
        The input catalog of the associated data

    z_step : float, optional
        Width of the redshift bins. Default is ``0.5``.

    delta_mag : float, optional
        Width of the absolute–magnitude bins used to construct the luminosity
        functions. Default is ``0.2``.

    z_max : float, optional
        Maximum redshift to consider. Redshift bins are generated from
        ``z = 0`` to ``z_max`` in steps of ``z_step``. Default is ``3``.

    brightMag : float, optional
        Bright (more negative) limit of the magnitude range to be binned.
        Default is ``-22.5``.

    faintMag : float, optional
        Faint limit of the magnitude range to be binned. Default is ``-15.4``.

    fname : str, optional
        Filename used when saving the output figure. Default is
        ``"SkySimSchecter.jpg"``.

    tight : bool, optional
        If True, apply ``fig.tight_layout()`` before saving/showing the figure.
        Default is True.

    save : bool, optional
        If True, save the figure to disk in the current working directory using
        ``fname``. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib Figure object.

    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of Axes objects with shape ``(N_z_bins, 6)``, where ``N_z_bins``
        is the number of redshift slices.

    Notes
    -----
    - The resulting figure contains one row per redshift bin and six columns
      corresponding to the LSST u, g, r, i, z, and y bands.
    - This implementation computes raw number counts per magnitude bin; it does
      not apply any volume normalization, completeness correction, or Schechter
      function fitting.
    """
    fig, axs = plt.subplots(
        round(z_max / z_step), 6, figsize=(24, 4 * round(z_max / z_step)), sharex=True
    )

    rowIter = 0

    results = {}  # The fitted results

    for z_lower in np.arange(0, z_max, step=z_step):
        z1, z2 = z_lower, z_lower + z_step
        V = (
            inputCatalog.cosmology.comoving_volume(z2)
            - inputCatalog.cosmology.comoving_volume(z1)
        ).value  # Mpc^3
        V_eff = V * inputCatalog.sky_area / ((180 / (np.pi)) ** 2)
        # V_eff = V * inputCatalog.sky_area / (4*np.pi)

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
                n_gals = len(
                    data[
                        np.logical_and(
                            data[columnName] > mag_low,
                            data[columnName] <= mag_low + delta_mag,
                        )
                    ]
                )

                bin_num[mag_low + delta_mag / 2] = n_gals

            bad_keys = [k for k, v in bin_num.items() if v <= 0]
            for k in bad_keys:
                bin_num.pop(k)

            k, v = bin_num.keys(), bin_num.values()

            if fit_schecter:
                mag_low_fit = max(bin_num, key=bin_num.get)

                mag_high_fit, high_value = min(
                    ((k, v) for k, v in bin_num.items() if v > 0),
                    key=lambda item: item[0],
                )
                assert high_value > 0, f"Low value ({low_value}) not greater than zero."

                phi = np.array(list(v)) / (V_eff * inputCatalog.cosmology.h**3)

                M_centers = np.array(list(k))
                phi_vals = phi
                mask = (M_centers >= mag_high_fit) & (M_centers <= mag_low_fit)

                popt, pcov = curve_fit(
                    schechter_M, M_centers[mask], phi_vals[mask], p0=p0, maxfev=maxfev
                )

                M_plot = np.linspace(mag_high_fit, mag_low_fit, 400)
                ax.plot(M_plot, schechter_M(M_plot, *popt))  # Plot the fit

                ax.axvline(
                    mag_high_fit, color="red", alpha=0.6
                )  # Plot the magnitude limits for fitting purposes
                ax.axvline(mag_low_fit, color="red", alpha=0.6)

                ax.text(
                    0.65,
                    0.3,
                    rf"$\phi_*$ = {popt[0]:0.1E}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.65,
                    0.2,
                    rf"$M_*$={popt[1]:0.3f}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.65,
                    0.1,
                    rf"$\alpha$={popt[2]:0.3f}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.25,
                    0.1,
                    rf"$M_1$={mag_low_fit:0.3f}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.text(
                    0.25,
                    0.2,
                    rf"$M_2$={mag_high_fit:0.3f}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

                results[(z1, z2, band)] = dict(
                    phi_star=popt[0], M_star=popt[1], alpha=popt[2], cov=pcov
                )
                ax.plot(k, phi, "-o")  # Plot mag vs phi
            else:
                ax.plot(k, v, "-o")  # Plot

            colIter += 1
        ax.text(
            0.25,
            0.3,
            f"{z_lower:0.1f}<z<{z_lower+z_step:0.1f}",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        rowIter += 1

    # Formatting each plot
    for a, band in zip(axs[-1, :], LSST_bands):
        a.set_xlabel("$M_{}$".format(band))

    for a in axs[:, 0]:
        if fit_schecter:
            a.set_ylabel("$\phi [h^3 Mpc^{-3]}]$")
        else:
            a.set_ylabel("$N_{gals}$")

    # a.xaxis.set_inverted(True)

    # for a in axs[0,:]:

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
    e_dict,
    v_dict,
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
    limiting_mags = GCR_mag_filter_from_year(year, LSST_bands, e_dict, v_dict, airmass)

    band_limit_dict, band_limit_min_dict, band_limit_max_dict = (
        perBand_mag_distribution(LSST_bands, limiting_mags, mag_scatter, nside)
    )

    # Add the magnitude limiting filter
    filters = []
    for band in LSST_bands:
        filters.append(
            f"mag_true_{band}_lsst_no_host_extinction<{limiting_mags[band]+band_limit_max_dict[band]}"
        )  # limiting_mags has been swapped for band_limit_max_dict here

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
