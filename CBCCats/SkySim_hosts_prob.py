import numpy as np
import matplotlib.pyplot as plt 
import os
from astropy.cosmology import FlatLambdaCDM
import GCR
import GCRCatalogs as GCRCat
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import itertools
from itertools import permutations 
from itertools import product 
import os

for k in range(2,5):
    cosmo = FlatLambdaCDM(H0=70, Om0=0.283, Tcmb0=2.725) # Define a base cosmology
    basePath = "../data/CBCCatFigures" # Base figure path
    plt.rcParams.update({"text.usetex":True, "figure.dpi":150})
    
    psicbc = lambda z :(1+z)**(1.82) /(1 + (z/2)**3.82) * 1/(1+z) # eqn 15 from 2405.07904
    frate = lambda M,  z, Mstar = -20.378, eps=1 : 10**(-0.4 * eps * (M - Mstar)) * psicbc(z) # eqn 17 from 2405.07904
    
    Marr=  np.linspace(-28, -16) # Mass and redshift bins
    zarr = np.linspace(0, 10, 100)
    
    
    plt.figure(figsize=(4,6))
    plt.subplot(211)
    for _zz in np.linspace(0, 2, 4):
        plt.plot(Marr, frate(Marr, _zz), label=f"z={_zz:.2f}")
    plt.legend(loc='best')
    plt.xlabel("$M$ (r-band)")
    plt.ylabel("$f_{rate}(z,M)$")
    
    plt.subplot(212)
    for _MM in np.linspace(-24, -19, 6):
        plt.plot(zarr, frate(_MM, zarr), label=f"M={_MM:.2f}")
    plt.legend(loc='best')
    plt.ylabel("$f_{rate}(z,M)$")
    plt.xlabel("$z$")
    plt.savefig(os.path.join(basePath,"z_Magi_vs_frequency.png"),dpi=180)
    plt.close()
    
    # 
    _mgrid = np.linspace(-23, 0, 1000) #  Mmax = 0, fixes the Nevents in 52.243 for z in (0,10)
    _zgrid = np.linspace(0, 10, 10000) # z grid
    fratez = np.array([np.trapz(frate(_mgrid, _zz), _mgrid) for _zz in _zgrid])
    nevents = np.trapz(fratez, _zgrid)
    print("Number of events:",nevents)
    
    plt.plot(_zgrid, fratez/nevents)
    plt.ylabel("$f_{rate}(z)$ [normalized]")
    plt.xlabel("$z$")
    plt.savefig(os.path.join(basePath,"z_vs_rate_norm.png"),dpi=180)
    plt.close()
    
    # SkySim5000
    
    cat_name2 = "skysim5000_v1.2_small"
    print("Loading",cat_name2)
    skysimCat = GCRCat.load_catalog(cat_name2)
    
    skysimCat.get_catalog_info()
    
    allNativeQuants = skysimCat.list_all_native_quantities()
    
    allQuantitiesOfInterest = np.array(["ra","ra_true","dec","dec_true","redshift","redshiftHubble","galaxyID","baseDC2/sfr","hostHaloTag","baseDC2/sod_halo_radius",
                            "LSST_filters/magnitude:LSST_i:rest","LSST_filters/magnitude:LSST_g:rest", # magnitudes in rest and obs frames
                            "LSST_filters/magnitude:LSST_y:rest","LSST_filters/magnitude:LSST_r:observed",
                            "LSST_filters/magnitude:LSST_z:rest","LSST_filters/magnitude:LSST_z:observed",
                            "LSST_filters/magnitude:LSST_u:observed","LSST_filters/magnitude:LSST_i:observed",
                            "LSST_filters/magnitude:LSST_r:rest","LSST_filters/magnitude:LSST_y:observed",
                            "LSST_filters/magnitude:LSST_u:rest","LSST_filters/magnitude:LSST_g:observed",
                            "LSST_filters/totalLuminositiesStellar:LSST_y:rest","LSST_filters/totalLuminositiesStellar:LSST_u:observed", # stellar luminosities in all bands, rest and obs frames
                            "LSST_filters/totalLuminositiesStellar:LSST_g:rest","LSST_filters/totalLuminositiesStellar:LSST_r:observed",
                            "LSST_filters/totalLuminositiesStellar:LSST_y:observed","LSST_filters/totalLuminositiesStellar:LSST_i:rest",
                            "LSST_filters/totalLuminositiesStellar:LSST_i:observed","LSST_filters/totalLuminositiesStellar:LSST_g:observed",
                            "LSST_filters/totalLuminositiesStellar:LSST_z:observed","LSST_filters/totalLuminositiesStellar:LSST_r:rest",
                            "LSST_filters/totalLuminositiesStellar:LSST_u:rest","LSST_filters/totalLuminositiesStellar:LSST_z:rest",
                            "totalMassStellar","hostHaloMass","isCentral"])
    
    print("Creating mask for LSST")
    filterMask =  np.array([not x.__contains__("LSST") for x in allQuantitiesOfInterest])
    filteredQuantities = np.append(allQuantitiesOfInterest[filterMask],"LSST_filters/magnitude:LSST_r:rest")
    
    # %%time
    print("Getting quantities from {}".format(cat_name2))
    print("Quantities:",filteredQuantities)
    data = pd.DataFrame(skysimCat.get_quantities(filteredQuantities))
    
    # data.index
    
    print("Data info:",data.info())
    
    ## Set P_host
    print("Setting P_host")
    phost = frate(data["LSST_filters/magnitude:LSST_r:rest"], data["redshiftHubble"])
    phost = phost/ np.sum(phost.values)
    
    
    # plt.figure(dpi=120)
    nn = 6000000
    plt.scatter(data['redshiftHubble'].values[:nn], 
                data['LSST_filters/magnitude:LSST_r:rest'][:nn],  
                c= np.log10(phost.values[:nn]), cmap=plt.get_cmap('viridis', 11), s=0.5, alpha=0.8, vmax=-4.5, vmin=-10)
    plt.ylabel("M (r-band)")
    plt.xlabel("$z$")
    _cb = plt.colorbar()
    _cb.ax.set_ylabel(r"$log_{10}(P\_host)$")
    plt.gca().set_facecolor('k')
    plt.savefig(os.path.join(basePath,"z_vs_magR_vs_hostProb.jpg"),dpi=180)
    plt.close()
    
    def stretch(ra_in,dec_in,ra_range=(60.46709862957921, 70.3338108507918),dec_range=(-46.57034567653332, -32.79850657784978)):
        delta_ra = ra_in - ra_range[0]
        delta_dec = dec_in - dec_range[0]
    
        stretched_ra = (1000*360 * (delta_ra / (ra_range[1]-ra_range[0])))%360
        stretched_dec = (1000*180 * (delta_dec / (dec_range[1]-dec_range[0])))%180 - 90
    
        return stretched_ra,stretched_dec
    
    
    print("Creating a list of mergers")
    
    np.random.seed(1503170817)
    n_merges = 10000000 # Updated now to 1E7
    # Getting indexes based on p_host
    events_index = np.random.choice(data.index, 
                                    size=n_merges, 
                                    p=phost.values)
    
    print("Data shape:",data.shape)
    
    print("Number of unique hosts:",np.unique(events_index).shape)
    
    print("Making the CBC catalog")
    
    cat_cbc = data.iloc[events_index]
    
    print("Number of CBC's in each CBC host:",cat_cbc.galaxyID.value_counts())
    
    print("Merging data IDs from skysim to mock CBC catalog")
    
    catmock = cat_cbc.apply(lambda x: stretch(x['ra'], x['dec']), result_type='expand', axis=1)
    catmock = catmock.rename(columns={0:'mra', 1: 'mdec'}, )
    
    cat_cbc[["mra","mdec"]] = catmock[['mra', 'mdec']]
    
    ofn = "../data/mockCBCCatalogs/CBCs_{},n=1e7,FromSkySim50.csv".format(k)
    print("Saving CBC catalog to",ofn)
    assert not os.path.isfile(ofn), "File exists!!!"
    cat_cbc[["galaxyID", 'redshiftHubble', 'ra', 'dec', 'mra', 'mdec' ]].to_csv(ofn)
    
    ### Check distribution for mock_ra and mock_dec
    h,yedges,xedges,img = plt.hist2d(x=cat_cbc['mra'], y=cat_cbc['mdec'], bins=40, cmap=plt.get_cmap("magma", 20)); 
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    cax = plt.colorbar()
    plt.savefig(os.path.join(basePath,"AllSkyDistribution.png"),dpi=180)
    plt.close()
    print("Finished, file saved to",ofn)

