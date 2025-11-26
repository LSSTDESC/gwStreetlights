#!/usr/bin/env python
# coding: utf-8

import sys
import importlib

# Ensure we're using the clean astropy cosmology, not GCRCatalogs'
if "astropy.cosmology" in sys.modules:
    importlib.reload(sys.modules["astropy.cosmology"])

import bilby as bb
import numpy as np
import pandas as pd
import GCRCatalogs as GCRCat
import matplotlib.pyplot as plt
import os
import argparse
from astropy.time import Time
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import gwcosmo


# Manually set Bilby's default cosmology
bb.gw.cosmology.DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=71* u.km / (u.Mpc*u.s), Om0=0.2648, Tcmb0=0*u.K, Neff=3.04, m_nu=None, Ob0=0.0448)

def getGWTC4MassPrior(typ):
    if typ.lower()=="bbh":
        return gwcosmo.priors.BBH_broken_powerlaw_multi_peak_gaussian(alpha_1=1.7282865329473678,alpha_2=4.511690397723029,b=0.5,beta=1.1709777343204348,mminbh=5.058572081868678,mmaxbh=300,lambda_g=0.36102328981694193,lambda_g_low=0.5860680995810035,mu_g_low=9.763667347989355,sigma_g_low=0.6491643865532204,mu_g_high=32.76291758105389,sigma_g_high=3.9181194675793933,delta_m=4.320691590156577)
    elif typ.lower()=="nsbh":
        bhPrior = gwcosmo.priors.BBH_broken_powerlaw_multi_peak_gaussian(alpha_1=1.7282865329473678,alpha_2=4.511690397723029,b=0.5,beta=1.1709777343204348,mminbh=5.058572081868678,mmaxbh=300,lambda_g=0.36102328981694193,lambda_g_low=0.5860680995810035,mu_g_low=9.763667347989355,sigma_g_low=0.6491643865532204,mu_g_high=32.76291758105389,sigma_g_high=3.9181194675793933,delta_m=4.320691590156577)
        nsPrior = bb.core.prior.TruncatedGaussian(1.4, 0.68, 0.1, 3, name="mass_2", latex_label="$m_2$", unit=None, boundary=None) # This is the GWTC-4 result
        return [bhPrior,nsPrior]
    else:
        raise ValueError(f"Provided type ({typ}) is not one of BBH or NSBH")

def chirp(m1,m2):
    return np.power(np.power(m1*m2,3)/(m1+m2),0.2) 

def getMassParamSample(mPrior):
    if type(mPrior)==gwcosmo.priors.BBH_broken_powerlaw_multi_peak_gaussian:
        m1_m2 = mPrior.sample(1)
        m1,m2 = m1_m2[0][0],m1_m2[1][0]
    elif type(mPrior)==list:
        # Handle the multi-prior case
        m1Prior = mPrior[0]
        m2Prior = mPrior[1]
        m1,m2 = m1Prior.sample(1)[0][0],m2Prior.sample(1)[0]
    else:
        raise ValueError(f"Provided prior ({mPrior}) is incorrect, something went wrong here...")
    q = m2/m1
    chrp = chirp(m1,m2)
    return m1,m2,q,chrp

def main(args):
    cbcType = args.cbcType
    dataDir = args.dataDir
    cat_name2 = args.catalogName
    CBCCatalogPath = args.CBCCatalogPath
    priorPath = args.priorPath
    outFileName = args.outFileName
    
    print("Loading catalog:",cat_name2)
    skysimCat = GCRCat.load_catalog(cat_name2) # Load the skysim catalog
    cosmology = skysimCat.cosmology
    hostDF = pd.read_csv(CBCCatalogPath) # Load the CBC catalog
    print("Loading prior:",priorPath)
    # exit()
    prior = bb.gw.prior.PriorDict(priorPath) # The bbh prior, spins aligned
    
    prior_sampled = prior.sample()
    
    print("Injection keys before pop:",prior_sampled.keys())
    prior_sampled.pop("ra") # Pop RA
    prior_sampled.pop("dec") # Pop Dec

    keys = prior_sampled.keys()
    
    injDict = {}
    for k in keys:
        injDict[k] = []
    injDict["mass_1"] = []
    injDict["mass_2"] = []
    injDict["mass_ratio"] = []
    injDict["chirp_mass"] = []
   
    print("Injection dictionary:",injDict)
    
    massPrior = getGWTC4MassPrior(cbcType)

    cnt = 0
    for ids,row in hostDF.iterrows():
        thisSample = prior.sample()
        mass1,mass2,q_,churp = getMassParamSample(massPrior)
        for k in keys:
            if k!="luminosity_distance":
                injDict[k].append(thisSample[k])
        injDict["luminosity_distance"].append(float(str(skysimCat.cosmology.luminosity_distance(row["redshiftHubble"])).split(" ")[0]))
        injDict["mass_1"].append(mass1)
        injDict["mass_2"].append(mass2)
        injDict["mass_ratio"].append(q_)
        injDict["chirp_mass"].append(churp)
        if cnt % 100000 == 0:
            print("{}% finished".format(cnt//100000))
        cnt+=1
    
    for k in injDict.keys():
        hostDF[k] = injDict[k]

    hostDF["mass_1"] = injDict["mass_1"]
    hostDF["mass_2"] = injDict["mass_2"]
    hostDF["mass_ratio"] = injDict["mass_ratio"]
    hostDF["chirp_mass"] = injDict["chirp_mass"]
    
    print("All host columns:",hostDF.columns.values)
    saveColumns = hostDF.columns.values[1:] # This is because of the extra column - check this always!!!
    print("All columns to be saved:",saveColumns)
    
    savePath=os.path.join(dataDir,outFileName)
    hostDF.to_csv(savePath,columns=saveColumns,index=False)
    print("File saved:",savePath)

    return 0

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataDir", type=str, 
                        default="/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs",
                        help="The path to the data directory where the output csv will be written.")
    parser.add_argument("-c", "--catalogName", type=str, default="skysim5000_v1.2_small",
                        help="The galaxy catalog name, used to identify the cosmology in the luminosity distance calculation.")
    parser.add_argument("-C", "--CBCCatalogPath", type=str, 
                        default="/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs/CBCs_0,n=1e7,FromSkySim50.csv",
                        help="The path to the CBC catalog csv that is read in.")
    parser.add_argument("-p", "--priorPath", type=str, 
                        default="/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/bbh_aligned_gwtc.prior",
                        help="The path to the prior file.")
    parser.add_argument("-o", "--outFileName", type=str, default="BBHs_0,aligned,sampledOnly.csv",
                        help="The name of the output csv, appended to the dataDir argument.")
    parser.add_argument("-t", "--cbcType", type=str, default="BBH",
                        help="The CBC type to be sampled.")
    args = parser.parse_args()
    main(args)
