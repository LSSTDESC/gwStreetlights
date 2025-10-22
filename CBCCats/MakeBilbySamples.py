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

def main(args):
    dataDir = args.dataDir
    cat_name2 = args.catalogName
    CBCCatalogPath = args.CBCCatalogPath
    priorPath = args.priorPath
    outFileName = args.outFileName
    
    print("Loading catalog:",cat_name2)
    skysimCat = GCRCat.load_catalog(cat_name2) # Load the skysim catalog
    cosmology = skysimCat.cosmology
    hostDF = pd.read_csv(CBCCatalogPath) # Load the CBC catalog
    prior = bb.gw.prior.PriorDict(priorPath) # The bbh prior, spins aligned
    
    prior_sampled = prior.sample()
    
    print("Injection keys before pop:",prior_sampled.keys())
    prior_sampled.pop("ra") # Pop RA
    prior_sampled.pop("dec") # Pop Dec

    keys = prior_sampled.keys()
    
    injDict = {}
    for k in keys:
        injDict[k] = []
    
    print("Injection dictionary:",injDict)
    
    cnt = 0
    for ids,row in hostDF.iterrows():
        thisSample = prior.sample()
        for k in keys:
            if k!="luminosity_distance":
                injDict[k].append(thisSample[k])
        injDict["luminosity_distance"].append(float(str(skysimCat.cosmology.luminosity_distance(row["redshiftHubble"])).split(" ")[0]))
        if cnt % 100000 == 0:
            print("{}% finished".format(cnt//100000))
        cnt+=1
    
    for k in injDict.keys():
        hostDF[k] = injDict[k]
    
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
    args = parser.parse_args()
    main(args)
