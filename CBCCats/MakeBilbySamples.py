import bilby as bb
import numpy as np
import gwcosmo as gwc
import pandas as pd
import GCR
import GCRCatalogs as GCRCat
import matplotlib.pyplot as plt
import os
from astropy.time import Time

cat_name2 = "skysim5000_v1.2"
print("Loading catalog:",cat_name2)
skysimCat = GCRCat.load_catalog(cat_name2) # Load the skysim catalog
hostDF = pd.read_csv("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs/CBCs_0,n=1e7,FromSkySim50.csv") # Load the CBC catalog

prior = bb.gw.prior.ConditionalPriorDict("/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/bbh_aligned_gwtc.prior") # The bbh prior, spins aligned

keys = list(prior.sample())
print("Injection keys before pop:",keys)
keys.pop(3) # Pop RA
keys.pop(3) # Pop Dec

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

dataDir = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs"
# hostDF.to_csv(os.path.join(dataDir,".csv"),columns=saveColumns,index=False)

# hostDF = pd.read_csv(os.path.join(dataDir,"CBCs_0,n=1e7,FromSkySim50.csv"))

hostDF["sampled"] = False
savePath=os.path.join(dataDir,"BBHs_0,aligned,sampledOnly.csv")
hostDF.to_csv(savePath,columns=saveColumns,index=False)
print("File saved:",savePath)
