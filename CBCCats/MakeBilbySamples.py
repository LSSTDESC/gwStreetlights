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
skysimCat = GCRCat.load_catalog(cat_name2) # Load the skysim catalog
hostDF = pd.read_csv("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mergers-w=Lum,n=1e7,FromSkySim50.csv") # Load the CBC catalog

prior = bb.gw.prior.BBHPriorDict(aligned_spin=False) # The bbh prior, spins misaligned
prior["luminosity_distance"] = bb.gw.prior.UniformSourceFrame(0,5000,cosmology=skysimCat.cosmology,name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None) # Update the luminosity distance prior, based on 

keys = list(prior.sample())
keys.pop(3)
keys.pop(3)

injDict = {}
for k in keys:
    injDict[k] = []

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

saveColumns = hostDF.columns.values[1:] # This is because of the extra column - check this always!!!

dataDir = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data"
hostDF.to_csv(os.path.join(dataDir,"mergers-w=Lum,n=1e7,FromSkySim50_withBilby.csv"),columns=saveColumns,index=False)

hostDF = pd.read_csv(os.path.join(dataDir,"mergers-w=Lum,n=1e7,FromSkySim50_withBilby.csv"))

hostDF["sampled"] = False

hostDF.to_csv(os.path.join(dataDir,"mergers-w=Lum,n=1e7,FromSkySim50_withBilby.csv"),columns=saveColumns,index=False)