#!/usr/bin/env python3

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import bilby as bb
import json
import sys
sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # or "sans-serif" or "monospace"
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

plt.style.use("seaborn-v0_8-paper")

def getInjectionFilePath(startPath,label):
    return os.path.join(startPath,"injectionFiles",label)+"_injection.dat"

def makeLabel(batch,item,spin):
    """
    Label should follow the following structure:
    batch_item_spin
    Where each keyword means the following
    batch: the batch production of the CBC catalog, 
           where all the CBC items are children of
    item: the individual CBC within a batch, usually 
          ranging from 0-300ish
    spin: 0/1, indicating spins misaligned (0), or 
          spins aligned (1)
    """
    return "{}_{}_{}".format(batch,spin,item)

def makeOutDir(base,spec):
    return os.path.join(base,spec)

def getWaveformModel(typ):
    if typ not in ("BBH","NSBH"):
        raise ValueError("{} is not one of supported types: ('BBH','NSBH')".format(typ))
        return -1
    if typ=="BBH":
        return "IMRPhenomXPHM"
    else:
        return "IMRPhenomNSBH"

def makePriorFilePath(typ,spin):
    if typ not in ("BBH","NSBH"):
        raise ValueError("{} is not one of supported types: ('BBH','NSBH')".format(typ))
        return -1
    if spin not in (1,0):
        raise ValueError("{} is not one of supported spins: (0,1)".format(spin))
        return -1

    if typ=="BBH":
        if spin==1: # BBH, spins aligned
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/bbh_aligned_gwtc4_forBilbyPE.prior"
        else: # BBH, spins misaligned
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/bbh_precessing_gwtc4_forBilbyPE.prior"
    else:
        if spin==1: # NSBH, spins aligned
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/nsbh_aligned_gwtc4_forBilbyPE.prior"
        else: # NSBH, spins misaligned
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/nsbh_precessing_gwtc4_forBilbyPE.prior"

def getSpinFromName(name):
    if name=='aligned':
        return 1
    elif name=='precessing':
        return 0
    else:
        print("Something went wrong in getSpinFromName function, name supplied: {}".format(name))

def selectSpin():
    return np.random.randint(0,2)
    
def makeInjection_dict(injKeys,row):
    myDict = {}
    for k in injKeys:
        # if k in ("chi_1", "chi_2"): # handling the tilt/chi issues
        #     myDict[k] = row["tilt_{}".format(k[-1])]
        if k in ["ra"]:
            myDict[k] = np.deg2rad(row[k]) # SkySim is over -180 to 180 deg, while Bilby defaults to 0 to 2pi
            # Originally had 'm' before here...
        elif k in ["dec"]:
            myDict[k] = np.deg2rad(row[k]) # Skysim is from -90 to 90 deg, while bilby is -pi/2 to pi/2
            # Originally had 'm' before here...
        else:
            myDict[k] = row[k]
    gcTime = 1
    while abs(gcTime)>0.1:
        gcTime = np.random.normal(0,0.01,size=1)[0]
    myDict["geocent_time"] =gcTime
    for ent in myDict.keys():
        myDict[ent] = float(myDict[ent]) # Formatting fix
    return myDict

def getInjection_keys(typ,spin,ppath):
    if typ not in ("BBH","NSBH"):
        raise ValueError("{} is not one of supported types: ('BBH','NSBH')".format(typ))
        return -1
    if spin not in (1,0):
        raise ValueError("{} is not one of supported spins: (0,1)".format(spin))
        return -1
    return bb.gw.prior.CBCPriorDict(ppath).sample().keys()

basePath = "/pscratch/sd/s/seanmacb/proj411ProdRuns/catalogs"
alignment = ["precessing","aligned"]
# subDir = ["Uniform,r","Uniform,u","Uniform,y","Uniform,StellarMass","Uniform,Uniform"]
# subDir = ["Uniform,r","Uniform,u","Uniform,y"]

dataDir="/global/u1/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs"
# pd.read_csv()
msk = [(x.endswith("withSNRs_gwtc4.csv") and (x.__contains__("BBH"))) for x in os.listdir(dataDir)] 
# To include NSBH's, add an or statement to the second conditional
files = np.sort(np.array(os.listdir(dataDir))[msk])

print(f"Relevant files: {files}")

fieldKeys = ["label","outdir","prior-file", "injection-dict","injection-file","waveform-approximant"]

for f in files:
    subsampledDF = pd.read_csv(os.path.join(dataDir,f))
    parentWeighting,cbcWeighting,CBCType,alignment,__ = f.split(",")
    cbcWeightingDir = cbcWeighting.split("Weight")[0]
    parentWeightingDir = parentWeighting.split("Parent")[0]

# print("cbcWeighting:",cbcWeighting)

    toWrite = np.array([])
    labels_snrs_dict = {}
    
# for b,alignment in zip(batchPaths,alignmentArr):
    # batchDF = pd.read_csv(os.path.join(os.getcwd(),catalogPath,b))
    batchPath = os.path.join(basePath,"{},{}".format(parentWeightingDir,cbcWeightingDir))
    os.makedirs(batchPath,exist_ok=True)

    for index,row in subsampledDF.iterrows():
        fullItem = "{}_{}".format(CBCType,index)
        # spin=alignment # Choose spins to be aligned (1) or misaligned (0)
        label = makeLabel(parentWeighting+","+cbcWeighting,fullItem,alignment)
        outDir = makeOutDir(batchPath,label)
        priorPath = makePriorFilePath(CBCType,getSpinFromName(alignment))
        injKeys = getInjection_keys(CBCType,getSpinFromName(alignment),priorPath)
        injDict = makeInjection_dict(injKeys,row)        
        injectionFPath = getInjectionFilePath(batchPath,label)
        wfModel = getWaveformModel(CBCType)

        labels_snrs_dict[label] = row["Network SNR"]

        toWrite = np.append(toWrite,[label,outDir,priorPath,injDict,injectionFPath,wfModel])

# Reshape the final array
    toWrite = np.reshape(toWrite,(-1,len(fieldKeys)))

    # Create the .ini 
    ini_basepath = os.path.join(batchPath,"iniFiles")
    if not os.path.isdir(ini_basepath):
        os.mkdir(ini_basepath)

    # Writing all the .ini files
    for writeables in toWrite:
        with open("{}_base.ini".format(CBCType), "r") as myf:
            read = myf.readlines()
    
        writeFile = os.path.join(ini_basepath,writeables[0])+".ini"
        
        for field,value in zip(fieldKeys,writeables):
            if field!="injection-dict": # Not using the injection dict anymore
                read.append("\n")
                read.append("{}={}\n".format(field,value))
    
        with open(writeFile, "w") as myf:
            myf.writelines(read)
    
    # Write the injection file
    injectionFilePath = os.path.join(batchPath,"injectionFiles")
    os.makedirs(injectionFilePath,exist_ok=True)
    for injectionDictionary,label,out in zip(toWrite[:,3],toWrite[:,0],toWrite[:,1]):
        with open(os.path.join(injectionFilePath,"{}_injection.dat".format(label)), "w") as myf:
            for ky in injectionDictionary.keys():
                myf.write(f"{ky} ")
            myf.write("\n")
            for val in injectionDictionary.values():
                myf.write(f"{val} ")
            

    # labels_snrs_dict
    sortedDict = {k: v for k, v in sorted(labels_snrs_dict.items(), key=lambda item: item[1])}
    result = [k for k in list(sortedDict.keys())[9::10]]
    print("For {}".format(f))
    for r in result:
        print(r,sortedDict[r])
    print("Finished writing for {}".format(f),end="\n\n")
