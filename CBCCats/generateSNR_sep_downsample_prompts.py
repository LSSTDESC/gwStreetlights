#!/usr/bin/env python3

"""
This file will generate the prompts for `SNR_sep_downselect.py`

Argument syntax:

--csv1 path/to/file1.csv \
--csv2 path/to/file2.csv \
--n_samples 1000 \
--network 9 \
--individual 2 \
--out_csv_1 path/to/out1.csv
--out_csv_2 path/to/out2.csv
--prior_path_one path/to/prior.prior
--prior_path_two path/to/prior.prior
--duration 8
"""

basePath = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs/UniformParent,"
suffix='withCBCParams.csv'
outDir = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/CBCCats/outFiles"

'''
Files take the form 'UniformParent,{weight},{CBCType},{CBC_alignment},withCBCParams.csv', where
weight = ["UniformWeight","StellarMassWeight","uWeight","rWeight","yWeight"]
CBC_alignment = ["aligned","precessing"]
CBCType = ["BBH","NSBH"]
'''

def priorSelector(alignment,CBCType):
    if alignment=="aligned":
        if CBCType=="BBH":
            return "/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/bbh_aligned_gwtc.prior" # Return the BBH aligned prior
        elif CBCType=="NSBH":
            return "/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/nsbh_aligned_gwtc.prior" # Return the NSBH aligned prior
        else:
            print("Something went wrong in the prior selector, inputs were alignment: {}. CBCType: {}".format(alignment,CBCType))
    elif alignment=="precessing":
        if CBCType=="BBH":
            return "/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/bbh_precessing_gwtc.prior" # Return the BBH precessing prior
        elif CBCType=="NSBH":
            return "/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/nsbh_precessing_gwtc.prior" # Return the NSBH precessing prior
        else:
            print("Something went wrong in the prior selector, inputs were alignment: {}. CBCType: {}".format(alignment,CBCType))
    else:
        print("Something went wrong in the prior selector, inputs were alignment: {}. CBCType: {}".format(alignment,CBCType))

for weight in ["UniformWeight","StellarMassWeight","uWeight","rWeight","yWeight"]:
    for CBCType,n_samps,duration,network in zip(["BBH","NSBH"],[300,10],[8,32],[9,8]):
        inPath1 = basePath+weight+","+CBCType+","+"aligned"+","+suffix # Aligned
        inPath2 = basePath+weight+","+CBCType+","+"precessing"+","+suffix # Precessing
        individual = 2
        out1 = ",".join(inPath1.split(",")[:-1])+",withSNRs.csv"
        out2 = ",".join(inPath2.split(",")[:-1])+",withSNRs.csv"
        prior1 = priorSelector("aligned",CBCType)
        prior2 = priorSelector("precessing",CBCType)
        print(f"python SNR_sep_downselect.py --csv1 {inPath1} --csv2 {inPath2} --out_csv_1 {out1} --out_csv_2 {out2} --prior_path_one {prior1} --prior_path_two {prior2} --n_samples {n_samps} --network {network} --individual {individual} --duration {duration} &> {outDir}/{weight}_{CBCType}.out &",end="\n\n")