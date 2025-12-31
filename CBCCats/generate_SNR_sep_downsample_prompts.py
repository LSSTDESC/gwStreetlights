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
suffix = "withCBCParams_gwtc4.csv"
outDir = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/CBCCats/outFiles"

"""
Files take the form 'UniformParent,{weight},{CBCType},{CBC_alignment},withCBCParams.csv', where
weight = ["UniformWeight","StellarMassWeight","uWeight","rWeight","yWeight"]
CBC_alignment = ["aligned","precessing"]
CBCType = ["BBH","NSBH"]
"""


def priorSelector(alignment, CBCType):
    if alignment == "aligned":
        if CBCType == "BBH":
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/bbh_aligned_gwtc4_forBilbyPE.prior"  # Return the BBH aligned prior
        elif CBCType == "NSBH":
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/nsbh_aligned_gwtc4_forBilbyPE.prior"  # Return the NSBH aligned prior
        else:
            print(
                "Something went wrong in the prior selector, inputs were alignment: {}. CBCType: {}".format(
                    alignment, CBCType
                )
            )
    elif alignment == "precessing":
        if CBCType == "BBH":
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/bbh_precessing_gwtc4_forBilbyPE.prior"  # Return the BBH precessing prior
        elif CBCType == "NSBH":
            return "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors/nsbh_precessing_gwtc4_forBilbyPE.prior"  # Return the NSBH precessing prior
        else:
            print(
                "Something went wrong in the prior selector, inputs were alignment: {}. CBCType: {}".format(
                    alignment, CBCType
                )
            )
    else:
        print(
            "Something went wrong in the prior selector, inputs were alignment: {}. CBCType: {}".format(
                alignment, CBCType
            )
        )


cbcWeights = ["UniformWeight", "StellarMassWeight", "uWeight", "rWeight", "yWeight"]
cbcTypes = ["BBH", "NSBH"]
sampNumbers = [300, 10]
signalDurations = [8, 32]
networkThresholds = [9, 8]

# This is for the BBH only prompts
cbcTypes = ["BBH"]
cbcWeights = ["UniformWeight"]
sampNumbers = [14]  # Modified for testing
signalDurations = [8]
networkThresholds = [9]  # Modified for testing

# This is for the NSBH only prompts
# cbcTypes = ["NSBH"]
# # cbcWeights = ["uWeight","rWeight","yWeight"]
# sampNumbers = [10]
# signalDurations = [32]
# networkThresholds = [9]

for weight in cbcWeights:
    for CBCType, n_samps, duration, network in zip(
        cbcTypes, sampNumbers, signalDurations, networkThresholds
    ):
        inPath1 = (
            basePath + weight + "," + CBCType + "," + "aligned" + "," + suffix
        )  # Aligned
        inPath2 = (
            basePath + weight + "," + CBCType + "," + "precessing" + "," + suffix
        )  # Precessing
        individual = 2
        out1 = ",".join(inPath1.split(",")[:-1]) + ",withSNRs_gwtc4_fourthPass"
        out2 = ",".join(inPath2.split(",")[:-1]) + ",withSNRs_gwtc4_fourthPass"
        prior1 = priorSelector("aligned", CBCType)
        prior2 = priorSelector("precessing", CBCType)
        print(
            f"python SNR_sep_downselect.py --csv1 {inPath1} --csv2 {inPath2} --out_csv_1 {out1} --out_csv_2 {out2} --prior_path_one {prior1} --prior_path_two {prior2} --n_samples {n_samps} --network {network} --individual {individual} --duration {duration} --cbc_type {CBCType} &> {outDir}/{weight}_{CBCType}_fourthPass.out &",
            end="\n\n",
        )
