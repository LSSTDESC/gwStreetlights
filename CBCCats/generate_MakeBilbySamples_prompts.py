#!/usr/bin/env python
# coding: utf-8

# Here is the argument structure
"""
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


-d DATADIR

-c CATALOGNAME

-C CBCCATALOGPATH

-p PRIORPATH

-o OUTFILENAME
"""

import os

priorBasePath = (
    "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/gwtc-like-priors"
)
dataDirectory = (
    "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs"
)
# CBCTypes = ["BBH", "NSBH"]
CBCTypes = ["BBH"]
alignmentTypes = ["aligned", "precessing"]
catalogName = "skysim5000_v1.2_small"
# CBCWeights = ["UniformWeight", "StellarMassWeight", "uWeight", "rWeight", "yWeight"]
CBCWeights = ["UniformWeight", "StellarMassWeight", "uWeight"]


def priorSelector(alignment, CBCType):
    if alignment == "aligned":
        if CBCType == "BBH":
            return os.path.join(
                priorBasePath, "bbh_aligned_gwtc4_noMass.prior"
            )  # Return the BBH aligned prior
        elif CBCType == "NSBH":
            return os.path.join(
                priorBasePath, "nsbh_aligned_gwtc4_noMass.prior"
            )  # Return the NSBH aligned prior
        else:
            print(
                "Something went wrong in the prior selector, inputs were alignment: {}. CBCType: {}".format(
                    alignment, CBCType
                )
            )
    elif alignment == "precessing":
        if CBCType == "BBH":
            return os.path.join(
                priorBasePath, "bbh_precessing_gwtc4_noMass.prior"
            )  # Return the BBH precessing prior
        elif CBCType == "NSBH":
            return os.path.join(
                priorBasePath, "nsbh_precessing_gwtc4_noMass.prior"
            )  # Return the NSBH precessing prior
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


def prompt(
    datadir, catname, catpath, priorpath, outfile, cbctype, cbcweight, alignment
):
    return f"./MakeBilbySamples.py -d {datadir} -c {catname} -C {catpath} -p {priorpath} -o {outfile} -t {cbctype} &> outFiles/gwtc4_{cbcweight}_{cbctype}_{alignment}.log &"


if __name__ == "__main__":
    for cbcweight in CBCWeights:
        for cbctype in CBCTypes:
            for alignment in alignmentTypes:
                outFileName = os.path.join(
                    dataDirectory,
                    f"UniformParent,{cbcweight},{cbctype},{alignment},withCBCParams_gwtc4.csv",
                )
                # Takes the form of {parent catalog weighting},{cbc catalog weighting},{cbc type [BBH or NSBH]},{cbc alignment [aligned or precessing]},withCBCParams.csv

                readPath = os.path.join(dataDirectory, f"UniformParent,{cbcweight}.csv")
                # Takes the form of {parent catalog weighting},{cbc catalog weighting}.csv
                print()
                print(
                    prompt(
                        dataDirectory,
                        catalogName,
                        readPath,
                        priorSelector(alignment, cbctype),
                        outFileName,
                        cbctype,
                        cbcweight,
                        alignment,
                    )
                )
