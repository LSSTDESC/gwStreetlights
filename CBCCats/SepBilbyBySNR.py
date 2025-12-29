#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import bilby as bb
import sys

sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut
import numpy as np
import argparse

cbcCatPath = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalog_csvs/BBHs_0,aligned,sampledOnly.csv"  # Modify this
print("Loading CBC dictionary from", cbcCatPath)
cbcCat = pd.read_csv(cbcCatPath)
print("CBC DF Header:", cbcCat.columns.values)

bb.core.utils.log.setup_logger(outdir=".", label=None, log_level=30)

priorPath = "/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/bbh_aligned_gwtc.prior"  # Modify this
dataPath = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data"


def main(run):
    fname = "batch_0_bbh_aligned_{}.txt".format(run)
    prior, cbcKeys = loadPrior(priorPath)
    ifos, duration, sampling_frequency, waveform_arguments = configureIFOS()
    cleanVars()
    lastRow = startup(dataPath, fname)

    breadth = len(cbcCat) - lastRow
    print("Slurm NTasks:", int(os.environ["SLURM_NTASKS"]))
    startPerc = run / int(256)
    endPerc = startPerc + 1 / int(256)
    print(
        "Indices:",
        int(lastRow + np.floor(startPerc * breadth)),
        int(lastRow + np.ceil(endPerc * breadth)),
    )
    CBCSubset = cbcCat.iloc[
        int(lastRow + np.floor(startPerc * breadth))
        + 1 : int(lastRow + np.ceil(endPerc * breadth))
        + 1
    ]

    tabulateCBCRows(
        CBCSubset,
        cbcKeys,
        sampling_frequency,
        duration,
        waveform_arguments,
        ifos,
        fname,
    )


def tabulateCBCRows(subsetDF, keys, sampling_freq, durat, waveform_args, ifos, fname):

    for ids, row in subsetDF.iterrows():

        injDict = makeInjectionDictionary(keys, row)

        waveform_generator = bb.gw.waveform_generator.WaveformGenerator(
            sampling_frequency=sampling_freq,
            duration=durat,
            frequency_domain_source_model=ut.get_source_model("BBH"),
            parameters=injDict,
            waveform_arguments=waveform_args,
            start_time=injDict["geocent_time"] - durat / 2,
        )

        ifos.set_strain_data_from_power_spectral_densities(  # Set the strain
            duration=durat,
            sampling_frequency=sampling_freq,
            start_time=injDict["geocent_time"] - durat / 2,
        )

        network_snr, indiv_snr = ComputeSNRs(ifos, waveform_generator, injDict)

        with open(os.path.join(dataPath, fname), "a") as myfile:
            myfile.write(
                str([ids, network_snr, indiv_snr])[1:-1] + "\n"
            )  # index, netSnr, minIndivSNR


def loadPrior(path):
    print("Loading prior from", path)
    prior = bb.gw.prior.ConditionalPriorDict(priorPath)
    cbcKeys = prior.sample().keys()
    print("Prior keys:", cbcKeys)
    return prior, cbcKeys


def configureIFOS():
    ifos = bb.gw.detector.InterferometerList(["H1", "V1", "L1"])
    duration = 8  # Modify this
    sampling_frequency = 2048
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP",  # waveform approximant name
        reference_frequency=50.0,  # gravitational waveform reference frequency (Hz)
    )
    return ifos, duration, sampling_frequency, waveform_arguments


def ComputeSNRs(ifos, generator, myDict):
    """
    Compute network and minimum individual SNRs of a given GW injection
    """
    network_snr = 0
    indivSNR = []
    for ifo in ifos:
        signal = ifo.get_detector_response(
            waveform_polarizations=generator.frequency_domain_strain(myDict),
            parameters=myDict,
        )
        single_snr_squared = ifo.optimal_snr_squared(signal).real
        indivSNR.append(np.sqrt(single_snr_squared))
        network_snr += single_snr_squared
    network_snr = np.sqrt(network_snr)

    indiv_snr = np.min(indivSNR)

    return network_snr, indivSNR


def cleanVars():
    try:
        del lastRow, l
    except:
        pass
    return


def startup(dataPath, fname):
    if not os.path.isfile(os.path.join(dataPath, fname)):
        print(
            "File does not exist, creating empty file at {}, setting last row index to -1".format(
                os.path.join(dataPath, fname)
            )
        )

        open(os.path.join(dataPath, fname), "a").close()  #
        lastRow = -1

        return lastRow
    else:
        with open(os.path.join(dataPath, fname), "r") as f:
            try:
                for l in f.readlines():
                    pass
                lastRow = int(l.split(",")[0])
                print("Last row set as {}".format(lastRow))
            except:
                print("Unable to read last line, setting last row index to -1")
                lastRow = -1
        return lastRow


def makeInjectionDictionary(keys, r):
    injDict = {}
    for key in keys:
        if key in ("ra", "dec"):
            injDict[key] = r["m" + key]
        else:
            injDict[key] = r[key]
    injDict["geocent_time"] = ut.fakeGeoCentTime()
    return injDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run", type=int, help="The run number for the specific subset"
    )
    args = parser.parse_args()
    main(args.run)

# Afterward...

# Open the file, read the three columns

# Add them to cbcCat

# Then save the file
