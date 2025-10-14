#!/usr/bin/env python3
"""
Sample from two input CSVs, compute SNRs, and write to output files.

Usage:
    python sample_two_csvs.py \
        --csv1 path/to/file1.csv \
        --csv2 path/to/file2.csv \
        --n_samples 1000 \
        --network 9 \
        --individual 2 \
        --out_csv_1 path/to/out1.csv
        --out_csv_2 path/to/out2.csv
        --prior_path_one path/to/prior.prio
        --prior_path_two path/to/prior.prio 
        --duration 8
"""

import argparse
import numpy as np
import pandas as pd
import os
import bilby as bb
import sys
sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut

def compute_snr(ifos,generator,row):
    """
    Placeholder function for computing network and individual SNRs.
    Customize this with your actual calculation.
    """
    network_snr = 0
    indivSNR = []
    for ifo in ifos:
        signal = ifo.get_detector_response(waveform_polarizations=generator.frequency_domain_strain(row),
                                           parameters=row)
        single_snr_squared = ifo.optimal_snr_squared(signal).real
        indivSNR.append(np.sqrt(single_snr_squared))
        network_snr += single_snr_squared
    network_snr = np.sqrt(network_snr)

    indiv_snr = np.min(indivSNR)

    return network_snr,indiv_snr

def loadPrior(path):
    """
    Loading the prior from a path.
    """
    print("Loading prior from",path)
    prior = bb.gw.prior.ConditionalPriorDict(path) 
    cbcKeys = prior.sample().keys()
    print("Prior keys:",cbcKeys)
    return prior,cbcKeys

def configureIFOS():
    """
    Configure the interferometers as needed. For now, I am leaving these default.
    """
    ifos = bb.gw.detector.InterferometerList(["H1","V1","L1"])
    sampling_frequency = 2048
    waveform_arguments = dict(waveform_approximant="IMRPhenomXP",  # waveform approximant name
                                      reference_frequency=50.0,  # gravitational waveform reference frequency (Hz)
                             )
    return ifos,sampling_frequency,waveform_arguments

def makeInjectionDictionary(keys,r):
    """
    Make the injection dictionary for the csv.
    """
    injDict = {}
    for key in keys:
        if key in ("ra","dec"):
            injDict[key] = r["m"+key]
        else:
            injDict[key] = r[key] 
    injDict["geocent_time"] = ut.fakeGeoCentTime()
    return injDict

def main(args):
    path1 = args.csv1
    path2 = args.csv2

    prior1,cbcKeys1 = loadPrior(args.prior_path_one)
    prior2,cbcKeys2 = loadPrior(args.prior_path_two)
    duration = args.duration
    ifos,sampling_frequency,waveform_arguments = configureIFOS()

    # Randomly split total samples between the two catalogs
    n1 = int(np.random.normal(args.n_samples/2,np.sqrt(args.n_samples)))
    assert n1>0, "Value for n1 out of range: {} less than 0".format(n1)
    assert n1<args.n_samples, "Value for n1 out of range ({} greater than {})".format(n1,args.n_samples)
    n2 = args.n_samples - n1
    assert n1+n2==args.n_samples, "{} + {} != {}, something went wrong in the subsample distribution generation".format(n1,n2,args.n_samples)
    print(f"Drawing {n1} samples from {os.path.basename(args.csv1)} "
          f"and {n2} from {os.path.basename(args.csv2)}")

    # Prepare output files
    out1 = f"{args.out_csv_1}.csv"
    out2 = f"{args.out_csv_2}.csv"
    open(out1, "w").close()  # Clear existing file
    open(out2, "w").close()

    # Define thresholds
    network_snr_threshold = args.network
    individual_snr_threshold = args.individual

    for n, path, outpath,keys,prior in zip([n1, n2], [path1, path2], [out1, out2],
                                         [cbcKeys1,cbcKeys2], [prior1,prior2]):

        df = pd.read_csv(path)
        
        accepted_rows = []
        progress=0
        while len(accepted_rows) < n:
            # Draw a random row
            row = df.sample(1).iloc[0]

            # Create the injection dictionary for that row
            injDict = makeInjectionDictionary(keys,row)

            # Set up the waveform generator
            waveform_generator = bb.gw.waveform_generator.WaveformGenerator(sampling_frequency=sampling_frequency,duration=duration,
                                                                                       frequency_domain_source_model=ut.get_source_model("BBH"),parameters=injDict,
                                                                                       waveform_arguments=waveform_arguments,start_time=injDict["geocent_time"]-duration/2)

            # Set the interferometer strain
            ifos.set_strain_data_from_power_spectral_densities(
                        duration=duration,
                        sampling_frequency=sampling_frequency,
                        start_time=injDict["geocent_time"] - duration/2,
                    )

            # Compute SNRs
            network_snr, individual_snr = compute_snr(ifos,waveform_generator,injDict)
            injDict["Minimum Individual SNR"] = individual_snr
            injDict["Network SNR"] = network_snr

            # Apply thresholds
            if (network_snr >= network_snr_threshold and individual_snr >= individual_snr_threshold):
                accepted_rows.append(list(injDict.values()))

            # Print progress
            if len(accepted_rows)*100//n==progress:
                print("{}% complete for {}".format(len(accepted_rows)*100//n,outpath))
                progress+=1

        # Append accepted rows to output CSV
        pd.DataFrame(accepted_rows,columns=list(injDict.keys())).to_csv(
            outpath, mode="a", header=True, index=False
        )
        print(f"Output .csv written to {outpath}.")

    print(f"Results written to:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample from two CSVs, computing SNR's of the CBC, and only permitting CBC's that meet SNR threshold criteria.")
    parser.add_argument("--csv1", required=True, help="Path to first CSV file.")
    parser.add_argument("--csv2", required=True, help="Path to second CSV file.")
    parser.add_argument("--out_csv_1", required=True, help="Path to first output CSV file.")
    parser.add_argument("--out_csv_2", required=True, help="Path to second output CSV file.")
    parser.add_argument("--prior_path_one", required=True, help="Path to the prior file associated with the first csv.")
    parser.add_argument("--prior_path_two", required=True, help="Path to the prior file associated with the second csv.")
    parser.add_argument("--n_samples", type=int, required=True, help="Total number of samples.")
    parser.add_argument("--network", type=int, required=True, help="The network SNR threshold.")
    parser.add_argument("--individual", type=int, required=True, help="The individual SNR threshold.")
    parser.add_argument("--duration", type=int, default=8, help="The duration of signal to analyze in seconds.")
    args = parser.parse_args()

    main(args)