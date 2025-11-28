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
    cbcKeys = list(prior.sample().keys())
    for k in ["mass_1","mass_2","mass_ratio","chirp_mass"]:
        if k not in cbcKeys:
            cbcKeys.append(k)
    print("Prior keys:",cbcKeys)
    return prior,cbcKeys

def getWaveformModel(typ):
    if typ not in ("BBH","NSBH"):
        raise ValueError("{} is not one of supported types: ('BBH','NSBH')".format(typ))
        return -1
    if typ=="BBH":
        return "IMRPhenomXPHM"
    else:
        return "IMRPhenomNSBH"

def configureIFOS(cbcT):
    """
    Configure the interferometers as needed. For now, I am leaving these default.
    """
    approximant = getWaveformModel(cbcT)
    
    approximant = "IMRPhenomXPHM" 
    # This is a stupid placeholder for now. 
    # I don't know how any GW infrastructure is held together.
    # When all of their waveform models fail when breathed upon.
    
    ifos = bb.gw.detector.InterferometerList(["H1","V1","L1"])
    sampling_frequency = 2048
    waveform_arguments = dict(waveform_approximant=approximant,  # waveform approximant name
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
    gcTime = 1
    while abs(gcTime)>0.1:
        gcTime = np.random.normal(0,0.01,size=1)[0]
    injDict["geocent_time"] =gcTime
    return injDict

def lal_binary_black_hole_getArgs(injDict,cbcT):
    """
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Angle between the total binary angular momentum and the line of sight
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    kwargs: dict
        Optional keyword arguments
        Supported arguments:
    
        - waveform_approximant
        - reference_frequency
    """
    args = {}

    print("Injection dictionary:")
    for k,v in zip(injDict.keys(),injDict.values()):
        print(k,v)

    args["mass_1"] = injDict["mass_1"]
    args["mass_2"] = injDict["mass_2"]
    args["luminosity_distance"] = injDict["luminosity_distance"]
    args["theta_jn"] = injDict["theta_jn"]
    args["phase"] = injDict["phase"]
    args["waveform_approximant"] = getWaveformModel(cbcT)
    args["reference_frequency"] = 50

    if "a_1" in injDict.keys(): # Precessing spins
        args["a_1"] = injDict["a_1"]
        args["a_2"] = injDict["a_2"]
        args["tilt_1"] = injDict["tilt_1"]
        args["tilt_2"] = injDict["tilt_2"]
        args["phi_12"] = injDict["phi_12"]
        args["phi_jl"] = injDict["phi_jl"]
    else: # Aligned spins
        args["a_1"] = injDict["chi_1"]
        args["a_2"] = injDict["chi_2"]
        args["tilt_1"] = 0
        args["tilt_2"] = 0
        args["phi_12"] = 0
        args["phi_jl"] = 0

    args["frequency_array"] = np.arange(20,2048,step=0.1)
    return args

def main(args):
    path1 = args.csv1
    path2 = args.csv2

    cbcType = args.cbc_type

    prior1,cbcKeys1 = loadPrior(args.prior_path_one)
    prior2,cbcKeys2 = loadPrior(args.prior_path_two)
    duration = args.duration
    ifos,sampling_frequency,waveform_arguments = configureIFOS(cbcType)

    # Randomly split total samples between the two catalogs
    n1 = int(np.random.normal(args.n_samples/2,np.sqrt(args.n_samples)))
    while n1<=0:
        print("Value for n1 out of range: {} less than 0. Retrying.".format(n1))
        n1 = int(np.random.normal(args.n_samples/2,np.sqrt(args.n_samples)))
    while n1>=args.n_samples:
        print("Value for n1 out of range ({} greater than {}). Retrying".format(n1,args.n_samples))
        n1 = int(np.random.normal(args.n_samples/2,np.sqrt(args.n_samples)))

    n2 = args.n_samples - n1
    while n1+n2!=args.n_samples:
        print("{} + {} != {}, something went wrong in the subsample distribution generation. Retrying.".format(n1,n2,args.n_samples))
        n1 = int(np.random.normal(args.n_samples/2,np.sqrt(args.n_samples)))
        n2 = args.n_samples - n1
   
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

    for n, path, outpath,keys in zip([n1, n2], [path1, path2], [out1, out2],
                                         [cbcKeys1,cbcKeys2]):

        df = pd.read_csv(path)
        
        accepted_rows = []
        progress=0
        while len(accepted_rows) < n:
            # Draw a random row
            row = df.sample(1).iloc[0]

            if cbcType.lower()=="nsbh" and row["mass_ratio"]<0.01:
                print(f"Mass ratio is too low, skipping ({row['mass_ratio']})")
                continue # Skip this row...
            if row["luminosity_distance"]>7000:
                print(f"Luminosity distance of galaxy {row['galaxyID']} is too high ({row['luminosity_distance']})") # This is a crude solution for now
                continue # Skip this row
            # Create the injection dictionary for that row
            injDict = makeInjectionDictionary(keys,row)

            lalArgs = lal_binary_black_hole_getArgs(injDict,cbcType)

            # Set up the waveform generator
            waveform_generator = bb.gw.waveform_generator.WaveformGenerator(sampling_frequency=sampling_frequency,duration=duration,
                                                                            frequency_domain_source_model=bb.gw.source.lal_binary_black_hole,
                                                                            parameters=injDict,waveform_arguments=waveform_arguments,start_time=injDict["geocent_time"]-duration/2)

            # waveform_generator = bb.gw.waveform_generator.GWSignalWaveformGenerator(spinning=True,sampling_frequency=sampling_frequency,
            #                                                                         duration=duration,parameters=injDict,
            #                                                                         start_time=injDict["geocent_time"]-duration/2,
            #                                                                         waveform_arguments={"waveform_approximant":"IMRPhenomNSBH"},
            #                                                                        )

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
    parser.add_argument("--cbc_type", required=True, help="The type of CBC that we are sampling.")
    parser.add_argument("--n_samples", type=int, required=True, help="Total number of samples.")
    parser.add_argument("--network", type=int, required=True, help="The network SNR threshold.")
    parser.add_argument("--individual", type=int, required=True, help="The individual SNR threshold.")
    parser.add_argument("--duration", type=int, default=8, help="The duration of signal to analyze in seconds.")
    args = parser.parse_args()

    main(args)
