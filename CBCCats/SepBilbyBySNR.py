import pandas as pd
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
cbcCatPath = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalogs/BBHs_0,n=1e7,NSBHs,FromSkySim50_withBilby.csv" # Modify this
print("Loading CBC dictionary from",cbcCatPath)
cbcCat = pd.read_csv(cbcCatPath)
print("CBC DF Header:",cbcCat.columns.values)

import bilby as bb
import sys
sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut
import numpy as np
bb.core.utils.log.setup_logger(outdir='.', label=None, log_level=30)

priorPath = "/pscratch/sd/s/seanmacb/gwCosmoDesc/lib/python3.10/site-packages/bilby/gw/prior_files/precessing_spins_nsbh_gwtc3.prior" # Modify this
print("Loading prior from",priorPath)
prior = bb.gw.prior.ConditionalPriorDict(priorPath) 
cbcKeys = prior.sample().keys()
print("Prior keys:",cbcKeys)

ifos = bb.gw.detector.InterferometerList(["H1","V1","L1"])
duration = 8 # Modify this
sampling_frequency = 2048
waveform_arguments = dict(waveform_approximant="IMRPhenomXP",  # waveform approximant name
                                  reference_frequency=50.0,  # gravitational waveform reference frequency (Hz)
                                 )

net_snr = []
indiv_snr = []
print("Computing SNR")
for ids,row in cbcCat.iterrows():
    injDict = {}
    for key in cbcKeys:
        if key in ("ra","dec"):
            injDict[key] = row["m"+key]
        else:
            injDict[key] = row[key] 
    injDict["geocent_time"] = ut.fakeGeoCentTime()
    
    waveform_generator = bb.gw.waveform_generator.WaveformGenerator(sampling_frequency=sampling_frequency,duration=duration,
                                                                               frequency_domain_source_model=ut.get_source_model("BBH"),parameters=injDict,
                                                                               waveform_arguments=waveform_arguments,start_time=injDict["geocent_time"]-duration/2)
    
    ifos.set_strain_data_from_power_spectral_densities( # Set the strain
                duration=duration,
                sampling_frequency=sampling_frequency,
                start_time=injDict["geocent_time"] - duration/2,
            )
    
    # Compute SNR
    
    network_snr = 0
    indivSNR = []
    for ifo in ifos:
        signal = ifo.get_detector_response(
            waveform_polarizations=waveform_generator.frequency_domain_strain(injDict),
            parameters=injDict)
        single_snr_squared = ifo.optimal_snr_squared(signal).real
        indivSNR.append(np.sqrt(single_snr_squared))
        network_snr += single_snr_squared
    network_snr = np.sqrt(network_snr)

    net_snr.append(network_snr)
    indiv_snr.append(np.min(indivSNR))

cbcCat["Individual SNR minimum"] = indiv_snr
cbcCat["Network SNR"] = net_snr

outPath = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mockCBCCatalogs/n=1e7,NSBHs,FromSkySim50_withBilbySNRs.csv" # Modify this
cbcCat.to_csv(outPath,index=False) 
print("Finished, csv written to",outPath)
