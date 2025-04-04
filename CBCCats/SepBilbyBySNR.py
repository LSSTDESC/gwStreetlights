import pandas as pd
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
cbcCatPath = "/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mergers-w=Lum,n=1e7,FromSkySim50_withBilby.csv"
cbcCat = pd.read_csv(cbcCatPath)

import bilby as bb
import sys
sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut
import numpy as np
bb.core.utils.log.setup_logger(outdir='.', label=None, log_level=30)

prior = bb.gw.prior.BBHPriorDict(aligned_spin=False) # The bbh prior, spins misaligned
cbcKeys = prior.sample().keys()

ifos = bb.gw.detector.InterferometerList(["H1","V1","L1"])
duration = 5
sampling_frequency = 2048
waveform_arguments = dict(waveform_approximant="IMRPhenomXP",  # waveform approximant name
                                  reference_frequency=50.0,  # gravitational waveform reference frequency (Hz)
                                 )

net_snr = []
indiv_snr = []

for ids,row in cbcCat.iterrows():
    injDict = {}
    for key in cbcKeys:
        if key in ("ra","dec"):
            injDict[key] = row["m"+key]
        else:
            injDict[key] = row[key] 
    injDict["geocent_time"] = ut.fakeGeoCentTime()
    ut.get_source_model("BBH")
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

cbcCat.to_csv("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/mergers-w=Lum,n=1e7,FromSkySim50_withBilby_andSNR.csv",index=False)
