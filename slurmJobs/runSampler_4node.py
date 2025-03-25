#!/global/homes/s/seanmacb/.conda/envs/mydesc
import sys
sys.path.append("/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/utils")
import utils as ut
import argparse

def populateWaveformDict(w):
    """
    A function to populate the waveform dictionary
    """
    if "default" == w.lower():
        return dict(waveform_approximant="IMRPhenomXP",  # waveform approximant name
                                  reference_frequency=50.0,  # gravitational waveform reference frequency (Hz)
                                 )
    else:
        raise ValueError("Only 'default' is supported as a waveform generator right now. Current value is '{}'".format(args.w.lower()))
        return 2

def populateDict(keys,values):
    """
    A function to populate a dictionary
    """
    myDict = {}
    for j,k in zip(keys,values):
        myDict[j]=k
    return myDict

def addGeocentTime(t):
    """
    A function to compute the geocentric time
    """
    if "now" == t.lower():
        # Use the current time
        return ut.Time.now().unix
    elif "random" == t.lower():
        # Pull a random time from O4
        return ut.fakeGeoCentTime()
    else:
        raise ValueError("Only 'now' and 'o4normal' are supported as a time selection right now. Current value is '{}'".format(t.lower()))
        return 3

def setWaveformParameters(w):
    """
    A function to set the waveform parameters
    Subtly different from setWaveformDict
    """
    if "default" == w.lower():
        duration = 5
        sampling_frequency = 2048
    else:
        raise ValueError("Only 'default' is supported as a waveform generator right now. Current value is '{}'".format(w.lower()))
        return 2
    return duration, sampling_frequency

def createInterferometers(i):
    """
    A function to return an interferometer list
    """
    if 'all'==i.lower():
        detList = ["H1","V1", "L1"]
    elif 'o4'==i.lower():
        detList = ut.detectorListing("O4")
    else:
        raise ValueError("Only 'all' and 'o4' are supported as a waveform generator right now. Current value is '{}'".format(i.lower()))
        return 4
    print("Interferometers selected: {}".format(detList))
    ifos = ut.bb.gw.detector.InterferometerList(detList)
    return ifos

def adjustParamDict(a,duration,injection_parameters,priorCopy):
    """
    A function to adjust the parameter dictionary
    Used for the priors on the sampler
    Different from the priors on the injection sample
    """
    if a=='default':
        priorCopy["mass_1"] = ut.bb.core.prior.Constraint(10, 100, name="mass_1")
        priorCopy["mass_2"] = ut.bb.core.prior.Constraint(10, 100, name="mass_2")
        priorCopy["luminosity_distance"] = ut.bb.core.prior.Uniform(1000, 6000, "luminosity_distance")
        priorCopy['geocent_time'] = ut.bb.core.prior.Uniform(injection_parameters["geocent_time"]-duration/2.5,
                                                          injection_parameters["geocent_time"]+duration/2.5,
                                                          name="geocent_time")
    else:
        raise ValueError("Only 'default' is supported as a prior parameter search adjustment right now. Current value is '{}'".format(a.lower()))
        return 5
    return priorCopy

def run(args):
    """
    The function to run the sampler to completion
    """
    
    # Parse args
    merger_type = args.merger_type
    time = args.time
    n_injs = args.number
    baseDir = args.baseDir
    runDir = args.runDir

    if not ut.os.path.exists(baseDir):
        ut.os.makedirs(baseDir)
    
    # Get prior
    print("Getting prior for {}".format(merger_type))
    prior = ut.get_merger_prior(merger_type)
    
    # Turning entries in the prior dictionary from Constraint type to Uniform over the constraint range
    print("Turning entries in the prior dictionary from Constraint type to Uniform over the constraint range")
    prior = ut.constraintToUniform(prior)
    
    # Making x injections
    print("Making {} injections".format(n_injs))
    injections = prior.sample(n_injs)
    
    # Populate the waveform dictionary
    print("Populating the waveform dictionary with {} parameters".format(args.waveform))
    waveform_arguments = populateWaveformDict(args.waveform)

    injKeys = prior.keys() # Injection keys are identical to the prior keys
    
    for runNum in ut.np.arange(n_injs):
        # Select the injection values based on x injections
        print("Starting sample {}/{}".format(runNum+1,n_injs))
        print("=====" * 5)
        injValues = ut.np.array(list(injections.values()))[:,runNum]

        print("Populating injection parameters and adding geocentric time based on {}".format(time))
        # Populating the injection parameters dictionary
        injection_parameters = populateDict(injKeys,injValues)
        
        # Add a geocentric time
        injection_parameters["geocent_time"] = addGeocentTime(time)

        print("Setting the waveform parameters")
        # Set the parameters for the waveform generator
        duration,sampling_frequency = setWaveformParameters(args.waveform)

        # Set the output directory
        outDir = ut.get_next_available_dir(ut.os.path.join(baseDir,runDir))
        print("Setting the output directory to {}".format(outDir))

        # Make the output directory
        ut.os.mkdir(outDir)

        # Create the label for this run
        label=outDir.split("/")[-1]
        print("Run label: {}".format(label))

        print("Creating the waveform generator")
        # Creating the waveform generator
        waveform_generator = ut.bb.gw.waveform_generator.WaveformGenerator(sampling_frequency=sampling_frequency,duration=duration,
                                                                           frequency_domain_source_model=ut.get_source_model(merger_type),parameters=injection_parameters,
                                                                           waveform_arguments=waveform_arguments)
        
        print("Creating the interferometers based on {}".format(args.interferometers))
        # Instantiate the interferometers
        ifos = createInterferometers(args.interferometers)

        print("Setting the strain on the interferometers")
        # Set the strain on the interferometers
        ifos.set_strain_data_from_power_spectral_densities(duration=duration,
                                                           sampling_frequency=sampling_frequency,
                                                           start_time=injection_parameters["geocent_time"] - duration/2)

        print("Injecting GW signal into the interferometers")
        # inject a signal into the interferometers
        _ = ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

        print("Setting parameter search range")
        # Create a copy of the prior dict for the parameter space search
        priorCopy = prior
        
        # Adjust copied parameter dict as needed
        priorParameters = adjustParamDict(args.adjust_params,duration,injection_parameters,priorCopy)

        print("Computing likelihood")
        # Compute the likelihood
        likelihood = ut.bb.gw.likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                                    waveform_generator=waveform_generator)

        print("Running the sampler")
        # Run the sampler
        result = ut.bb.core.sampler.run_sampler(likelihood=likelihood,
                                                priors=priorParameters,
                                                sampler="dynesty",
                                                injection_parameters=injection_parameters,
                                                outdir=outDir,
                                                label=label,
                                                sample="act-walk",
                                                nact=2,
                                                proposals=['diff'], # I had to wrap this in a list for some reason, now it works.......
                                                bound='live-multi',
                                                npool=1024)
    return 0
        

if __name__ == "__main__":
    print("Starting")
    print("====="*10)
    
    # Run the script
    parser = argparse.ArgumentParser(description="A script to run a single bilby sample.")

    # Add arguments
    parser.add_argument("-m", "--merger_type", type=str, default="BBH",
                        help="The merger type ('BBH', 'NSBH', 'BNS').")
    parser.add_argument("-t", "--time", type=str, default="now",
                        help="The time of the merger ('now', 'randomO4').")
    parser.add_argument("-n", "--number", type=int, default=1,
                        help="The number of injections to run.")
    parser.add_argument("-b", "--baseDir", type=str, default="/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/data/{}".format(str(ut.Time.now())[:10]),
                        help="The base directory of these results.")
    parser.add_argument("-d", "--runDir", type=str, default="test_",
                        help="The specific directory where the sample is stored.")
    parser.add_argument("-w", "--waveform", type=str, default="default",
                        help="The parameters of the waveform dictionary")
    parser.add_argument("-i", "--interferometers", type=str, default="all",
                        help="The selection of interferometers to be used for the source injection")
    parser.add_argument("-a", "--adjust-params", type=str, default="default",
                        help="Any adjustments to the parameter search, different from defaults or ")

    print("Parsing arguments")
    # Parse the arguments
    args = parser.parse_args()

    # Call the run function with parsed arguments
    run(args)
