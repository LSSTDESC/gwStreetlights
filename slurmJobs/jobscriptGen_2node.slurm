#!/bin/bash

#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J bilbyTest
#SBATCH --mail-user=sean.macbride@physik.uzh.ch
#SBATCH --mail-type=ALL
#SBATCH -A m1727
#SBATCH -t 48:00:00

# OpenMP settings:
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

mkdir -p /global/u1/s/seanmacb/DESC/DESC-GW/gwStreetlights/slurmJobs/jobs/$SLURM_JOB_ID
cd /global/u1/s/seanmacb/DESC/DESC-GW/gwStreetlights/slurmJobs/jobs/$SLURM_JOB_ID
pwd

python /global/common/software/lsst/common/miniconda/start-kernel-cli.py desc-python

#run the application: 
# srun -n 512 -c 32 --cpu_bind=cores python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/slurmJobs/runSampler.py

python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/slurmJobs/runSampler_2node.py
