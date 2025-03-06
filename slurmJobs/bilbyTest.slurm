#!/bin/bash
#SBATCH --job-name=bilby_test_0
#SBATCH --time=02:00:00           # Time limit: 2 hours
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=32               # Total number of tasks
#SBATCH --cpus-per-task=1         # CPUs per task
#SBATCH --mem=64G                 # Memory allocation
#SBATCH --partition=system           # Use CPU partition (or change to 'gpu' if needed)
#SBATCH --qos=system
#SBATCH --account=m1727           # Replace with your NERSC account
#SBATCH --output=bilby_output_0.log
#SBATCH --error=bilby_error_0.log

# Load modules (if needed)
module load python
module load mpi

# Activate your bilby environment
source $(conda info --base)/etc/profile.d/conda.sh && conda activate mydesc

# Run the bilby sampler
srun -n $SLURM_NTASKS python your_bilby_script.py