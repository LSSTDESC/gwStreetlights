#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/fiducial.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/fiducial.err

#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/fiducial_%a.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/fiducial_%a.err
#SBATCH --job-name=galcat_fiducial
#SBATCH --array=0-9
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data_large.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_fiducial.yaml $SLURM_ARRAY_TASK_ID
