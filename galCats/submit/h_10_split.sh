#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/h_10_%a.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/h_10_%a.err
#SBATCH --job-name=galcat_h_10
#SBATCH --array=0-9
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data_large.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_h10.yaml $SLURM_ARRAY_TASK_ID
