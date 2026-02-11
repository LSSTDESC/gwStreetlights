#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=19:00:00
#SBATCH --job-name=galcat_y4
#SBATCH --array=9
#SBATCH --array=0-8
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/y4_%a.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/y4_%a.err
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m1727
#SBATCH --account=m5042

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data_large.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_y4.yaml $SLURM_ARRAY_TASK_ID