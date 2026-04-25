#!/bin/bash
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --array=0-8
#SBATCH --array=1-6
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/generate_combined_metrics_noKmeans_%A_%a.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/generate_combined_metrics_noKmeans_%A_%a.err
#SBATCH --job-name=generate_combined_metrics_fiducial
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_metrics_combined.py --kmean_label $SLURM_ARRAY_TASK_ID
