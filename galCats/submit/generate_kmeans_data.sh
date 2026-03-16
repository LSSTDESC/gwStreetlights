#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --array=0-8
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/kmeans_%A_%a.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/kmeans_%A_%a.err
#SBATCH --job-name=kmeans_generation

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_kmeans_data.py -i $SLURM_ARRAY_TASK_ID
