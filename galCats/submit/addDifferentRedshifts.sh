#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --array=0-8
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/addDifferentRedshifts_spectro_%A_%a.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/addDifferentRedshifts_spectro_%A_%a.err
#SBATCH --job-name=addDifferentRedshifts_spectro
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/addDifferentRedshifts.py $SLURM_ARRAY_TASK_ID --manifest prodCombinedPaths.txt --input_col redshift_true --output_col redshift_spectro
# python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/addDifferentRedshifts.py $SLURM_ARRAY_TASK_ID --manifest prodCombinedPaths.txt --input_col redshift_measured --output_col redshift_modeled
