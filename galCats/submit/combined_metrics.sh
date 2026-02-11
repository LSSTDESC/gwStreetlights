#!/bin/bash
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/prod_fiducial_combined_metrics_%A.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/prod_fiducial_combined_metrics_%A.err
#SBATCH --job-name=galcat_fiducial_combined_metrics
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_metrics_combined.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_fiducial.yaml /pscratch/sd/s/seanmacb/SkySim_realizations/prod_fiducial/combined_data.csv
