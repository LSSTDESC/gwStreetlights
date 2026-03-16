#!/bin/bash
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/generate_combined_metrics_fiducial_%A.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/generate_combined_metrics_fiducial_%A.err
#SBATCH --job-name=generate_combined_metrics_fiducial
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_metrics_combined.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_fiducial.yaml /global/cfs/cdirs/lsst/groups/MCP/standardSirens/mockGalCats/prod_fiducial/combined.csv # Path to combined csv
