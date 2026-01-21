#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/fiducial.out
#SBATCH --error=/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/fiducial.err
#SBATCH --job-name=galcat_fiducial
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH --account=m1727

python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_fiducial.yaml