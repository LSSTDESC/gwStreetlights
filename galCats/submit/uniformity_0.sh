#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/uniformity_00.out
#SBATCH --error=/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/uniformity_00.err
#SBATCH --job-name=galcat_uniformity_00
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH --account=m1727

python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_uniformity_00.yaml