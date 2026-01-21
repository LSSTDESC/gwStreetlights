#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/uniformity_02.out
#SBATCH --error=/global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/uniformity_02.err
#SBATCH --job-name=galcat_uniformity_02
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH --account=m1727

python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_uniformity_02.yaml