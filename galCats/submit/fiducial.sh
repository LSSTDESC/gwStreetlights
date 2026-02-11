#!/bin/bash
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/fiducial.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/fiducial.err
#SBATCH --job-name=galcat_fiducial
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m1727

source ~/.bashrc
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_fiducial.yaml
