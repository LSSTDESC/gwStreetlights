#!/bin/bash
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/small_fiducial.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/small_fiducial.err
#SBATCH --job-name=small_galcat_fiducial
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m1727

source ~/.bashrc
# setup-desc-python

python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/test_small.yaml
