#!/bin/bash
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=168:00:00
#SBATCH -o /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/out/h_05.out
#SBATCH -e /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/err/h_05.err
#SBATCH --job-name=galcat_h_05
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=m5042
source ~/.bashrc
setup-desc-python
python /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/generate_sim_data.py /global/homes/s/seanmacb/DESC/DESC-GW/gwStreetlights/galCats/config/prod_h_05.yaml
