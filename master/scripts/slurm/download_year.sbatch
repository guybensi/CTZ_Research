#!/bin/bash
# One year of Aqua MODIS (MOD02, MOD03, MOD35, MOD06) and CERES SSF for the BGU Slurm cluster.
#
#   mkdir -p /colony/bgu-eytane/users/$USER/logs
#   sbatch --dependency=singleton master/scripts/slurm/download_year.sbatch 2003
#   for Y in $(seq 2003 2022); do sbatch --dependency=singleton master/scripts/slurm/download_year.sbatch $Y; done
#
# All jobs share the name ctz_dl, so --dependency=singleton runs them one at a time
# (download_modis.py allows one run per --out anyway). A job that stops (time limit, node failure)
# is resubmitted with the same year: finished files are skipped.
# Override the defaults by exporting OUT, PY or REPO before sbatch.
#SBATCH --job-name=ctz_dl
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/colony/bgu-eytane/users/%u/logs/dl_%j.out

Y=${1:?usage: sbatch download_year.sbatch <year>}
OUT=${OUT:-/colony/bgu-eytane/users/$USER}
PY=${PY:-$HOME/miniforge3/envs/ctz/bin/python}
REPO=${REPO:-$HOME/CTZ_Research}
PRODUCTS="MYD021KM MYD03 MYD35_L2 MYD06_L2"

cd "$REPO/master/scripts" || exit 1
echo "=== year $Y start $(date) host $(hostname) out $OUT"
$PY download_modis.py --products $PRODUCTS --start "$Y-01-01" --end "$Y-12-31" --out "$OUT"
echo "=== second pass (retries failed files) $(date)"
$PY download_modis.py --products $PRODUCTS --start "$Y-01-01" --end "$Y-12-31" --out "$OUT"
echo "=== CERES $(date)"
$PY download_ceres.py --start "$Y-01-01" --end "$Y-12-31" --out "$OUT"
echo "=== year $Y done $(date)"
