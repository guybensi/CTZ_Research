#!/bin/bash
# One year of Aqua MODIS (MOD02, MOD03, MOD35, MOD06) and CERES SSF for the BGU Slurm cluster.
#
#   mkdir -p /colony/bgu-eytane/users/$USER/logs
#   N=4; for Y in $(seq 2003 2022); do L=$(( (Y-2003) % N )); OUT=/colony/bgu-eytane/users/$USER/dl/$Y \
#       sbatch -J ctz_dl_$L --dependency=singleton master/scripts/slurm/download_year.sbatch $Y; done
#
# download_modis.py allows one run per --out, so each year gets its own OUT (dl/<year>) and the years
# are spread over N lanes: jobs with the same name (-J) run one at a time (--dependency=singleton).
# The cluster's total rate is about 200 GB/h whatever N is, so keep N at 4-5 or a year (4 TB) will not
# finish within the 7-day limit. Move a finished year with: mv dl/<year>/AQUA/<year> AQUA/
# A job that stops (time limit, node failure) is resubmitted with the same year and OUT: finished files
# are skipped.
# Override the defaults by exporting OUT, PY or REPO before sbatch.
#SBATCH --job-name=ctz_dl
#SBATCH --time=7-00:00:00
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
