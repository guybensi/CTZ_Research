# איך מורידים את הנתונים (MODIS + CERES)

שלושה סקריפטים בתיקייה `master/scripts/` (חייבים להישאר באותה תיקייה):

| סקריפט | מה הוא עושה |
|---|---|
| `download_modis.py` | מוריד גרנולות MODIS/Aqua מ־LAADS (`MYD021KM`→`MOD02`, `MYD03`→`MOD03`, `MYD35_L2`→`MOD35`, `MYD06_L2`→`MOD06`) |
| `download_ceres.py` | מוריד CERES SSF (Aqua, Edition4A), חותך לאזור ול־17 המשתנים של ה־Order Tool (12 של המנחה ו־5 של GEOS-5) וממזג לקובץ אחד לחודש |
| `organize_ceres.py` | ממיין, ממזג ובודק קבצי CERES (משמש את `download_ceres.py`; גם לגיבוי ידני מ־Order Tool) |
| `slurm/download_year.sbatch` | job של Slurm: שנה אחת של ארבעת מוצרי MODIS ואחריהם CERES (לאשכול BGU) |

## אילו מוצרים ולמה

מורידים את **ארבעת** מוצרי MODIS. ברירת המחדל של `download_modis.py` היא רק `MYD021KM MYD03`, ולכן תמיד מעבירים
`--products MYD021KM MYD03 MYD35_L2 MYD06_L2` (ה־sbatch עושה את זה לבד).

- `MOD35` (מסכת עננים): בלעדיו `CF`, `glintf`, `landf`, `mDFNC`, `stdDFNC` יוצאים NaN ב־`build_dataset.py`, ואי אפשר לסנן אוקיינוס בלבד (`landf == 0`).
- `MOD06` (מוצר עננים): בלעדיו `meanCTH`, `stdCTH`, `LiqPhsf` יוצאים NaN.
- את `MOD06` ו־`MOD35` צריך רק בזמן ההמרה. אחרי `build_dataset.py` אפשר למחוק אותם כדי לחסוך מקום.

## פריסת הנתונים

```
<out>/AQUA/<שנה>/<אזור>/<חודש>/MOD02/   MYD021KM.*.hdf
                              /MOD03/   MYD03.*.hdf
                              /MOD35/   MYD35_L2.*.hdf
                              /MOD06/   MYD06_L2.*.hdf
                              /ssf_parts/                       החלקים השעתיים של CERES
                              /CERES_SSF_Aqua_<שנה>_<חודש>.nc
<out>/download_log.txt   <out>/manifest.csv
```
שמות החודשים: `jan feb mrch apr may june july aug sep oct nov dec`.

## נפח וזמן (Pacific, Aqua, נמדד על 2008)

| מה | לשנה |
|---|---|
| MOD02 | ~2.0 TB |
| MOD03 | ~0.4 TB |
| MOD35 + MOD06 | ~1.6 TB |
| CERES (12 קבצים ו־`ssf_parts`) | ~5 GB |
| **סך הכול** | **~4 TB** |

שנה היא 14,208 גרנולות יום. 20 שנה (2003–2022) הן כ־81 TB, או כ־48 TB אם מוחקים את MOD35/MOD06 אחרי ההמרה.
בכ־57–60 MB/s (השרת הישן, וגם הקצב הכולל מאשכול BGU) זה כ־20 שעות לשנה, ו־20 שנה הן כ־17 יום.

## השרתים

### השרת הישן (`ilai@tzag`)
```
~/GUY/CTZ_Research/          הריפו (main). הסקריפטים: master/scripts/
~/GUY/old/                   העותק הישן של download_modis.py, מלפני הריפו
/archive/GUY/                הנתונים: AQUA/2008/Pacific/<חודש>/, download_log.txt, manifest.csv, run.out, ceres_run.out
/archive/GUY/dataset/        הפלט של build_dataset.py (Pacific/scalars, Pacific/moments)
```
מצב ב־18.11.2026: כל ארבעת מוצרי MODIS ו־CERES קיימים ב־12 החודשים של 2008 (MOD02 כ־2 TB, ‏MOD03 ‏390 GB, ‏MOD06+MOD35 ‏1.6 TB, ‏CERES ‏2.7 GB). ‏MOD06 ו־MOD35 ירדו ב־16.11 (‏`--products MYD35_L2 MYD06_L2`, כ־11.5 שעות, כ־40 MB/s). ההמרה והבנצ'מרק הראשון: `master/HOW_TO_RUN.md`, סעיף 2.
`ssf_parts/` בכל חודש הם החלקים השעתיים שמהם נבנה קובץ ה־CERES (כ־200 MB לחודש). כדאי להשאיר אותם, כי `--merge-only` בונה מהם מחדש.
תיקייה `/archive/GUY/tmp_*` היא שארית של הורדה שנקטעה. אפשר למחוק אותה כשאין הורדה שרצה.
Python עם `scipy` ו־`netCDF4`: ‏`/home/ilai/miniconda3/envs/climate/bin/python`.
שם מריצים ברקע עם `nohup ... & disown` (ראה "הרצה בשרת רגיל" למטה).

### אשכול Slurm של BGU (`hpc-slurm.bgu.ac.il`)
מתחברים מרשת האוניברסיטה או מ־VPN: `ssh <username>@hpc-slurm.bgu.ac.il`. מדריך: Help ← Researcher guide בפורטל https://hpc-slurm.bgu.ac.il.
```
~/CTZ_Research/                              הריפו
~/miniforge3/envs/ctz/                       סביבת Python
/colony/bgu-eytane/users/$USER/  (= ~/colony) הנתונים: AQUA/<שנה>/Pacific/..., logs/
/colony/bgu-eytane/projects/                 תיקייה משותפת לקבוצה
```
200 TiB לכל הקבוצה. הורדה ארוכה **לא** מריצים על צומת ה־login אלא כ־job (`sbatch`).

## הכנה (פעם אחת)

1. **Python:** ‏`earthaccess`, ‏`numpy`, ‏`scipy`, ‏`netCDF4` עם תמיכה ב־HDF4 (של conda-forge):
```bash
conda create -n ctz -c conda-forge python=3.11 numpy scipy pandas netcdf4 earthaccess -y
```
2. **משתמש Earthdata** ב־`~/.netrc` (`chmod 600 ~/.netrc`):
```
machine urs.earthdata.nasa.gov login <USER> password <PASS>
```

## ארגומנטים (זהים בשני הסקריפטים)

| ארגומנט | משמעות | ברירת מחדל |
|---|---|---|
| `--start`, `--end` | תאריכים `YYYY-MM-DD` (כולל) | 2008-01-01 .. 2008-12-31 |
| `--out` | תיקיית השורש | `/archive/GUY` |
| `--region` | שם תיקיית האזור | `Pacific` |
| `--box WEST EAST SOUTH NORTH` | תיבה במעלות, `WEST > EAST` חוצה את קו התאריך | `150 -120 -60 60` |

ל־MODIS בנוסף: `--products`, `--dry-run`, `--reorganize`.
ל־CERES בנוסף: `--dry-run`, `--find`, `--inspect FILE`, `--merge-only`, `--all-hours`.

## הרצה באשכול Slurm (שנה לכל job, כמה שנים במקביל)

כל job מוריד שנה אחת: MODIS פעמיים (המעבר השני מוריד מחדש רק מה שנכשל) ואז CERES.
`download_modis.py` נועל את תיקיית ה־`--out`, ולכן כל שנה יורדת לתיקייה משלה, `dl/<שנה>/`, עם מנעול, לוג ו־`manifest.csv` משלה.
השנים מחולקות ל"מסלולים": כל מסלול הוא שם job (`-J ctz_dl_<L>`), ו־`--dependency=singleton` מריץ את השנים של אותו מסלול אחת אחרי השנייה.

```bash
mkdir -p /colony/bgu-eytane/users/$USER/logs
cd ~/CTZ_Research
N=4; for Y in $(seq 2003 2022); do L=$(( (Y-2003) % N )); OUT=/colony/bgu-eytane/users/$USER/dl/$Y \
    sbatch -J ctz_dl_$L --dependency=singleton master/scripts/slurm/download_year.sbatch $Y; done
```

**כמה מסלולים:** נמדד ב־18.11.2026: הקצב הכולל מהאשכול נעצר על כ־200 GB לשעה (כ־57 MB/s, כמו בשרת הישן), גם עם 9 הורדות במקביל.
יותר מסלולים רק מחלקים את אותו קצב. עם N במקביל כל שנה מקבלת כ־200/N GB לשעה, ושנה (4 TB) חייבת להסתיים במגבלת ה־job של 7 ימים,
ולכן N=4 עד 5 (כ־4 ימים לשנה). 20 שנה: כ־17 יום.
אם האשכול דורש מחיצה, מוסיפים `-p <partition>`. משנים נתיבים בלי לערוך את הקובץ: `OUT=... PY=... REPO=... sbatch ...`.

**מעקב:**
```bash
squeue -u $USER -o "%.6i %.10j %.2t %.10M %R"
du -sh /colony/bgu-eytane/users/$USER/dl/*/AQUA/*; date
grep -c FAILED /colony/bgu-eytane/users/$USER/dl/*/manifest.csv
```
job שנקטע: שולחים שוב את אותה שנה עם אותו `OUT`; קבצים שירדו מדולגים.
בסוף כל שנה הלוג `logs/dl_<jobid>.out` מדפיס טבלה של 12 חודשי CERES; כולם צריכים להיות `OK`. אז מעבירים את השנה לתיקייה הראשית (אותה מערכת קבצים, מיידי):
```bash
cd /colony/bgu-eytane/users/$USER && mv dl/2004/AQUA/2004 AQUA/
```
ב־`dl/<שנה>/AQUA/` מופיעה לפעמים גם תיקייה של השנה הקודמת עם גרנולה אחת (31.12 בשעה 23:55, שהחיפוש מחצות מחזיר). היא יורדת גם בשנה שלה, ואפשר למחוק אותה.

## הרצה בשרת רגיל (בלי Slurm)

1. **MODIS קודם** (CERES מוריד רק שעות שיש להן MODIS):
```bash
cd ~/GUY/CTZ_Research/master/scripts
P="MYD021KM MYD03 MYD35_L2 MYD06_L2"
python3 download_modis.py --products $P --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --dry-run
nohup python3 download_modis.py --products $P --start 2008-01-01 --end 2008-12-31 --out /archive/GUY \
    > /archive/GUY/run.out 2>&1 & disown
```
2. **בדיקה:**
```bash
pgrep -af download_modis; tail -5 /archive/GUY/download_log.txt
du -sh /archive/GUY; grep -c FAILED /archive/GUY/manifest.csv   # אם >0: להריץ שוב את אותה פקודה
```
3. **CERES** (אחרי שה־MODIS הסתיים). קודם בדיקה על קובץ גולמי אחד, כל 17 השורות חייבות להיות `OK`:
```bash
python3 download_ceres.py --inspect <קובץ SSF גולמי>
python3 download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --dry-run
nohup python3 download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY \
    > /archive/GUY/ceres_run.out 2>&1 & disown
tail -f /archive/GUY/ceres_run.out
```
בסוף מודפסת טבלה של 12 חודשים (`OK` / `PROBLEM`). מיזוג ובדיקה בלבד: אותה פקודה עם `--merge-only`.

## אזור או שנה אחרת

מריצים את שני הסקריפטים עם אותם ערכים:
```bash
python3 download_modis.py --products MYD021KM MYD03 MYD35_L2 MYD06_L2 --start 2009-03-01 --end 2009-03-31 --out /data --region Atlantic --box -60 -10 -50 50
python3 download_ceres.py --start 2009-03-01 --end 2009-03-31 --out /data --region Atlantic --box -60 -10 -50 50
```

## הערות

- הרצה חוזרת בטוחה: קבצים שכבר ירדו במלואם מדולגים; קובץ חתוך מורד מחדש. עותק אחד בלבד של `download_modis.py` רץ על `--out` (מנעול).
- הוספת מוצר לשנה שכבר ירדה (למשל MOD35/MOD06): מריצים עם `--products MYD35_L2 MYD06_L2` בלבד. MOD02/MOD03 לא נוגעים בהם.
- CERES מוריד כ־60 MB לשעה, אבל מוחק את הגולמי אחרי החיתוך.
- גיבוי ידני ל־CERES (אם ההורדה האוטומטית נכשלת): `organize_ceres.py --urls fileURLs.txt --out /archive/GUY`, ראה `master/HOW_TO_RUN.md`.
