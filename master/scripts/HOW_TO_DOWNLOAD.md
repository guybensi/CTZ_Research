# איך מורידים את הנתונים (MODIS + CERES)

שלושה סקריפטים בתיקייה `master/scripts/` (חייבים להישאר באותה תיקייה):

| סקריפט | מה הוא עושה |
|---|---|
| `download_modis.py` | מוריד גרנולות MODIS/Aqua מ־LAADS (`MYD021KM`→`MOD02`, `MYD03`→`MOD03`, אופציונלית `MYD06_L2`→`MOD06`, `MYD35_L2`→`MOD35`) |
| `download_ceres.py` | מוריד CERES SSF (Aqua, Edition4A), חותך לאזור ול־12 המשתנים של המנחה וממזג לקובץ אחד לחודש |
| `organize_ceres.py` | ממיין, ממזג ובודק קבצי CERES (משמש את `download_ceres.py`; גם לגיבוי ידני מ־Order Tool) |

## פריסת הנתונים

```
<out>/AQUA/<שנה>/<אזור>/<חודש>/MOD02/   MYD021KM.*.hdf
                              /MOD03/   MYD03.*.hdf
                              /MOD06/  /MOD35/
                              /CERES_SSF_Aqua_<שנה>_<חודש>.nc
```
שמות החודשים: `jan feb mrch apr may june july aug sep oct nov dec`.

## הכנה (פעם אחת)

```bash
conda activate base            # או הסביבה שלך
pip install earthaccess
# netCDF4 עם תמיכה ב-HDF4 (ל-CERES): conda install -c conda-forge netcdf4
```
משתמש Earthdata ב־`~/.netrc` (`chmod 600 ~/.netrc`):
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

## הרצה

1. **MODIS קודם** (CERES מוריד רק שעות שיש להן MODIS):
```bash
cd ~/GUY
python3 download_modis.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --dry-run
nohup python3 download_modis.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY \
    > /archive/GUY/run.out 2>&1 & disown
```
2. **בדיקה:**
```bash
pgrep -af download_modis; tail -5 /archive/GUY/download_log.txt
du -sh /archive/GUY; grep -c FAILED /archive/GUY/manifest.csv   # אם >0: להריץ שוב את אותה פקודה
```
3. **CERES** (אחרי שה־MODIS הסתיים). קודם בדיקה על קובץ גולמי אחד, כל 12 השורות חייבות להיות `OK`:
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
python3 download_modis.py --start 2009-03-01 --end 2009-03-31 --out /data --region Atlantic --box -60 -10 -50 50
python3 download_ceres.py --start 2009-03-01 --end 2009-03-31 --out /data --region Atlantic --box -60 -10 -50 50
```

## הערות

- הרצה חוזרת בטוחה: קבצים שכבר ירדו במלואם מדולגים; קובץ חתוך מורד מחדש. עותק אחד בלבד רץ על `--out` (מנעול).
- נפח: בערך 2.3 TB ל־MODIS של שנה, ו־CERES מוריד כ־60 MB לשעה אבל מוחק את הגולמי אחרי החיתוך.
- אפשר לסגור את הטרמינל אחרי `nohup ... & disown`.
- גיבוי ידני ל־CERES (אם ההורדה האוטומטית נכשלת): `organize_ceres.py --urls fileURLs.txt --out /archive/GUY`, ראה `master/HOW_TO_RUN.md`.
