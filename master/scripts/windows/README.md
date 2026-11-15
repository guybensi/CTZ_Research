# בדיקה קטנה על Windows (הורדה והמרה)

התיקייה הזו מכילה את גרסת Windows של סקריפטי ההורדה וההמרה. גרסת הלינוקס, שרצה על השרת, נשארת ב־`master/scripts/` בלי שינוי.

| סקריפט | ההבדל מגרסת הלינוקס |
|---|---|
| `download_modis.py` | נעילה עם `msvcrt` (אין `fcntl` ב־Windows), `--max-granules N` להורדה קטנה |
| `download_ceres.py`, `organize_ceres.py` | מייבאים את הגרסאות של התיקייה הזו |
| `build_dataset.py` | תהליכים מקבילים עם `spawn` (אין `fork`), `--gather` עם קישורים קשיחים או העתקות כש־Windows לא מרשה symlinks |

בכולם ברירת המחדל לנתונים היא `%USERPROFILE%\GUY_DATA` (למשל `C:\Users\guy19\GUY_DATA`), מחוץ ל־OneDrive.

כל הפקודות כאן רצות ב־**Miniforge Prompt** (לא PowerShell). שינוי באחת הגרסאות צריך להיעשות גם בשנייה.

## 1. התקנה (פעם אחת)

1. **Miniforge:** להוריד ולהתקין מ־https://github.com/conda-forge/miniforge/releases (`Miniforge3-Windows-x86_64.exe`), עם ההגדרות של ההתקנה.
2. **Git:** להתקין מ־https://git-scm.com/download/win אם עוד לא מותקן.
3. לפתוח את **Miniforge Prompt** מתפריט התחל ולהריץ:
```bat
conda create -n ctz -c conda-forge python=3.11 numpy scipy pandas netcdf4 -y
conda activate ctz
pip install earthaccess
```
   חשוב: `netcdf4` מ־conda-forge ולא מ־pip, כי צריך תמיכה ב־HDF4 (קבצי MODIS).

## 2. הריפו

```bat
mkdir C:\GUY
cd /d C:\GUY
git clone https://github.com/guybensi/CTZ_Research.git
cd CTZ_Research\master
```
בפעמים הבאות: `cd /d C:\GUY\CTZ_Research\master` ואז `git pull`.

## 3. הורדת MODIS קטנה (4 גרנולות, יום אחד)

```bat
python scripts\windows\download_modis.py --start 2008-01-15 --end 2008-01-15 --max-granules 4 --products MYD021KM MYD03 MYD35_L2 MYD06_L2 --dry-run
python scripts\windows\download_modis.py --start 2008-01-15 --end 2008-01-15 --max-granules 4 --products MYD021KM MYD03 MYD35_L2 MYD06_L2
```
- **התחברות:** בפעם הראשונה הסקריפט מבקש שם משתמש וסיסמה של Earthdata, ושומר אותם לפעם הבאה.
- **נפח:** בערך 1 GB. הקבצים נכנסים ל־`%USERPROFILE%\GUY_DATA\AQUA\2008\Pacific\jan\MOD02` (וגם MOD03, MOD06, MOD35), בדיוק כמו בשרת.

**בדיקה ש־HDF4 עובד** (חייב להצליח לפני שממשיכים):
```bat
python -c "import netCDF4, glob, os; f = glob.glob(os.path.expanduser(r'~\GUY_DATA\AQUA\2008\Pacific\jan\MOD02\*.hdf'))[0]; print(netCDF4.Dataset(f).variables['EV_1KM_Emissive'].shape)"
```
צריך להודפס משהו כמו `(16, 2030, 1354)`.

## 4. הורדת CERES לשעות של הגרנולות האלה

```bat
python scripts\windows\download_ceres.py --start 2008-01-15 --end 2008-01-15 --dry-run
python scripts\windows\download_ceres.py --start 2008-01-15 --end 2008-01-15
```
הסקריפט מוריד רק את השעות שיש להן קובץ ב־`MOD02` (בערך 4 קבצים של כ־60 MB), חותך אותן, מוחק את הגולמי, ומסיים בטבלה. בשורה `jan` צריך להופיע `OK`.

## 5. המרה לדאטה של המודל

```bat
python scripts\windows\build_dataset.py --start 2008-01-15 --end 2008-01-15 --diagnose 4
python scripts\windows\build_dataset.py --start 2008-01-15 --end 2008-01-15 --workers 2
python scripts\windows\build_dataset.py --verify
```
- **`--diagnose`:** מראה לכל גרנולה כמה footprints נדחו ולמה, ואילו ערוצים לא תקינים. הוא לא בונה כלום.
- **הפלט:** `%USERPROFILE%\GUY_DATA\dataset\Pacific\{scalars,moments,images}`, וגם `Pacific\2008\jan\granules.csv` עם סטטוס לכל גרנולה.

**בדיקת התוכן:**
```bat
python -c "import numpy as np, glob, os; d = os.path.expanduser(r'~\GUY_DATA\dataset\Pacific'); f = sorted(glob.glob(d + r'\scalars\*.npz'))[0]; n = os.path.basename(f); [print(p, {k: (v.shape, str(v.dtype)) for k, v in np.load(os.path.join(d, p, n)).items()}) for p in ('scalars', 'moments', 'images')]"
python -c "from flux1km import io; import os; d = os.path.expanduser(r'~\GUY_DATA\dataset\Pacific'); ids = io.find_granules(d); print(len(ids), 'granules'); print(io.load_granule(d, ids[0]).n_footprints, 'footprints in', ids[0])"
```
המפתחות והטיפוסים צריכים להיות כמו ב־`data/sample/` (23 סקלרים, 10 מומנטים, 4 תמונות).

## 6. ניקוי

כשהכול עובד ואין צורך יותר בנתוני הבדיקה:
```bat
rmdir /s /q "%USERPROFILE%\GUY_DATA"
```
זה מוחק רק את נתוני הבדיקה במחשב. הריפו והשרת לא נפגעים.
