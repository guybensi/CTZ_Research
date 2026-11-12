# איך בונים את הדאטה לאימון (סקלרים, מומנטים, תמונות)

`build_dataset.py` הוא הגרסה המתוקנת של הממיר של המנחה (`pre_master/advisor_code/Generate_AllSky_2.py` ו־`MOD_CERES_allsky` ב־`Twilight_module.py`). הקבצים המקוריים נשארו כמו שהם, כי `pre_master/` קפוא לפי `CLAUDE.md`. הפלט זהה לשלו במפתחות, בצורות ובטיפוסים, אז כל מה שקורא את הדאטה שלו (`flux1km`, `All_Sky_AIflux.py`) קורא גם את זה.

## מה מקבלים

```
<out>/<אזור>/<שנה>/<חודש>/scalars/<granule>.npz     ערך אחד לכל footprint של CERES (23 מפתחות)
                         /moments/<granule>.npz     mean/std/skew/kurt משוקללי PSF של 11 ערוצי SW ו־16 ערוצי LW
                         /images/<granule>.npz      טלאים של 27x55 (אפשר לוותר עם --no-images)
                         /granules.csv              מה קרה לכל גרנולה (ok / empty / error)
<out>/<אזור>/{scalars,moments,images}/              קישורים לכל הקבצים של האזור: שלוש התיקיות שהמודלים קוראים
```
את המודלים מריצים עם `--data-root <out>/Pacific`.

## לפני ההרצה

1. **כל נתוני ההורדה במקום:** `MOD02`, `MOD03`, `MOD06`, `MOD35` וקובץ `CERES*.nc` אחד בכל תיקיית חודש (`download_modis.py` ו־`download_ceres.py`).
   - אם אין `MOD06` או `MOD35`, הגרנולה נבנית בכל זאת. ב־`MOD35` חסר: `CF`, `glintf`, `landf`, `mDFNC`, `stdDFNC` יוצאים NaN. ב־`MOD06` חסר: `meanCTH`, `stdCTH`, `LiqPhsf` יוצאים NaN. רוצים את `MOD35` כדי לסנן אוקיינוס בלבד (`landf == 0`), ולכן כדאי להוריד אותו: `download_modis.py --products MYD021KM MYD03 MYD35_L2 MYD06_L2`.
2. **קובץ ה־PSF `ceres_psf.csv`** של המנחה כבר בריפו, ב־`master/resources/ceres_psf.csv`, והסקריפט משתמש בו כברירת מחדל. אפשר להחליף עם `--psf <קובץ>`, או `--psf ellipse` לאליפסה מקורבת (לבדיקות בלבד).
3. **חבילות:** `numpy`, `scipy`, `pandas`, ו־`netCDF4` מ־conda-forge עם תמיכה ב־HDF4. אין צורך ב־`matplotlib` ולא ב־`scikit-image` (הממיר של המנחה דורש אותן).

## הרצה

```bash
cd ~/GUY/CTZ_Research/master
python3 scripts/build_dataset.py --data /archive/GUY --out /archive/GUY/dataset --dry-run
```
בדיקה קטנה (3 גרנולות בכל חודש, חודש אחד):
```bash
python3 scripts/build_dataset.py --data /archive/GUY --out /archive/GUY/dataset_test \
    --start 2008-01-01 --end 2008-01-31 --limit 3 --workers 2
```
הרצה מלאה ברקע (אפשר לסגור את הטרמינל):
```bash
nohup python3 scripts/build_dataset.py --data /archive/GUY --out /archive/GUY/dataset \
    --start 2008-01-01 --end 2008-12-31 --workers 8 > /archive/GUY/build_run.out 2>&1 & disown
tail -f /archive/GUY/dataset/build_log.txt
```
- **זיכרון וזמן:** כל worker צריך בערך 1.5 GB, וגרנולה אחת לוקחת בערך חצי דקה (4,000 footprints). `--workers` לפי הזיכרון של השרת.
- **דיסק:** `images/` הוא העיקרי, בערך 180 KB ל־footprint. כשחסר מקום אפשר `--no-images`.
- **הרצה חוזרת בטוחה:** גרנולות שנגמרו מדולגות, וגרנולות שנכשלו (`error` ב־`granules.csv`) נבנות שוב. גרנולה בלי footprints (`empty`) לא נבנית מחדש.
- **בדיקה וקישורים:** `--verify` בודק כל שלישיית קבצים. `--gather` בונה מחדש את שלוש התיקיות אם הוספת חודשים.

```bash
python3 scripts/build_dataset.py --out /archive/GUY/dataset --gather
python3 scripts/build_dataset.py --out /archive/GUY/dataset --verify
```

## מה תוקן לעומת הקוד של המנחה

הקוד שלו הורץ על נתונים אחרים. הנה מה שלא עבד או שגוי, ומה נעשה בגרסה הזו:

| # | הבעיה בקוד המקורי | מה קורה בגרסה המתוקנת |
|---|---|---|
| 1 | `swallnanraw=~np.nonzero(np.isnan(swF))[0]`: ה־`~` על אינדקסים הופך אותם לאינדקסים מהסוף, ולכן נשמרות שורות לא נכונות: שורות ריקות (NaN) ו־footprints חסרים | נשמרים כל ה־footprints שנבנו |
| 2 | מילוי היום בשנה: `if dayj<100: dayj='0'+str(dayj)` נותן `05` במקום `005` לימים 1 עד 9, ולכן אף footprint לא מותאם לימים האלה | התאמה לפי מפתח שלם (שנה, יום, שעה) |
| 3 | גרנולה שחוצה את קו התאריך (180°) לא מתאימה אף footprint (החלפת מזרח ומערב הופכת את התנאי לבלתי אפשרי). האוקיינוס השקט בדיוק שם | חיפוש הפיקסל הקרוב על כדור הארץ (KD-tree), בלי קפיצת אורך |
| 4 | `LiqPhsf` מחלק ספירה של כל הגרנולה (`CldPhs`) בספירה של ה־footprint, ויוצא בערך 400 | `LiqPhsf` הוא חלק הפיקסלים המעוננים עם פאזה נוזלית, בין 0 ל־1 |
| 5 | פענוח ביטים של MYD35: מחרוזת הפוכה, ולכן קוד היבשה/ים הפוך (חוף עם מדבר), והביטים של `byte2` נקראים מהמחרוזת הלא נכונה | פענוח בפעולות ביט (`landf == 0` עדיין אומר "כל המים") |
| 6 | פיקסלים ש־MYD35 לא קבע להם מסכה נספרים כ־glint (`1-glint`) | glint רק כשהמסכה נקבעה |
| 7 | `CERESraa[fov+1]` בלי בדיקת גבול: קריסה ב־footprint האחרון של קובץ | ה־footprint האחרון מדולג |
| 8 | הטלאי מוסט בעמודה אחת: שיא ה־PSF נופל על הפיקסל שליד מרכז ה־footprint (`PSFc-1` נשאר מהמרת אינדקס 1 מבוסס) | שיא ה־PSF על מרכז ה־footprint. `PSF_COLUMN_OFFSET = 1` בראש הקובץ משחזר בדיוק את הקוד שלו |
| 9 | `cldmskfov=BWmsk[:subsubRi:, ...]`: טעות חיתוך כשהטלאי נוגע בקצה התחתון | חיתוך אחד לכל המשתנים |
| 10 | ערכי MYD021KM של 65533 ו־65534 (רוויה/לא שמיש) לא מוסרים, רק 65535 | כל מה שמעל 32767 הוא NaN |
| 11 | מערכים קבועים של 10,000 footprints לגרנולה | רשימות, בלי גבול |
| 12 | `byte1[byte1<0]=256+byte1[byte1<0]` נכשל ב־numpy 2 (`OverflowError`) | פעולות ביט שעובדות ב־numpy 1 וב־2 |
| 13 | ה־RAA של התמונות יכול לחרוג מ־180° | קיפול ל־0 עד 180 |
| 14 | חיפוש הפיקסל הקרוב על כל הגרנולה (2.7 מיליון פיקסלים) לכל footprint פעמיים, לולאת פייתון על כל פיקסל לפענוח המסכה, והמערך של CERES של כל החודש נשלח לכל גרנולה (גיגה־בייטים) | KD-tree, פענוח וקטורי, ולכל גרנולה נשלחת רק השעה שלה |
| 15 | הדרייבר: נתיבים קבועים בשרת שלו, `os.makedirs` בלי `exist_ok` (קורס בהרצה שנייה), גרנולה אחת שנכשלת (למשל בלי קובץ MOD06) מפילה את כל החודש, בלי המשך מנקודת עצירה ובלי לוג | ארגומנטים, `try/except` לכל גרנולה, המשך, `granules.csv` ו־`build_log.txt` |

**לא שונה, כדי להישאר תואם לדאטה ולקוד הישנים:**
- פיקסל שבו המסכה לא נקבעה נחשב מעונן.
- `meanCTH`, `stdCTH`, `mDFNC` ו־`stdDFNC` הם 0 (לא NaN) כשאין פיקסל מתאים, כי `All_Sky_AIflux.py` ו־`build_legacy_dataset_from_pairs.py` משתמשים בהם.
- `landf` הוא ממוצע קוד, לא אחוז.
- footprint עם `Fsw` NaN אבל עם `Flw` תקין נשאר.
- הטיית והבלטות (skew, kurt) של ערוץ קבוע יוצאות 0 ו־‎−3, כמו אצלו.

## מה נבדק, ומה לא

- **74 בדיקות** (`tests/test_build_dataset.py` ו־doctests). הן כוללות גרנולה שחוצה את קו התאריך, אזור ליד הקצה, פיקסלים רוויים, הרצה חוזרת, גרנולה שנכשלת, וקריאה של הפלט עם `flux1km.io`.
- **השוואה לקוד המקורי:** הרצתי את `MOD_CERES_allsky` של המנחה (עם תיקון הטעות ב־`~np.nonzero`, ובלי שורות ה־`256+` שנכשלות ב־numpy 2) ואת הגרסה הזו על אותה גרנולה סינתטית עם `PSF_COLUMN_OFFSET=1`. כל 23 הסקלרים, כל המומנטים, התמונות וה־`FOVorientation` יצאו זהים, חוץ מ־`LiqPhsf`, שתוקן בכוונה.
- **לא נבדק על קבצי HDF4 אמיתיים.** לא היו לי קבצי MODIS אמיתיים, אז שמות המשתנים ב־MOD02/03/06/35 נלקחו מהקוד שלו. מומלץ להריץ קודם את הבדיקה הקטנה (`--limit 3`) ולהשוות לשלישייה של המנחה, `data/sample/` (גרנולה A2014060.0245).
