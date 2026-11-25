# העברת העבודה לשרת אחר

המסמך הזה מסכם את מה שצריך כדי להריץ את כל הצינור (הורדה, המרה, benchmark) על שרת חדש. הוא כתוב כך שלא יהיה תלוי בשרת הנוכחי (`ilai@tzag`). כל המספרים נמדדו על השרת הנוכחי בנובמבר 2026, ומופיעים כאן כדי שאפשר יהיה לחשב מראש מה ייכנס בשרת החדש.

## 1. מה תלוי בשרת, ואיפה משנים אותו

שום נתיב לא קבוע בקוד. כל נתיב הוא ארגומנט של פקודה, ולערכים בעמודה "ברירת מחדל" צריך לשים לב רק אם לא מעבירים את הארגומנט.

| מה | ארגומנט | ברירת מחדל (מותאמת לשרת הנוכחי) |
|---|---|---|
| שורש הנתונים הגולמיים | `download_modis.py --out`, ‏`download_ceres.py --out`, ‏`build_dataset.py --data` | `/archive/GUY` |
| שורש הדאטה שנבנה (scalars/moments/images) | `build_dataset.py --out` | `/archive/GUY/dataset` |
| הדאטה למודלים | `flux1km.cli --data-root` | אין, חובה |
| מטמון זמני להרצת מודלים | `flux1km.cli --cache-dir` | אין (בלי מטמון הכול בזיכרון) |
| תוצאות | `flux1km.cli --out` | `benchmark_outputs/flux1km` |
| מספר תהליכים בהמרה | `build_dataset.py --workers` | 4 |
| אזור ותיבה | `--region`, ‏`--box` בשלושת הסקריפטים | `Pacific`, ‏`150 -120 -60 60` |

הנתיבים הספציפיים לשרת הנוכחי מופיעים רק בתיעוד: בטבלה "מה הורץ על השרת" ב־`HOW_TO_RUN.md`, ובסעיף "בשרת" ב־`scripts/HOW_TO_DOWNLOAD.md`. בשרת חדש מוסיפים שם שורה חדשה, ולא משנים את הקוד.

## 2. כמה מקום, זיכרון וזמן צריך

### דיסק, לשנה אחת של אזור Pacific ‏(Aqua, ‏14,208 גרנולות יום)

| שלב | נפח | אפשר למחוק אחרי? |
|---|---|---|
| MOD02 (קרינה) | 2.0 TB | צריך אותו לבניית תמונות. אם כל התמונות כבר נבנו, אפשר |
| MOD03 (גאומטריה) | 0.4 TB | כמו MOD02 |
| MOD06 + MOD35 | 1.6 TB | כן, אחרי ההמרה. אבל בנייה חוזרת של גרנולה, למשל כדי להוסיף לה תמונות, צריכה אותם |
| CERES (12 קבצים ו־`ssf_parts`) | 5 GB | `ssf_parts` אחרי המיזוג, כן |
| scalars + moments | 26 GB | לא, אלה הדאטה של המודלים |
| images (כל הגרנולות) | 6.2 TB, כ־470 MB לגרנולה | לא |
| מטמון של benchmark ‏(`--cache-dir`) | בשיא: התמונות של ה־fold, כ־180 KB לכל footprint, ועוד הקלט של האימון שרץ באותו רגע, עד כ־475 KB לכל footprint של אימון ב־M4. ב־6 גרנולות זה כ־10 עד 30 GB | כל אימון מוחק את הקבצים שלו, וה־fold מוחק את התיקייה שלו בסוף. הקוד משאיר תמיד לפחות `CTZ_CACHE_MIN_FREE_GB` פנויים, ברירת מחדל 20 GB |

**דוגמה לשרת עם 5 TB:** שנה של נתונים גולמיים (4 TB) ועוד סקלרים ומומנטים נכנסים, אבל אז נשאר מקום לתמונות של כ־2 חודשים בלבד. אפשרויות:
1. **מוחקים MOD06+MOD35 אחרי ההמרה** (חוסך 1.6 TB, ומשאיר מקום לכ־5 חודשי תמונות). החיסרון: כל בנייה חוזרת של גרנולה, כולל הוספת תמונות אחר כך, תצטרך להוריד אותם שוב, אז את התמונות כדאי לבנות לפני המחיקה.
2. **מורידים ובונים חודש אחרי חודש:** הורדה, המרה עם תמונות, ואז מחיקת הגולמי של אותו חודש. כך נשארים רק scalars, ‏moments ותמונות, כ־0.5 TB לחודש.
3. **בונים תמונות רק לחלק מהגרנולות.** אין צורך בכולן: המודלים מתאמנים על מדגם.

**סוג הדיסק:** ההמרה מוגבלת על ידי קריאה מהדיסק. למטמון של ה־benchmark עדיף מאוד SSD/NVMe מקומי, כי האימון קורא שורות בסדר אקראי. דיסק רשת או HDD יאטו אותו מאוד.

### זיכרון (RAM)

| שלב | זיכרון |
|---|---|
| הורדה | זניח |
| המרה (`build_dataset.py`) | כ־1.5 GB לכל worker. לכן `--workers` נקבע לפי `free -g`, לא לפי מספר הליבות |
| benchmark **בלי** `--cache-dir` | כ־0.6 עד 0.7 MB לכל footprint. 6 גרנולות עם תמונות כבר עלולות למלא 30 GB |
| benchmark **עם** `--cache-dir` | כמה GB, ולא גדל עם כמות הנתונים. נמדד לכל שיטה בנפרד: 0.4 עד 3.4 GB |
| benchmark בלי תמונות (רק שיטות footprint) | 676 גרנולות, 3 מיליון footprints: כ־9 GB, ועוד רשומות תחזית של כ־1 KB לכל footprint לכל שיטה |

**תמיד מריצים עם `--cache-dir` כשיש תמונות.** התוצאות זהות לריצה בלי מטמון, יש על כך בדיקות.

### GPU
רשתות הנוירונים (M0-allsky37, ‏M0-advisor, ‏M1-ann, ‏M3, ‏M5, ‏M6a עד M6d, ‏M6f) משתמשות ב־GPU אוטומטית אם הוא קיים, ובלי GPU הן רצות על המעבד, לאט הרבה יותר. 24 GB של זיכרון GPU מספיקים בשפע, כי הנתונים נשלחים אליו במנות קטנות.

### זמן (נמדד על השרת הנוכחי: 32 ליבות, ‏RTX PRO 4000 Blackwell)
| שלב | זמן |
|---|---|
| הורדת MOD35+MOD06 לשנה (1.6 TB) | כ־11.5 שעות (כ־40 MB/s מ־LAADS) |
| המרה בלי תמונות, שנה, ‏12 workers | כ־40 שעות (כ־6 גרנולות לדקה) |
| המרה עם תמונות, 6 חודשים, ‏12 workers | כ־24 שעות (כ־4.6 גרנולות לדקה) |
| benchmark של 5 שיטות footprint על 676 גרנולות | כ־5 שעות |
| benchmark של כל השיטות על 6 גרנולות עם תמונות | fold אחד כ־2.5 שעות, כלומר כ־15 שעות ל־6 folds |

זמן ה־benchmark גדל בערך ביחס ישר למספר ה־footprints, כי כל רשת עוברת על כל הנתונים בכל epoch. אם מכפילים את הנתונים, מכפילים גם את הזמן.

## 3. התקנה בשרת חדש

1. **הריפו:**
```bash
git clone https://github.com/guybensi/CTZ_Research.git && cd CTZ_Research/master
```
2. **סביבת Python נפרדת** (לא ב־`base` של שרת משותף):
```bash
conda create -n ctz -c conda-forge python=3.11 numpy scipy pandas scikit-learn netcdf4 matplotlib -y
```
‏`netcdf4` מ־conda-forge ולא מ־pip, כי צריך תמיכה ב־HDF4 בשביל קבצי MODIS.
3. **PyTorch:** בשרת הנוכחי PyPI היה איטי מאוד (כ־50 KB/s), אז ההתקנה הייתה מהשרת של PyTorch עצמו. הגרסה `cu128` מתאימה לדרייבר שתומך ב־CUDA 12.8 ומעלה, כולל כרטיסי Blackwell. את גרסת ה־CUDA בודקים עם `nvidia-smi`, בפינה הימנית העליונה.
```bash
~/miniconda3/envs/ctz/bin/pip install earthaccess && ~/miniconda3/envs/ctz/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
```
4. **בדיקה:**
```bash
~/miniconda3/envs/ctz/bin/python -c "import sklearn, pandas, torch, netCDF4; print('GPU:', torch.cuda.is_available())"; ~/miniconda3/envs/ctz/bin/python -m flux1km.cli --demo --models M0-ridge M1-gbm M5 --out /tmp/ctz_demo && rm -rf /tmp/ctz_demo
```
5. **Earthdata:** קובץ `~/.netrc` עם `machine urs.earthdata.nasa.gov login <USER> password <PASS>`, ואחריו `chmod 600 ~/.netrc`.

## 4. סדר העבודה

| שלב | פקודה (מ־`master/`) | מדריך |
|---|---|---|
| הורדה | `scripts/download_modis.py --products MYD021KM MYD03 MYD35_L2 MYD06_L2 --start ... --end ... --out <ROOT>`, ואחריו `scripts/download_ceres.py` עם אותם ערכים | `scripts/HOW_TO_DOWNLOAD.md` |
| המרה | `scripts/build_dataset.py --data <ROOT> --out <ROOT>/dataset --start ... --end ... --workers N [--no-images]` | `scripts/HOW_TO_BUILD_DATASET.md` |
| מדגם | תיקייה עם symlinks לחלק מהגרנולות, למשל כל גרנולה עשרים (הפקודה ב־`HOW_TO_RUN.md` סעיף 2) | `HOW_TO_RUN.md` |
| benchmark | `python -m flux1km.cli --data-root <SUBSET> --split day --k 5 --manifest <SUBSET>/split_day5.json --psf resources/ceres_psf.csv --cache-dir <FAST_LOCAL_DISK> --out <RESULTS>` | `HOW_TO_RUN.md`, ‏`MODELS.md` |

כל שלב ארוך מריצים ברקע עם `nohup ... > LOG 2>&1 & disown`. כך הוא ממשיך גם אם החיבור נופל, ושלב שנקטע ממשיך מאיפה שעצר כשמריצים אותו שוב.

## 5. מעקב ותקלות שכבר ראינו

| תקלה | הסימן | מה עושים |
|---|---|---|
| הדיסק של המטמון התמלא | `df -h` מראה 100%. ב־22.11 זה קרה ב־`/` של השרת, כי קבצים של כל אימון נשארו עד סוף ה־fold | תוקן בקוד: מחיקה אחרי כל אימון, ובדיקת מקום שעוצרת שיטה עם שגיאה ברורה. אם זה קורה בכל זאת: עוצרים ומוחקים את `<cache-dir>/fold_*`. התוצאות לא שם, הן נכתבות ל־`--out` רק בסוף ההרצה |
| שיטה אחת לוקחת שעות | בלוג, `done in 471 min` אצל M4 | M4 מחשבת חיזוי לכל פיקסלי האימון בכל סבב boosting. בהרצות גדולות מריצים אותה בנפרד או משמיטים אותה (`--models`) |
| נגמר הזיכרון | התהליך נעלם בלי `Traceback`, ותיקיית `fold_*` נשארת ב־`--cache-dir` | מריצים עם `--cache-dir`. לפי שורות ההתקדמות בלוג רואים באיזו שיטה זה קרה |
| הקוד בשרת לא מעודכן | `unrecognized arguments` | `git pull` ובדיקה עם `git log -1` |
| `git pull` נכשל | `Your local changes would be overwritten` | `git stash`, אחר כך `git pull`, ואחר כך `git stash show -p` כדי להחליט אם לשמור את השינוי |
| PyPI איטי | הורדה בקצב של kB/s | `--index-url https://download.pytorch.org/whl/cu128` |
| אין הרשאת push ל־GitHub מהשרת | השרת מבקש שם משתמש | מעבירים את `results.txt` בדרך אחרת. קבצים מעל 100 MB, כמו `predictions.csv` או `split_*.json`, לא נכנסים ל־GitHub בכלל |
| MOD06/MOD35 חסרים | `granules.csv` מראה `no MOD35/MOD06` | מורידים אותם **לפני** ההמרה. הרצה חוזרת לא בונה מחדש גרנולה שכבר הצליחה |

שורות ההתקדמות של ה־benchmark (`fold i/n ...`, ‏`<method> seed 0: done in X min  mem ... (machine available ...)`) מראות את הזמן ואת הזיכרון של כל שיטה. מהן אפשר לחשב את הזמן הכולל ולזהות שיטה שתופסת הרבה זיכרון.
