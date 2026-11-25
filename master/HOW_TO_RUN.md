# מדריך הרצה: מה כל דבר ואיך מריצים אותו

המדריך הזה מיועד למי שרוצה להריץ: מה יש בכל קובץ, איזו פקודה מריצה אותו, ומה יוצא ממנו.

- ההסבר המדעי המלא של כל שיטה נמצא ב־[MODELS.md](MODELS.md).
- התוכנית נמצאת ב־[PLAN.md](PLAN.md), וסטטוס המימוש שלה בסעיף 10 שם.
- כל הפקודות כאן רצות מתוך התיקייה `master/`.
- העברה לשרת אחר (דיסק, זיכרון, זמן, התקנה ותקלות שכבר ראינו): [NEW_SERVER.md](NEW_SERVER.md).

---

## 1. התקנה

```bash
cd master
pip install -e ".[dev,ml]"
```

- `dev`: ‏pytest, ‏ruff, ‏scikit-learn ו־pandas.
- `ml`: ‏PyTorch. הוא נחוץ רק לרשתות: M0 ‏(ANN), ‏M1-ann, ‏M3, ‏M5, ‏M6a–M6d ו־M6f. בלי `ml` המודלים האלה מדולגים עם הודעה, וכל השאר רץ כרגיל.
- על GPU מריצים את הדוקר מהשורש. אחרי השלב הזה עוברים לתיקייה `master/`, כי כל הפקודות במדריך רצות ממנה:

```bash
docker build -t ctz_research:gpu .
docker run --gpus all --rm -it -v "$PWD":/app ctz_research:gpu
```

---

## 2. סדר עבודה מומלץ כשהדאטה מגיע

| שלב | פקודה | מה יוצא |
|---|---|---|
| 1. פריסה ובדיקת הדאטה (E0) | `python3 -m flux1km.e0 --zip Data_toGuy.zip --data-root data/paired --out benchmark_outputs/e0` | `E0_report.txt`, ‏`split_manifest.json`, ‏`fitted_operator.npz`, ‏`advisor_bug_report.txt` |
| 2. הרצה מחדש של M0 בפרוטוקול החדש | `python3 -m flux1km.e0 --data-root data/paired --run-baselines` | סעיף 8 ב־`E0_report.txt` |
| 3. כל השיטות | `python3 -m flux1km.cli --data-root data/paired --manifest benchmark_outputs/e0/split_manifest.json` | `run_<זמן>/results.txt` |
| 4. אמת סינתטית | `python3 -m flux1km.synthetic_study --models M2 M3 M5 M6b` | `synthetic_truth_results.txt` |
| 5. אסטרטגיות דגימה, עם 2–3 השיטות המובילות | `python3 -m flux1km.sampling_study --data-root data/paired --manifest benchmark_outputs/e0/split_manifest.json --models M2 M3 M9` | `sampling_results.txt` |
| 6. אימון מחדש של המנצח עם 5 seeds | `python3 -m flux1km.methods.<קובץ> --data-root data/paired --manifest ... --seeds 0 1 2 3 4` | `results.txt` |

הקובץ `ceres_psf.csv` של המנחה כבר בריפו, ב־`resources/ceres_psf.csv`, ומוסיפים אותו לכל פקודה: `--psf resources/ceres_psf.csv`. (`benchmark_outputs/e0/fitted_operator.npz` הוא ה־PSF ששוחזר בשלב 1, כגיבוי.)

כל פקודה עובדת גם עם `--demo` במקום `--data-root`. אז היא רצה על דאטה סינתטי עם אמת ידועה ב־1 ק"מ, ו־`--demo-hard` מריץ את הגרסה הקשה (ראו סעיף 5).

### מה הורץ על השרת (`ilai@tzag`)

| מתי | מה | איפה | מצב |
|---|---|---|---|
| 16–18.11.2026 | הורדה: MODIS Aqua ‏(MOD02, MOD03, MOD06, MOD35) ו־CERES SSF לכל 2008, ‏Pacific (150E עד 120W, ‏60S עד 60N) | `/archive/GUY/AQUA/2008/Pacific/` (כ־4 TB) | הסתיים, ‏14,208 גרנולות יום |
| 16–18.11.2026 | המרה: `build_dataset.py --no-images --workers 12` (בלי תמונות: כ־6 TB לשנה, לא נכנס בדיסק) | `/archive/GUY/dataset/Pacific/{scalars,moments}` (‏3.8 GB ו־22 GB) | הסתיים: 13,516 ‏ok, ‏689 ‏empty, ‏0 ‏error, ‏62,000,753 footprints, ‏`--verify`: ‏all triplets OK |
| 18.11.2026 | מדגם: כל גרנולה עשרים (symlinks בלבד) | `/archive/GUY/subsets/every20/` | 676 גרנולות, כ־3 מיליון footprints |
| 18.11.2026 | benchmark ראשון על נתונים אמיתיים: רק השיטות בלי תמונות | פלט מסוף: `/archive/GUY/results_run.out`; דוח מלא: `/archive/GUY/results/run_<זמן>/results.txt`; חלוקה: `/archive/GUY/subsets/every20/split_day5.json` | הסתיים ב־18.11. ‏`results.txt` ו־`summary.json` מועתקים לריפו: [results/2008_Pacific_every20/](docs/results/2008_Pacific_every20/) |
| 19–20.11.2026 | תמונות ל־6 חודשים (ינואר, מרץ, מאי, יולי, ספטמבר, נובמבר): `build_dataset.py` בלי `--no-images`, ‏`--workers 12`, חודש אחרי חודש, עם בלם שעוצר אם נשארים פחות מ־800 GB | `/archive/GUY/dataset/Pacific/2008/<חודש>/images/` (כ־3.1 TB, כ־470 MB לגרנולה); לוג: `/archive/GUY/images_run.out` | הסתיים: כ־6,560 גרנולות עם תמונות, ‏0 ‏error. נשארו פנויים 1.7 TB ב־`/archive` |
| 20.11.2026 | מדגם עם תמונות: גרנולה אחת מכל אחד מ־6 החודשים (`ls images \| awk 'NR%1093==1'`) | `/archive/GUY/subsets/img6/` | 6 גרנולות, כ־25 אלף footprints |
| 20.11.2026 | benchmark של כל השיטות על `img6`, ניסיון 1 (בלי `--cache-dir`) | — | נהרג בלי שגיאה (כנראה זיכרון: 17 GB לתהליך, 17 GB פנויים) |
| 20.11.2026 | ניסיון 2, עם `--cache-dir ~/ctz_cache` (קוד `2d21864`) | — | נהרג באמצע fold 2. ה־fold הראשון לקח כ־2.5 שעות. הסיבה: M4, ‏M10, ‏M6f ו־M6c בנו את כל פיקסלי ה־fold בזיכרון. תוקן בקומיטים `cf509c3`, ‏`d0a0b42` ו־`b92a051` |
| 21.11.2026 | ניסיון 3, אותה פקודה עם הקוד המתוקן (`b92a051`): כל השיטות, תמונות, מטמון על הדיסק | לוג עם שורת התקדמות לכל fold ושיטה: `/archive/GUY/results_img_run.out`; דוח: `/archive/GUY/results_img/run_<זמן>/results.txt`; חלוקה: `/archive/GUY/subsets/img6/split_day5.json`; מטמון זמני: `~/ctz_cache/fold_*` | נהרג ב־22.11 ב־02:09, בתחילת M6f ב־fold 1. הסיבה: המטמון מילא את הדיסק הראשי של השרת (583 GB ב־`~/ctz_cache`, ‏`/` ב־100%), כי הקבצים של כל אימון נשארו עד סוף ה־fold. תוקן: כל אימון מוחק את הקבצים שלו, ויש בדיקת מקום (`io.disk_array`). זמנים שנמדדו ב־fold 1: ‏M4 ‏471 דקות, ‏M10 ו־M10-cmp כ־25, ‏M5, ‏M6a ו־M6c כ־27, ‏M6d ‏16, ‏M3 ‏13, ‏M6b ‏8.5, וכל השאר פחות מדקה |
| 22–23.11.2026 | ניסיון 4: אותה הרצה עם הקוד המתוקן (`712f176`), בלי M4, שעולה כ־8 שעות ל־fold | `/archive/GUY/results_img/run_20261123_023239/` | הסתיים: כ־19 שעות, ‏0 שגיאות, המטמון לא עבר 8 GB. ‏`results.txt`, ‏`summary.json` וניתוח: [docs/results/2008_Pacific_img6/](docs/results/2008_Pacific_img6/) |

הפקודה של ה־benchmark, מתוך `~/GUY/CTZ_Research/master`:

```bash
nohup /home/ilai/miniconda3/envs/ctz/bin/python -m flux1km.cli --data-root /archive/GUY/subsets/every20 \
    --models M0-ridge M0-allsky37 M0-advisor M1-gbm M1-ann --split day --k 5 \
    --manifest /archive/GUY/subsets/every20/split_day5.json --psf resources/ceres_psf.csv \
    --out /archive/GUY/results > /archive/GUY/results_run.out 2>&1 & disown
```

- **השיטות:** רק אלה שחוזות את ה־footprint מהמומנטים, כי אין תמונות. שיטות המפה (B-clim, ‏M2–M10) מדולגות כשאין `images/`.
- **מדגם ולא כל השנה:** `io.load_dataset` טוען לזיכרון את כל התיקייה. 676 גרנולות תופסות בערך 9 GB של RAM, והשנה כולה לא נכנסת בזיכרון.
- **`--split day --k 5`:** ימים שלמים בכל fold, ועוד מבחן ה־interleave. החלוקה נשמרת ב־`split_day5.json`, כדי שהרצות הבאות יושוו על אותה חלוקה.
- **סביבות Python:** בניית הדאטה (`build_dataset.py`) רצה בסביבה `climate`, היחידה בשרת שיש בה `netCDF4` (‏1.7.4, עם HDF4). המודלים רצים ב־`ctz`, שאין בה `netCDF4`. ‏`ctz` (‏`conda create -n ctz -c conda-forge python=3.11 numpy scipy pandas scikit-learn`, ואז `pip install torch --index-url https://download.pytorch.org/whl/cu128`, כי PyPI איטי מהשרת). ‏torch 2.11.0+cu128 על ה־GPU של השרת (NVIDIA RTX PRO 4000 Blackwell, ‏24 GB). בסביבה `climate` חסרים sklearn ו־torch. ריצת `--demo` עם אותן 5 שיטות עברה בשרת בלי שגיאות.
- **תוצאות (macro RMSE של Fsw, ‏W/m²), הדוח המלא ב־[results/2008_Pacific_every20/results.txt](docs/results/2008_Pacific_every20/results.txt):** ‏M1-ann ‏10.86 (המנצחת), ‏M0-allsky37 ‏13.11, ‏M1-gbm ‏13.13, ‏M0-ridge ‏33.14 (נכשלת במבחני המגמה מול SZA ו־glint). לפי ההשוואות הזוגיות על 646 קבוצות גרנולות, M1-ann טובה מכל השאר באופן מובהק (p < 0.01 בכל המבחנים). השורה "not significant" ביומן ההחלטה אומרת רק שהמתחרה לא *שיפרה* מובהקת על M1-ann. לפי סוג עננות, RMSE של M1-ann: בהיר 5.99, חלקי 12.56, מלא 14.03 (M0-allsky37: ‏8.38, ‏14.74, ‏16.22). ‏**M0-advisor התבדרה** (RMSE בסדר גודל של 10^38), רק ב־footprints הבהירים: בחלקי ובמלא היא נותנת 24.33 ו־31.12. הרשת של המנחה חוזה ב־log1p ומחזירה עם expm1, כך שחיזוי חריג אחד בשמיים בהירים מתפוצץ. צריך לבדוק לפני שמשווים אליה.
- **מעקב:** `pgrep -af flux1km.cli` (יש שורה = רץ). בסוף, `tail -15 /archive/GUY/results_run.out` מציג את הטבלה ואת הנתיב של `results.txt`.

### ההרצה עם התמונות (`img6`)

```bash
nohup /home/ilai/miniconda3/envs/ctz/bin/python -m flux1km.cli --data-root /archive/GUY/subsets/img6 \
    --split day --k 5 --manifest /archive/GUY/subsets/img6/split_day5.json --psf resources/ceres_psf.csv \
    --cache-dir ~/ctz_cache --out /archive/GUY/results_img > /archive/GUY/results_img_run.out 2>&1 & disown
```

- **השיטות:** כולן (אין `--models`). שיטות המפה דורשות תמונות, ולכן רצות רק כאן.
- **`--cache-dir ~/ctz_cache`:** התמונות והקלט של הרשתות נשמרים כקבצים זמניים בדיסק ה־NVMe המקומי, כי `/archive` הוא דיסק איטי יותר. כך הזיכרון לא גדל עם כמות הנתונים. הלוגיקה וההגבלות מוסברות בסעיף 4 ("נתונים גדולים מהזיכרון"), ב־`flux1km/README.md` וב־[NEW_SERVER.md](NEW_SERVER.md).
- **מעקב:** `pgrep -af flux1km.cli; tail -6 /archive/GUY/results_img_run.out`. כל שורה בלוג מראה fold, שיטה, משך וזיכרון. תהליך שנעלם בלי `Traceback`, ותיקיית `fold_*` שנשארת ב־`~/ctz_cache`, אומרים שהתהליך נהרג מבחוץ. אז מוחקים את התיקייה ובודקים בלוג באיזו שיטה זה קרה.
- **דיסק:** בשיא, ה־fold תופס את התמונות שלו, כ־180 KB לכל footprint, ועוד את הקלט של האימון שרץ באותו רגע. ב־6 גרנולות זה כ־10 עד 30 GB. הקוד שומר שיישארו לפחות 20 GB פנויים, וזה ניתן לשינוי עם `CTZ_CACHE_MIN_FREE_GB`. כש־`df -h ~` מראה שהמקום יורד בלי הפסקה, משהו לא נמחק, ויש לעצור את ההרצה.
- **שינוי בתוצאות:** מהקומיט `2d21864`, רשתות המפה מקבלות seed לפני שהן נבנות. לפני כן M3 ו־M5 לא היו ניתנות לשחזור. תוצאות של רשתות מפה מלפני הקומיט הזה לא ניתנות להשוואה.
- **כמה נתונים:** 6 גרנולות הן בדיקה שהכול עובד וכיוון גס, לא בסיס להחלטה. יש רק 6 סצנות, כל fold נבחן על יום או שניים, והמבחנים הסטטיסטיים מבוססים על 6 יחידות בלבד. ההרצה הבאה מתוכננת על 50 עד 100 גרנולות.

---

## 3. השיטות: קובץ אחד לכל משפחה, וכל אחד רץ לבד

```bash
python3 -m flux1km.methods.<שם הקובץ> --demo                         # על דאטה סינתטי
python3 -m flux1km.methods.<שם הקובץ> --data-root DIR --manifest M.json   # על הדאטה האמיתי
```

כל הרצה של שיטה בודדת כוללת אוטומטית את מודלי הייחוס B-clim, ‏B-const ו־M0-ridge, כדי שתמיד יהיה למה להשוות. `--no-baselines` מבטל את זה.

הקבצים נמצאים ב־`master/flux1km/methods/`.

| קובץ | מודלים | מה זה | מפה ב־1 ק"מ? |
|---|---|---|---|
| `m0_legacy.py` | M0-allsky37, M0-advisor | ה־ANN של All_Sky (37 פיצ'רים) והמודל הנוכחי של אשכול, בפרוטוקול ההוגן: חלוקה לפי קבוצות, early stopping על validation, ובלי חיתוך `Fsw>400` | לא |
| `m1_guy.py` | M1-gbm, M1-ann | מודל גיא המשופר: ממוצע וסטיית תקן לכל ערוץ, יחסי ערוצים מול 0.47/4.05/11, טריגונומטריה של הזוויות, glint ושחזור broadband | לא |
| (`models.py`) | M2 | פולינום לכל פיקסל שמאומן במדויק מה־moments | כן |
| `m3_pixel_mlp.py` | M3 | MLP לכל פיקסל, ‏aggregation loss | כן |
| `m4_context_gbm.py` | M4 | עצים עם פיצ'רי הקשר מרחבי (חלונות, מרחק לענן) והגרדיאנט המצרפי | כן |
| `m5_unet.py` | M5 | UNet עם aggregation loss והחלקה ששומרת על קצוות | כן |
| `m6a_cnn_tabular.py` | M6a | ‏CNN יחד עם ענף טבלאי של ה־moments | כן |
| `m6b_mlp_context.py` | M6b | ‏MLP לכל פיקסל עם encoder הקשר שמסביר רק את השארית | כן |
| `m6c_unet_from_m2.py` | M6c | ‏UNet שמאומן מראש על המפה של M2 | כן |
| `m6d_distill_m1.py` | M6d | distillation מ־M1 על אריחים של 9×11 | כן |
| `m6e_footprint_norm.py` | M6e | נרמול המפה לתווית של כל footprint (**קורא את תוויות ה־test**) | כן |
| `m6f_pseudo_labels.py` | M6f | ‏M1 נותן pseudo-labels לחלונות ולגרנולות בלי תווית | כן |
| `m7_atprk.py` | M7 | מפת מגמה ועליה kriging של השאריות (**קורא את תוויות ה־test**) | כן |
| `m8_nb2bb.py` | M8 | ‏narrowband ל־broadband ‏(`Isw` כתווית בלבד) ואז ADM אמפירי | כן |
| `m9_homogeneous.py` | M9, M9-local | מודל footprint שאומן על footprints הומוגניים ומוחל על כל פיקסל (Gao). ‏M9-local **קורא את תוויות ה־test** | כן |
| `m10_gp.py` | M10, M10-cmp | ‏GP מצרפי (VBAgg) וגרסה מתווכת, עם אי־ודאות לכל פיקסל | כן |
| (`models.py`) | B-clim, B-const, B-bagpixel, M0-ridge | מודלי בסיס: אקלים; ערך ה־footprint (**קורא את התוויות**); ‏`f(mean x)` לכל פיקסל; ‏Ridge | — |

**"קורא את תוויות ה־test"**: השיטה משתמשת בערך CERES של ה־footprint הנבדק עצמו. זה חלק מההגדרה שלה (downscaling), ולכן היא לא מדורגת אף פעם. היא מוצגת רק מול האמת הסינתטית וב־interleave, ובדוח היא מסומנת `ranked = no`.

---

## 4. אפשרויות שעובדות בכל פקודה

| אפשרות | מה היא עושה |
|---|---|
| `--data-root DIR` | תיקייה עם `moments/`, ‏`scalars/` ו־`images/` |
| `--manifest M.json` | חלוקה קפואה. אם הקובץ לא קיים, הוא נוצר ונשמר |
| `--split logo/day/month/year/block` | סוג החלוקה, כשאין manifest |
| `--psf FILE` | ה־PSF: ‏`ceres_psf.csv` או ה־`.npz` שנוצר ב־E0 |
| `--operator` | אופרטור הצבירה הראשי. ברירת המחדל היא `ceres_psf` כשיש `--psf`, ואחרת `fov_ellipse` |
| `--input-set F1..F5` | סט הקלטים לאבלציה (סעיף 3 ב־PLAN) |
| `--sampling S0..S7` | אסטרטגיית הדגימה של סט האימון |
| `--budget N` | מספר הקונפיגורציות המקסימלי לכוונון, זהה לכל השיטות |
| `--seeds 0 1 2 3 4` | כמה seeds להריץ |
| `--models ...` | אילו מודלים להריץ (ב־`flux1km.cli` ברירת המחדל היא כולם) |
| `--demo`, `--demo-hard`, `--ctz-visible 0.3`, `--no-ctz`, `--demo-size 6x20` | הדאטה הסינתטי |
| `--out DIR` | לאן לכתוב. ברירת המחדל היא `benchmark_outputs/flux1km` |
| `--cache-dir DIR` | ‏(`flux1km.cli`) עבודה עם נתונים שגדולים מהזיכרון: התמונות נשארות על הדיסק, וכל fold נשמר כקבצים זמניים ב־`DIR`, שנמחקים בסוף ה־fold. התוצאות זהות. ההסבר המלא נמצא בסעיף הבא |

### נתונים גדולים מהזיכרון: `--cache-dir`

**הבעיה:** בלי `--cache-dir`, הקוד מחזיק בזיכרון את התמונות של כל הגרנולות, ועוד עותק שלהן לכל אחת מקבוצות האימון, הוולידציה והבדיקה. רשתות המפה מוסיפות עותק של הקלט שלהן. בסך הכול כ־0.6 עד 0.7 MB לכל footprint, ולכן 10 עד 15 גרנולות כבר ממלאות שרת עם עשרות GB של זיכרון.

**הפתרון:** עם `--cache-dir DIR`:
1. התמונות של כל גרנולה נקראות מהדיסק רק כשבונים ממנה קבוצה של fold (`io.LazyNpz`).
2. התמונות של כל קבוצה נכתבות לקבצי `.npy` בתיקייה זמנית `DIR/fold_*/batch_*`, ונקראות משם כ־memory map (`features.make_batch(..., store=...)`).
3. גם הקלט המנורמל של רשתות המפה (X, בגודל N×F×27×55) נכתב לאותה תיקייה. כל מנת אימון קוראת מהדיסק רק את השורות שלה (`_torch.x_rows`).
4. החישוב של מאפייני הפיקסלים ושל החיזוי נעשה בגושים של 500 footprints (`models.PREDICT_CHUNK`).
5. התיקייה של כל fold נמחקת בסוף ה־fold.

**מה זה נותן:** בבדיקה על נתונים סינתטיים (M5 ו־M2), הזיכרון האמיתי נשאר כ־1.3 GB גם כשמספר ה־footprints מוכפל. בלי המטמון הוא גדל ב־0.7 MB לכל footprint. הגבול עובר מהזיכרון לדיסק ולזמן הריצה.

**דרישות:**
- **מקום:** בערך 400 KB לכל footprint של ה־fold הגדול ביותר: תמונות, הקלט X ושאר הקבצים.
- **סוג הדיסק:** מומלץ דיסק מקומי מהיר (SSD/NVMe), כי מנות האימון קוראות שורות בסדר אקראי. דיסק רשת איטי יאט את האימון מאוד.

**התוצאות זהות:** הבדיקות ב־`tests/test_benchmark.py::TestDiskCache`, ‏`tests/test_models.py::TestDiskBackedTraining` ו־`tests/test_features.py::TestMakeBatchStore` משוות ריצה עם מטמון לריצה בלי מטמון, ובודקות שהתוצאות זהות.

**הגבלות:**
- **הבדיקות הפיזיקליות של המפות** מחושבות על מדגם אקראי של עד 5,000 footprints לכל שיטה ו־fold (`BenchmarkConfig.physics_max_footprints`). בהרצות קטנות נלקחים כולם.
- **מדידה לכל שיטה:** על נתונים סינתטיים עם מטמון, כשמספר ה־footprints מוכפל, הזיכרון האמיתי של כל שיטות המפה לא גדל. הוא נע בין 0.4 GB ל־3.4 GB, והגבוה ביותר הוא הגוש של M10.
- **רשומות התחזיות** (`predictions.csv`) נשמרות בזיכרון עד סוף ההרצה, כ־1 KB לכל footprint לכל שיטה. זה הגבול הבא אם מגדילים עוד.

---

## 5. דאטה סינתטי

- `--demo`: הסצנה הקלה. הרדיאנסים הם בערך האלבדו כפול ספקטרום השמש, וכל ה־CTZ נראה בערוצים.
- `--demo-hard`: הסצנה הקשה (סעיף 4.3 ב־PLAN):
  - ההמרה מ־narrowband ל־broadband אי־לינארית ותלויה ב־SZA;
  - לרדיאנסים יש אניזוטרופיה תלוית זווית: אונת glint, ‏limb, ‏forward/backscatter;
  - רק חלק מה־CTZ נראה בערוצים (`--ctz-visible`, ברירת מחדל 0.5). את השאר אפשר לשחזר רק מהקשר מרחבי. האמת עצמה זהה לכל ערך של `--ctz-visible`.
- `--no-ctz`: סצנת ביקורת בלי CTZ, שבה עקומת השטף מול המרחק לענן צריכה לצאת שטוחה.

---

## 6. הפלט: קובץ טקסט חדש בכל הרצה

כל הרצה יוצרת תיקייה חדשה `benchmark_outputs/flux1km/run_<זמן UTC>/`, והפקודה מדפיסה בסוף את הנתיב לקובץ:

| קובץ | מה יש בו |
|---|---|
| **`results.txt`** | **כל התוצאות, לקריאה בלי קוד** |
| `summary.json` | אותן תוצאות, בפורמט לתוכנה |
| `predictions.csv` | שורה לכל מודל, ‏fold, ‏seed ו־footprint: תווית, תחזית לפי כל אופרטור, ‏regime וזוויות |

**הסעיפים ב־`results.txt`, לפי הסדר:**
1. **WINNER**: המנצח, הדירוג, השוויונות ויומן ההחלטה. היומן מסביר למה כל שיטה החליפה או לא החליפה את הקודמת.
2. **Configuration**: מודלים, אופרטורים, ‏PSF, תקציב הכוונון, ‏seeds, דגימה, רצפות רעש ושערים.
3. **Folds**: גודל כל fold וההיפר־פרמטרים שנבחרו לכל מודל.
4. **Errors**: מודל שנכשל ב־fold נרשם כאן, וההרצה ממשיכה.
5. **Scores**: ‏macro RMSE (הציון הראשי), ‏RMSE, ‏MAE, ‏bias, ‏R², ציון לפי כל אופרטור רגישות, ציון interleave, וציוני פיקסל מול האמת הסינתטית (RMSE, קורלציה, ‏SSIM).
6. **Scores by cloud regime**: ‏clear, ‏broken ו־overcast.
7. **Residual flatness**: מגמת השאריות מול SZA, ‏VZA, ‏RAA ו־glint, ואסימטריית צדי ה־swath.
8. **Physics checks**: שטף השמיים הבהירים הרחוקים (היעד: 80±10), חריגה מעל S0·μ0, שיא בספקטרום במרווח בין ה־footprints, ועקומת השטף מול המרחק לענן.
9. **Pairwise comparisons**: ‏bootstrap עם Holm, ‏Wilcoxon ו־Diebold–Mariano.
10. **Gate results** ו־**Glossary**: מילון לכל מספר בדוח.

---

## 7. שאר הכלים

| קובץ | מה הוא עושה | פקודה |
|---|---|---|
| `e0.py` | שלב E0: אינוונטר, טווחים ויחידות, בדיקת `FOVorientation`, ‏moments מול patches, התאמת H, חפיפה בין footprints, ערוצים פגומים, ‏manifest עם בדיקת זליגה, ודוח הבאגים לאשכול | `python3 -m flux1km.e0 --data-root DIR` |
| `synthetic_study.py` | ניסוי האמת הסינתטית: כמה מספרי footprints, כמה גדלי צבירה, וסצנת ביקורת בלי CTZ | `python3 -m flux1km.synthetic_study --models ...` |
| `sampling_study.py` | אסטרטגיות S0–S7: ‏Jaccard, ‏KS, ‏MMD, מטריצת אימון לפי Si מול בדיקה לפי תת־אוכלוסייה, ו־Kendall τ | `python3 -m flux1km.sampling_study --models ...` |
| `decision.py` | כלל ההחלטה של סעיף 6. מופעל אוטומטית בכל הרצה | — |
| `cli.py` | הרצה של כל השיטות או של חלקן | `python3 -m flux1km.cli` |
| `mapviz.py` | איורים של מפות 1 ק"מ לכל שיטת מפה: הסצנה, כרטיס footprint לכל משטר, ממוצע משוקלל מול CERES, עקומת CTZ, וה־masks של העננים. פירוט בהמשך הסעיף | `python3 -m flux1km.mapviz --demo` |
| `scripts/download_modis.py` | הורדת גרנולות Aqua מ־LAADS DAAC ‏(ברירת מחדל `MYD021KM` ו־`MYD03`; אפשר להוסיף `MYD06_L2` ו־`MYD35_L2`), רק סטים מלאים ורק בשעות היום. תאריכים, יעד, שם אזור ותיבה נקבעים בארגומנטים (`--start --end --out --region --box --products`), בלי לערוך את הקובץ | `python3 scripts/download_modis.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --dry-run` |

הסקריפט עצמאי, ולכן אפשר להעתיק אותו לכל שרת. הוא צריך רק את `earthaccess` ומשתמש `Earthdata Login` שמוגדר ב־`~/.netrc`. ברירת המחדל היא כל 2008, האוקיינוס השקט בין 60°S ל־60°N, והפלט נשמר ב־`/archive/GUY`.

מבנה התיקיות זהה למה ש־`Generate_AllSky_2.py` של המנחה קורא, כך שאפשר להפעיל אותו ישר על הפלט:

```
/archive/GUY/AQUA/<year>/Pacific/<month>/MOD02/   MYD021KM
                                         /MOD03/   MYD03
                                         /MOD06/   MYD06_L2 (עם --products ... MYD06_L2)
                                         /MOD35/   MYD35_L2 (עם --products ... MYD35_L2)
```

שמות תיקיות החודשים הם כמו ברשימת `MONTH` של המנחה (`mrch`, ‏`june`, ‏`sep`, ‏`dec` וכו'). את קובץ ה־CERES SSF הסקריפט לא מוריד: צריך לשים אותו בכל תיקיית חודש, והשם שלו חייב לכלול `CERES`. קבצים שירדו במבנה הישן (`<product>/<year>/<doy>/`) מעבירים עם `--reorganize`. הרצה חוזרת מדלגת על קבצים שכבר ירדו.

מדריך הרצה קצר להורדה, כולל אשכול Slurm ‏(`scripts/slurm/download_year.sbatch`): [scripts/HOW_TO_DOWNLOAD.md](scripts/HOW_TO_DOWNLOAD.md). מורידים את ארבעת המוצרים: `--products MYD021KM MYD03 MYD35_L2 MYD06_L2`.

בניית הדאטה לאימון (`scalars/`, `moments/`, `images/`) מהקבצים שהורדו, עם הגרסה המתוקנת של הממיר של המנחה: `python3 scripts/build_dataset.py --data /archive/GUY --out /archive/GUY/dataset` (ה־PSF של המנחה הוא ברירת המחדל). מדריך ורשימת התיקונים: [scripts/HOW_TO_BUILD_DATASET.md](scripts/HOW_TO_BUILD_DATASET.md).

**CERES SSF (אוטומטי).** `scripts/download_ceres.py` מוריד את ה־SSF של Aqua FM3, ‏Edition4A (‏`CER_SSF_Aqua-FM3-MODIS`) מ־ASDC דרך `earthaccess`, באותו משתמש Earthdata כמו ב־MODIS. ב־2008 יש 8,752 קבצים שעתיים גלובליים של כ־60 MB. הסקריפט מוריד רק את השעות שיש להן גרנולות MODIS בתיקיות `MOD02`, חותך כל קובץ לתיבת האוקיינוס השקט ול־12 המשתנים שהמנחה קורא (בשמות של ה־Order Tool), מוחק את הקובץ הגולמי וממזג לקובץ אחד לחודש (`CERES_SSF_Aqua_<year>_<month>.nc`). הוא דורש את `organize_ceres.py` ואת `download_modis.py` באותה תיקייה. שני הסקריפטים מקבלים את אותם `--start --end --out --region --box` (ברירת מחדל: האוקיינוס השקט, 2008), אז מריצים אותם עם אותם ערכים; `--box WEST EAST SOUTH NORTH` במעלות, ו־`WEST > EAST` חוצה את קו התאריך. אפשר טווח של כמה שנים, ו־CERES יורד רק לשעות שיש להן MODIS ב־`--out` וב־`--region` שהוזנו.

```bash
python3 scripts/download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --dry-run   # כמה קבצים וכמה GB
python3 scripts/download_ceres.py --inspect <raw>    # בדיקה שקובץ גולמי אחד מכיל את 17 המשתנים
nohup python3 scripts/download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY \
    > /archive/GUY/ceres_run.out 2>&1 &
python3 scripts/download_ceres.py --start 2008-01-01 --end 2008-12-31 --out /archive/GUY --merge-only   # מיזוג ובדיקה בלבד
```

**CERES SSF (ידני, גיבוי).** אם ההורדה האוטומטית לא עובדת: מזמינים ב־CERES Order Tool (https://ceres.larc.nasa.gov/data/) חודש־חודש, בפורמט netCDF, עם אותה תיבת אזור והשדות Time/Location/Angles, CERES Observed TOA Fluxes and Radiances, Clear Footprint Area ו־Effective Optical Depth Ocean. את `fileURLs.txt` שמגיע במייל מעבירים ל־`scripts/organize_ceres.py`, שמוריד, ממיין, ממזג ובודק:

```bash
python3 scripts/organize_ceres.py --urls fileURLs.txt --out /archive/GUY
python3 scripts/organize_ceres.py --src <תיקייה עם קבצים שהורדו ידנית> --out /archive/GUY
python3 scripts/organize_ceres.py --verify-only --out /archive/GUY
```

---

### איורי המפות (`mapviz.py`)

הכלי מאמן כל שיטת מפה **פעם אחת** על fold אחד, עם הפרמטרים שה־benchmark כבר בחר ל־fold הזה (`--params-from summary.json`, בלי כוונון מחדש). אחר כך הוא חוזה מפות בחלון של 150 על 500 ק"מ בכמה גרנולות מסט הבדיקה. החלון נבחר כך שיהיו בו footprints בהירים, שבורים ומעוננים. המפות נשמרות (`maps_<granule>.npz`), ומהן נוצרים חמישה איורים לכל גרנולה:

| קובץ | מה רואים |
|---|---|
| `fig1_scene_*.png` | תמונת 0.66 µm עם גבולות ה־mask של MYD35 (רגיל ומחמיר), מה ש־CERES רואה (ערך ה־footprint צבוע על הפיקסלים שלו), ומפת 1 ק"מ של כל שיטה בסקאלת צבע אחת, עם ה־RMSE מול CERES |
| `fig2_footprint_<משטר>_*.png` | footprint אחד בהיר, שבור ומעונן: העננים בו, ה־CF לפי כל mask (רגיל ומשוקלל PSF), ה־PSF, הערך של CERES ו־100−Clrf, המפה של כל שיטה, וה־Σw·f שלה מול CERES עם רצועת ±10 |
| `fig3_scatter_*.png` | כל ה־footprints בחלון: Σw·f של כל שיטה מול CERES, צבוע לפי כיסוי העננים |
| `fig4_ctz_*.png` | השטף הממוצע של פיקסלים בהירים מול המרחק מענן, לכל שיטה, לפי ה־mask הרגיל והמחמיר. עקומה יורדת יכולה להיות CTZ, אבל גם תוצאה של החלקה |
| `fig5_masks_*.png` | שלושת ה־masks (MYD35 רגיל, MYD35 מחמיר, פרוקסי 0.66 µm), וה־CF של כל אחד מול האחוז המעונן של CERES ‏(100−Clrf) |

**מה הכלי צריך מהדאטה:** בקבצי ה־images צריכים להיות `row0` ו־`col0` (מיקום הטלאי בגרנולה) ו־`ImCloud` (ה־mask של MYD35 לכל פיקסל). `build_dataset.py` שומר אותם מ־25.11.2026. הגרנולות של `img6` נבנו לפני כן, ולכן בונים מחדש רק אותן לתיקייה נפרדת. כך לא נוגעים בקבצים שהרצות אחרות קוראות. גרנולה אחת לוקחת כחצי דקה, ו־worker אחד צריך כ־1.5 GB:

```bash
cd ~/GUY/CTZ_Research/master && git pull
nice -n 19 /home/ilai/miniconda3/envs/climate/bin/python scripts/build_dataset.py --data /archive/GUY \
    --out /archive/GUY/dataset_viz --workers 1 --granules $(ls /archive/GUY/subsets/img6/images | sed 's/\.npz$//')
```

ואז האיורים, עם החלוקה והפרמטרים של ריצת `img6`. ‏`pip install matplotlib` בסביבה `ctz` אם הוא חסר:

```bash
nohup nice -n 19 /home/ilai/miniconda3/envs/ctz/bin/python -m flux1km.mapviz --data-root /archive/GUY/dataset_viz/Pacific \
    --manifest /archive/GUY/subsets/img6/split_day5.json --fold day-kfold:0 --psf resources/ceres_psf.csv \
    --params-from docs/results/2008_Pacific_img6/summary.json --n-granules 2 \
    --models B-const B-bagpixel M2 M3 M5 M6a M6b M6c M6d M6f M8 M9 M10 \
    --cache-dir ~/ctz_cache --out /archive/GUY/maps > /archive/GUY/maps_run.out 2>&1 & disown
```

- **זמן:** כמו fold אחד של ה־benchmark בלי הכוונון, כלומר אימון אחד לכל שיטה: כ־2 שעות לכל הרשימה. ‏M4 לא ברשימה, כי היא לבדה לוקחת כ־8 שעות.
- **זיכרון:** כמו ה־benchmark עם `--cache-dir`, כמה GB בלבד, ו־`nice` נותן עדיפות להרצות אחרות.
- **החלוקה:** אם `--manifest` נכשל עם "manifest was built on a different footprint table", מספר ה־footprints שנבנו מחדש לא זהה למקור. במקרה כזה מריצים בלי `--manifest` ובלי `--params-from`, והכלי בונה חלוקה לפי ימים ובוחר את הפרמטר הראשון ברשת של כל שיטה.
- **ציור מחדש בלי אימון:** `python3 -m flux1km.mapviz --replot /archive/GUY/maps/run_<זמן>`.
- **לריפו:** את ה־PNG מעתיקים ל־`docs/results/2008_Pacific_img6/maps/`. קבצי ה־`maps_*.npz` גדולים ונשארים בשרת.

## 8. כללים שהקוד אוכף

- **אין קלטים מ־CERES.** טסט משבש את `Fsw`, ‏`Flw`, ‏`Isw`, ‏`Ilw` ו־`Clrf` של ה־test ובודק שאף תחזית לא משתנה, לכל אחד מהמודלים.
- **אין חלוקה אקראית לפי footprint.** כל fold עובר בדיקת שימוש חוזר בפיקסלים לפני ההרצה, ו־fold שדולף עוצר את ההרצה עם שגיאה.
- **אותו אופרטור לאימון ולציון.** שיטות שמתאמנות עם aggregation loss מקבלות את האופרטור שלפיו הן נבחנות.
- **מפות חסומות פיזיקלית.** רשתות מנבאות אלבדו דרך sigmoid, כך שהשטף לעולם לא עובר את S0·μ0.

---

## 9. טסטים

```bash
pytest -q flux1km tests     # הכל, כולל doctests. עם PyTorch זה לוקח כ־25 דקות על CPU
ruff check .
```

הטסטים של השיטות נמצאים ב־`tests/methods/test_<קובץ>.py`. ב־CI אין PyTorch, ולכן טסטי הרשתות מדולגים שם.
