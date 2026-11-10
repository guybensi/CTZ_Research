# מדריך הרצה: מה כל דבר ואיך מריצים אותו

המדריך הזה מיועד למי שרוצה להריץ: מה יש בכל קובץ, איזו פקודה מריצה אותו, ומה יוצא ממנו.

- ההסבר המדעי המלא של כל שיטה נמצא ב־[MODELS.md](MODELS.md).
- התוכנית נמצאת ב־[PLAN.md](PLAN.md), וסטטוס המימוש שלה בסעיף 10 שם.
- כל הפקודות כאן רצות מתוך התיקייה `master/`.

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

עד שהקובץ `ceres_psf.csv` מגיע מאשכול, מוסיפים לכל פקודה `--psf benchmark_outputs/e0/fitted_operator.npz`. זה ה־PSF ששוחזר בשלב 1. כשהקובץ האמיתי מגיע מחליפים ל־`--psf ceres_psf.csv`.

כל פקודה עובדת גם עם `--demo` במקום `--data-root`. אז היא רצה על דאטה סינתטי עם אמת ידועה ב־1 ק"מ, ו־`--demo-hard` מריץ את הגרסה הקשה (ראו סעיף 5).

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
| `scripts/download_modis.py` | הורדת גרנולות Aqua מ־LAADS DAAC ‏(ברירת מחדל `MYD021KM` ו־`MYD03`; אפשר להוסיף `MYD06_L2` ו־`MYD35_L2`), רק סטים מלאים ורק בשעות היום. התאריכים והפוליגונים מוגדרים במשתנים הגלובליים בראש הקובץ | `python3 scripts/download_modis.py --dry-run` |

הסקריפט עצמאי, ולכן אפשר להעתיק אותו לכל שרת. הוא צריך רק את `earthaccess` ומשתמש `Earthdata Login` שמוגדר ב־`~/.netrc`. ברירת המחדל היא כל 2008, האוקיינוס השקט בין 60°S ל־60°N, והפלט נשמר ב־`/archive/GUY`.

מבנה התיקיות זהה למה ש־`Generate_AllSky_2.py` של המנחה קורא, כך שאפשר להפעיל אותו ישר על הפלט:

```
/archive/GUY/AQUA/<year>/Pacific/<month>/MOD02/   MYD021KM
                                         /MOD03/   MYD03
                                         /MOD06/   ריקה, אלא אם MYD06_L2 ב־PRODUCTS
                                         /MOD35/   ריקה, אלא אם MYD35_L2 ב־PRODUCTS
```

שמות תיקיות החודשים הם כמו ברשימת `MONTH` של המנחה (`mrch`, ‏`june`, ‏`sep`, ‏`dec` וכו'). את קובץ ה־CERES SSF הסקריפט לא מוריד: צריך לשים אותו בכל תיקיית חודש, והשם שלו חייב לכלול `CERES`. קבצים שירדו במבנה הישן (`<product>/<year>/<doy>/`) מעבירים עם `--reorganize`. הרצה חוזרת מדלגת על קבצים שכבר ירדו.

**CERES SSF.** קובץ ה־SSF מוזמן ב־CERES Order Tool באתר (https://ceres.larc.nasa.gov/data/ ← SSF Aqua FM3, Edition4A). אין לו ממשק תכנותי מתועד, ולכן ההזמנה נעשית ידנית: חודש־חודש, פורמט netCDF, אותו תיבת אזור כמו ב־MODIS, והשדות Time/Location/Angles, CERES Observed TOA Fluxes and Radiances, Clear Footprint Area ו־Effective Optical Depth Ocean. את `fileURLs.txt` שמגיע במייל ההזמנה מעבירים ל־`scripts/organize_ceres.py`, שמוריד, ממיין לתיקיות החודשים, ממזג לקובץ אחד לחודש (`CERES_SSF_Aqua_<year>_<month>.nc`) ובודק שכל 12 המשתנים של המנחה קיימים:

```bash
python3 scripts/organize_ceres.py --urls fileURLs.txt --out /archive/GUY
python3 scripts/organize_ceres.py --src <תיקייה עם קבצים שהורדו ידנית> --out /archive/GUY
python3 scripts/organize_ceres.py --verify-only --out /archive/GUY
```

---

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
