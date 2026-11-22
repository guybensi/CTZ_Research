# Results

One folder per real-data run: `<year>_<region>_<subset>[_<note>]/`, e.g. `2008_Pacific_every20/`.

Each folder holds:
- `results.txt` (and `summary.json` if small), copied from the run's `--out` folder on the server (`run_<time>/`);
- `ANALYSIS.md`, a short reading of the scores and the next steps (optional).

Files over 100 MB (`predictions.csv`, `split_*.json`) stay on the server; write their server path in the analysis.
After adding a folder, add a line to [../TIMELINE.md](../TIMELINE.md).
