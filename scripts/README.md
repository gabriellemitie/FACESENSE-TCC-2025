FaceSense report generator

This folder contains a small CLI tool to generate a per-session report from the FaceSense CSV exports.

Files:
- report_generator.py — main CLI script. Reads `historico_postura.csv` and `historico_sessoes.csv` (defaults to `facesense_posture/`), computes metrics, creates PNG plots and writes `out/report/report.md`.
- requirements.txt — Python dependencies for the script.

Quick start (macOS / zsh):

1) Create and activate a venv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

2) Run the report generator (defaults assume CSVs are in `facesense_posture/`):

```bash
python3 scripts/report_generator.py --outdir out/report
```

3) Open the generated report:

```bash
open out/report/report.md  # macOS
```

Notes and assumptions:
- The script uses heuristics to find timestamp, stress-probability and posture/alert columns. If your CSV columns have non-standard names, pass the CSV through a small cleanup step (or edit the script to explicitly set `timestamp`, `stress_prob` and `posture` column names).
- The correction-rate calculation assumes timestamps are present and monotonic. It looks for posture "corrections" within a default 30-second window after an alert.
- The script is intentionally conservative: if a column can't be parsed it will warn and continue rather than fail.

If you want, I can adapt the script to your exact CSV columns (tell me the column names) and add a PDF/HTML output or integrate with your Flutter app backend.
