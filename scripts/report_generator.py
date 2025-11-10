#!/usr/bin/env python3
"""
Report generator for FaceSense session/posture data.

Reads two CSVs (posture timeline and session-level file), computes metrics,
creates plots and a Markdown report with embedded PNGs.

Flexible: will try to infer column names for timestamps, stress probabilities
and posture/alert events.

Usage:
  python3 scripts/report_generator.py \ 
    --posture facesense_posture/historico_postura.csv \ 
    --sessions facesense_posture/historico_sessoes.csv \ 
    --outdir out/report

Produces: out/report/report.md and PNG files.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

sns.set(style="whitegrid")

# Heuristics for column detection
TIMESTAMP_KEYS = ["timestamp", "time", "ts", "datetime", "date", "hora"]
STRESS_KEYS = ["stress", "stress_prob", "stress_probability", "prob_stress", "stressScore", "stress_score", "prob"]
POSTURE_KEYS = ["posture", "label", "posture_label", "pose", "alert", "posture_alert", "event"]


def find_column(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        for col in df.columns:
            if col.lower() == c.lower() or c.lower() in col.lower():
                return col
    return None


def parse_time_column(df: pd.DataFrame):
    # Try to find a time-like column and parse it to datetime
    col = find_column(df, TIMESTAMP_KEYS)
    if col is None:
        # if index looks like time, try converting index
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            return df, df.index.name or "index"
        except Exception:
            raise ValueError("No timestamp-like column found in dataframe")
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    if df[col].isna().all():
        # try epoch numeric
        try:
            df[col] = pd.to_datetime(df[col].astype(float), unit='s', errors='coerce')
        except Exception:
            pass
    if df[col].isna().any():
        # leave as is but warn
        print(f"Warning: some timestamps could not be parsed in column '{col}'", file=sys.stderr)
    df = df.set_index(col)
    return df, col


def detect_stress_column(df: pd.DataFrame):
    col = find_column(df, STRESS_KEYS)
    if col is None:
        # fallback: any numeric column bounded in [0,1]
        for c in df.select_dtypes(include=[np.number]).columns:
            if df[c].dropna().between(0, 1).all():
                return c
        return None
    return col


def detect_posture_column(df: pd.DataFrame):
    col = find_column(df, POSTURE_KEYS)
    return col


def compute_stress_metrics(ts: pd.Series, threshold: float = 0.6):
    # ts: time-indexed pd.Series of stress probabilities [0,1]
    s = ts.dropna().astype(float)
    if s.empty:
        return {}
    mean = float(s.mean())
    median = float(s.median())
    mx = float(s.max())
    pct_above = float((s >= threshold).mean())
    # approximate area under curve (mean * duration)
    duration = (s.index[-1] - s.index[0]).total_seconds() if len(s) > 1 else 0
    auc = float(s.mean() * duration)
    return {
        "mean": mean,
        "median": median,
        "max": mx,
        "pct_above_threshold": pct_above,
        "duration_seconds": duration,
        "auc_approx": auc,
        "n_samples": int(len(s)),
    }


def compute_posture_metrics(df: pd.DataFrame, posture_col: str | None, alert_values: list[str] | None = None):
    # posture_col may be categorical or boolean. We'll compute distribution and events.
    res = {}
    if posture_col is None:
        return res
    col = df[posture_col]
    # distribution
    value_counts = col.value_counts(dropna=True)
    res['distribution'] = value_counts.to_dict()
    # alerts: boolean-like (True/False) or certain labels
    if pd.api.types.is_bool_dtype(col) or set(col.dropna().unique()) <= {0, 1, True, False}:
        alerts = col.astype(bool)
    else:
        # if alert_values provided, treat those as alerts, else try labels containing 'bad'/'alert'/'mau' etc.
        if alert_values is None:
            candidates = [str(v).lower() for v in col.dropna().unique()]
            alert_values = [v for v in candidates if any(k in v for k in ("bad", "alert", "poor", "mau", "slouch", "incorrect"))]
        if alert_values:
            alerts = col.str.lower().isin(alert_values)
        else:
            # fallback: treat any label that is not the most common as an alert
            common = value_counts.idxmax()
            alerts = col != common
    res['alert_count'] = int(alerts.sum())
    # average time between alerts
    if alerts.sum() >= 2 and df.index.is_monotonic_increasing:
        alert_times = df.index[alerts.values]
        deltas = np.diff(alert_times.astype('datetime64[ms]').astype(np.int64)) / 1000.0
        res['avg_seconds_between_alerts'] = float(deltas.mean())
    else:
        res['avg_seconds_between_alerts'] = None
    return res


def compute_correction_rate(df: pd.DataFrame, posture_col: str, alert_matcher=None, correction_labels: list[str]=None, window_seconds: int=30):
    # alert_matcher: function(series_value)->bool
    if alert_matcher is None:
        def alert_matcher(v):
            if pd.isna(v):
                return False
            if isinstance(v, (int, float, np.integer, np.floating)):
                return bool(v)
            return str(v).lower() not in ['good', 'ok', 'normal', 'correct', 'sentado']

    col = df[posture_col]
    alerts_idx = [i for i, v in enumerate(col) if alert_matcher(v)]
    if not alerts_idx:
        return None
    corrected = 0
    times = df.index
    for idx in alerts_idx:
        t0 = times[idx]
        tmax = t0 + pd.Timedelta(seconds=window_seconds)
        # search next rows until tmax
        subsequent = df.loc[t0:tmax]
        # if any row in subsequent has a label in correction_labels or looks like 'good' then count
        ok_found = False
        for v in subsequent[posture_col].dropna().values:
            sval = str(v).lower()
            if correction_labels and sval in [c.lower() for c in correction_labels]:
                ok_found = True
                break
            if any(k in sval for k in ('good','ok','normal','correct','upright','aligned','reta','corrigido')):
                ok_found = True
                break
        if ok_found:
            corrected += 1
    return corrected / len(alerts_idx)


def plot_stress_and_alerts(stress_ts: pd.Series | None, df_posture: pd.DataFrame | None, posture_col: str | None, outdir: Path, prefix: str = 'session'):
    imgs = {}
    if stress_ts is not None and not stress_ts.dropna().empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        stress_ts.plot(ax=ax, color='C0')
        ax.set_ylabel('Stress probability')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Stress probability timeline')
        fn = outdir / f"{prefix}_stress_timeline.png"
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
        imgs['stress_timeline'] = fn.name
    if df_posture is not None and posture_col is not None:
        fig, ax = plt.subplots(figsize=(10, 2))
        # show posture label as stacked color bar
        vals = df_posture[posture_col].fillna('NA')
        counts = vals.value_counts()
        counts.plot(kind='bar', ax=ax)
        ax.set_ylabel('Frames / Samples')
        ax.set_title('Posture label counts')
        fn = outdir / f"{prefix}_posture_counts.png"
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
        imgs['posture_counts'] = fn.name
        # overlay alerts on timeline if possible
        if isinstance(df_posture.index, pd.DatetimeIndex) and df_posture[posture_col].dtype == object:
            # mark alert times
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot([t for t in df_posture.index], [0] * len(df_posture), alpha=0)  # space
            alert_mask = None
            # try boolean
            if df_posture[posture_col].dtype == bool:
                alert_mask = df_posture[posture_col]
            else:
                candidates = [str(v).lower() for v in df_posture[posture_col].dropna().unique()]
                alert_labels = [v for v in candidates if any(k in v for k in ("bad","alert","slouch","incorrect","mau"))]
                if alert_labels:
                    alert_mask = df_posture[posture_col].str.lower().isin(alert_labels)
            if alert_mask is not None and alert_mask.any():
                ax.scatter(df_posture.index[alert_mask], [0.5]*alert_mask.sum(), marker='v', color='red', label='alerts')
                ax.set_ylim(-1, 1)
                ax.set_yticks([])
                ax.set_title('Posture alerts timeline')
                fn = outdir / f"{prefix}_posture_alerts_timeline.png"
                fig.tight_layout()
                fig.savefig(fn)
                plt.close(fig)
                imgs['posture_alerts_timeline'] = fn.name
    return imgs


def generate_markdown_report(outdir: Path, metrics: dict, imgs: dict, note: str | None = None):
    md = []
    md.append(f"# FaceSense Session Report\n")
    md.append(f"Generated: {pd.Timestamp.now()}\n")
    if note:
        md.append(f"**Note**: {note}\n")
    md.append("## Key metrics\n")
    if 'stress' in metrics:
        m = metrics['stress']
        md.append("### Stress metrics\n")
        md.append("| Metric | Value |\n|---:|---:|\n")
        md.append(f"| Mean stress | {m.get('mean'):.3f} |\n")
        md.append(f"| Median stress | {m.get('median'):.3f} |\n")
        md.append(f"| Max stress | {m.get('max'):.3f} |\n")
        md.append(f"| % samples >= threshold | {m.get('pct_above_threshold'):.2%} |\n")
        md.append(f"| Duration (s) | {m.get('duration_seconds')} |\n")
        md.append(f"| AUC (approx) | {m.get('auc_approx'):.3f} |\n")
    if 'posture' in metrics:
        p = metrics['posture']
        md.append("### Posture metrics\n")
        md.append("| Metric | Value |\n|---:|---:|\n")
        md.append(f"| Alert count | {p.get('alert_count')} |\n")
        avg = p.get('avg_seconds_between_alerts')
        md.append(f"| Avg seconds between alerts | {avg if avg is not None else 'N/A'} |\n")
        md.append("\nPosture distribution:\n")
        md.append("| Label | Count |\n|---|---:|\n")
        for k, v in (p.get('distribution') or {}).items():
            md.append(f"| {k} | {v} |\n")
    if 'correction_rate' in metrics:
        cr = metrics['correction_rate']
        md.append("## Correction / Response\n")
        md.append(f"Correction rate within window: {cr:.2%}\n")
    # Images
    if imgs:
        md.append("## Visualizations\n")
        for name, fname in imgs.items():
            md.append(f"### {name.replace('_',' ').title()}\n")
            md.append(f"![{name}]({fname})\n")
    md_text = "\n".join(md)
    out_md = outdir / "report.md"
    out_md.write_text(md_text, encoding='utf-8')
    return out_md


def main():
    p = argparse.ArgumentParser(description="Generate session/posture report from FaceSense CSVs")
    p.add_argument("--posture", default="facesense_posture/historico_postura.csv", help="Path to posture timeline CSV")
    p.add_argument("--sessions", default="facesense_posture/historico_sessoes.csv", help="Path to sessions CSV (optional)")
    p.add_argument("--outdir", default="out/report", help="Output directory")
    p.add_argument("--stress-threshold", type=float, default=0.6, help="Threshold for counting high stress samples")
    p.add_argument("--correction-window", type=int, default=30, help="Seconds after alert to look for correction")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    imgs = {}

    # Load posture CSV
    posture_path = Path(args.posture)
    if posture_path.exists():
        df_posture = pd.read_csv(posture_path)
        try:
            df_posture, ts_col = parse_time_column(df_posture)
        except Exception as e:
            print(f"Warning: could not parse timestamps for posture CSV: {e}", file=sys.stderr)
            # continue with raw frame index
            df_posture = pd.read_csv(posture_path)
            df_posture.index = pd.RangeIndex(len(df_posture))
            ts_col = None
        posture_col = detect_posture_column(df_posture)
        if posture_col is None:
            print("Warning: no posture/event column detected. Posture metrics will be limited.")
        pm = compute_posture_metrics(df_posture, posture_col)
        metrics['posture'] = pm
        # optionally compute correction rate
        cr = compute_correction_rate(df_posture, posture_col, window_seconds=args.correction_window) if posture_col else None
        if cr is not None:
            metrics['correction_rate'] = cr
    else:
        print(f"Posture CSV not found at {posture_path}. Skipping posture metrics.")
        df_posture = None
        posture_col = None

    # Load session CSV and/or stress timeline
    sessions_path = Path(args.sessions)
    stress_ts = None
    if sessions_path.exists():
        df_sess = pd.read_csv(sessions_path)
        # try to detect a time-series of stress per small intervals (rare) else use per-session metrics
        # If sessions CSV contains timestamped rows and a stress probability column, use that
        try:
            df_sess, _ = parse_time_column(df_sess)
            stress_col = detect_stress_column(df_sess)
            if stress_col:
                stress_ts = df_sess[stress_col]
        except Exception:
            # maybe sessions CSV is per-session summary, look for stress column
            stress_col = detect_stress_column(df_sess)
            if stress_col is not None:
                # create a synthetic series from session means
                stress_ts = pd.Series(df_sess[stress_col].astype(float).values, index=pd.RangeIndex(len(df_sess)))
    else:
        print(f"Sessions CSV not found at {sessions_path}. Skipping sessions/stress metrics.")

    # If we didn't find stress in sessions, try to detect stress in posture CSV
    if stress_ts is None and df_posture is not None:
        stress_col = detect_stress_column(df_posture)
        if stress_col:
            stress_ts = df_posture[stress_col]

    if stress_ts is not None:
        sm = compute_stress_metrics(stress_ts, threshold=args.stress_threshold)
        metrics['stress'] = sm

    # Plots
    imgs = plot_stress_and_alerts(stress_ts, df_posture, posture_col, outdir)

    # Markdown report
    note = "Generated by scripts/report_generator.py"
    report_file = generate_markdown_report(outdir, metrics, imgs, note=note)
    print(f"Report generated: {report_file}\nImages: {', '.join([str(v) for v in imgs.values()])}")


if __name__ == '__main__':
    main()
