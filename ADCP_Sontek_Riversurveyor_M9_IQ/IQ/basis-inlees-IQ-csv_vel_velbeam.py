#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SonTek-IQ: volledige workflow (AUTOMATISCH) voor MAIN + VEL + VELBEAM + SNR
==========================================================================
Wat dit script doet (end-to-end, in 1 run):

A) Je kiest 1 bestand (meestal de MAIN:  <base>.csv)
   → script zoekt automatisch in dezelfde map:
      <base>.csv          = MAIN
      <base>_VEL.csv      = VEL        (verwerkte snelheden per cel)
      <base>_VELBEAM.csv  = VELBEAM    (ruwe beam-snelheden per cel)
      <base>_SNR.csv      = SNR        (SNR per beam per cel)  [optioneel]

B) Per gevonden file:
   - robuuste encoding/delimiter detectie
   - volledige parameter-inventaris (kolomnamen)
   - sanity check:
       * #records
       * tijd: parse, monotonic, duplicaten, dt-statistiek, gaps
       * sample: parse, monotonic, duplicaten, continuity gaps

C) Header-validatie (instrument-conform):
   - SONTEK-IQ velocity-beams = 1..4 (beam 5 is vertical beam voor waterstand → nooit velocity)
   - Detecteert cellen obv patronen zoals:
        "Cell1 Velocity (beam).1" ... ".4"
   - Controleert:
        * aantal cellen == EXPECTED_CELLS (default 20)
        * per cel exact beams 1..4 aanwezig (voor VELBEAM en/of bestanden die dit bevatten)
   - MAIN file mag géén CellX Velocity (beam).Y hebben → dat is OK

D) Koppelen tot 1 dataset:
   - Voorkeur: join op sample-kolom (exact)
   - Anders: join op tijd (exact)
   - Anders: nearest (merge_asof) met tolerantie (default 1s)

E) Outputbestanden (naast de gekozen file):
   - iq_full_report.txt
   - iq_full_report.json
   - iq_combined.csv  (samengevoegde dataset met suffixen)

Instellingen:
- EXPECTED_CELLS = 20
- TIME_TOL_SECONDS = 1.0

"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Tkinter (standaard Windows)
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None


# =========================
# CONFIG
# =========================
EXPECTED_CELLS = 20
EXPECTED_VELOCITY_BEAMS = [1, 2, 3, 4]
TIME_TOL_SECONDS = 1.0        # voor merge_asof (nearest op tijd)
GAP_FACTOR = 5.0              # gap = dt > GAP_FACTOR * median(dt)
WRITE_COMBINED_AS = "csv"     # "csv" of "tsv"
DECIMAL = "."                 # enkel relevant als je later formatteert; pandas schrijft standaard '.'


# =========================
# IO helpers: encoding/delim
# =========================
def _read_sample_bytes(path: str, n: int = 300_000) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)

def detect_encoding(path: str) -> str:
    raw = _read_sample_bytes(path)
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except Exception:
        return "latin-1"

def detect_delimiter(path: str, encoding: str) -> str:
    raw = _read_sample_bytes(path)
    text = raw.decode(encoding, errors="replace")
    lines = text.splitlines()
    sample = "\n".join(lines[:80])

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        header = lines[0] if lines else ""
        candidates = [",", ";", "\t", "|"]
        best = ","
        best_cols = 1
        for d in candidates:
            cols = header.count(d) + 1 if header else 1
            if cols > best_cols:
                best_cols = cols
                best = d
        return best

def read_csv_robust(path: str) -> Tuple[pd.DataFrame, str, str]:
    enc = detect_encoding(path)
    sep = detect_delimiter(path, enc)
    df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df, enc, sep


# =========================
# Auto-detect related files
# =========================
def detect_related_files(selected_file: str) -> Dict[str, Optional[str]]:
    folder = os.path.dirname(os.path.abspath(selected_file))
    base = os.path.splitext(os.path.basename(selected_file))[0]

    # Als gebruiker per ongeluk _VEL of _VELBEAM kiest: base terugzetten
    for suffix in ["_VELBEAM", "_VEL", "_SNR"]:
        if base.upper().endswith(suffix):
            base = base[: -len(suffix)]
            break

    expected = {
        "MAIN": os.path.join(folder, f"{base}.csv"),
        "VEL": os.path.join(folder, f"{base}_VEL.csv"),
        "VELBEAM": os.path.join(folder, f"{base}_VELBEAM.csv"),
        "SNR": os.path.join(folder, f"{base}_SNR.csv"),
    }

    found: Dict[str, Optional[str]] = {}
    for k, p in expected.items():
        found[k] = p if os.path.exists(p) else None
    return found


# =========================
# Column detection (time/sample)
# =========================
TIME_CANDIDATES = [
    "Timestamp", "TimeStamp", "DateTime", "Date Time", "Date/Time", "Time", "Datetime",
    "UTC", "Local Time", "Sample Time", "Measurement Time",
    "Date", "Date (UTC)", "Time (UTC)"
]
SAMPLE_CANDIDATES = [
    "Sample", "Sample #", "Sample#", "SampleNumber", "Sample Number",
    "Record", "Record #", "Ensemble", "Index"
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def find_best_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_norm = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in cols_norm:
            return cols_norm[key]
    for c in cols:
        cn = _norm(c)
        for cand in candidates:
            if _norm(cand) in cn:
                return c
    return None

def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    col = find_best_column(list(df.columns), TIME_CANDIDATES)
    if col is not None:
        return col

    # fallback: eerste kolom met >80% parsebare datetimes in top 50 rijen
    for c in df.columns[:12]:
        s = df[c].astype(str).head(50)
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if parsed.notna().mean() > 0.8:
            return c
    return None

def detect_sample_column(df: pd.DataFrame) -> Optional[str]:
    col = find_best_column(list(df.columns), SAMPLE_CANDIDATES)
    if col is not None:
        return col

    # fallback: integer-ish monotonic in begin
    for c in df.columns[:20]:
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().mean() < 0.9:
            continue
        s = ser.dropna()
        if len(s) < 20:
            continue
        if (s % 1 == 0).mean() < 0.98:
            continue
        if s.head(200).is_monotonic_increasing:
            return c
    return None


# =========================
# Header: cell/beam detectie
# =========================
CELL_BEAM_RE = re.compile(
    r"^Cell\s*0*(?P<cell>\d+)\s+Velocity\s*\(beam\)\.(?P<beam>\d+)\s*$",
    re.IGNORECASE
)
ALT_CELL_BEAM_RES = [
    re.compile(r"^Cell\s*0*(?P<cell>\d+)\s+Vel(?:ocity)?\s*\(beam\)\.(?P<beam>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^Cell\s*0*(?P<cell>\d+)\s+Velocity\s+Beam\s*(?P<beam>\d+)\s*$", re.IGNORECASE),
]

def match_cell_beam(colname: str) -> Optional[Tuple[int, int]]:
    s = colname.strip()
    m = CELL_BEAM_RE.match(s)
    if m:
        return int(m.group("cell")), int(m.group("beam"))
    for rx in ALT_CELL_BEAM_RES:
        m2 = rx.match(s)
        if m2:
            return int(m2.group("cell")), int(m2.group("beam"))
    return None

@dataclass
class HeaderValidation:
    has_cell_beam_velocity: bool
    detected_cells: List[int]
    beams_per_cell: Dict[int, List[int]]
    expected_cells: int
    expected_beams: List[int]
    missing_beams_per_cell: Dict[int, List[int]]
    unexpected_beams_per_cell: Dict[int, List[int]]
    wrong_cell_count: bool
    notes: List[str]
    ok: bool

def validate_cell_beam_headers(columns: List[str], expected_cells: int, expected_beams: List[int]) -> HeaderValidation:
    beams_per_cell: Dict[int, Set[int]] = {}
    for col in columns:
        cb = match_cell_beam(col)
        if cb is None:
            continue
        cell, beam = cb
        beams_per_cell.setdefault(cell, set()).add(beam)

    detected_cells = sorted(beams_per_cell.keys())
    has = len(detected_cells) > 0

    missing: Dict[int, List[int]] = {}
    unexpected: Dict[int, List[int]] = {}
    expected_set = set(expected_beams)

    notes: List[str] = []
    ok = True

    if not has:
        # geen cell/beam headers → kan main of vel zijn: niet per definitie fout
        return HeaderValidation(
            has_cell_beam_velocity=False,
            detected_cells=[],
            beams_per_cell={},
            expected_cells=expected_cells,
            expected_beams=expected_beams,
            missing_beams_per_cell={},
            unexpected_beams_per_cell={},
            wrong_cell_count=False,
            notes=["Geen 'CellX Velocity (beam).Y' kolommen gevonden (kan normaal zijn voor MAIN/_VEL)."],
            ok=True,
        )

    # check per cel
    for c in detected_cells:
        bset = beams_per_cell.get(c, set())
        miss = sorted(list(expected_set - bset))
        unexp = sorted(list(bset - expected_set))
        if miss:
            missing[c] = miss
        if unexp:
            unexpected[c] = unexp

    # check beam 5 velocity → instrument-fout
    if any(5 in beams_per_cell[c] for c in detected_cells):
        notes.append("ONVERWACHT: velocity-kolommen voor beam 5 gedetecteerd. Beam 5 is verticaal (waterstand), geen velocity.")
        ok = False

    wrong_cell_count = (len(detected_cells) != expected_cells)
    if wrong_cell_count:
        notes.append(f"Aantal cellen gedetecteerd = {len(detected_cells)}, verwacht = {expected_cells}.")
        ok = False

    if missing:
        notes.append("Ontbrekende beams (verwacht 1..4) in minstens één cel.")
        ok = False
    if unexpected:
        notes.append("Onverwachte beam-index in minstens één cel.")
        ok = False

    return HeaderValidation(
        has_cell_beam_velocity=True,
        detected_cells=detected_cells,
        beams_per_cell={c: sorted(list(beams_per_cell[c])) for c in detected_cells},
        expected_cells=expected_cells,
        expected_beams=expected_beams,
        missing_beams_per_cell=missing,
        unexpected_beams_per_cell=unexpected,
        wrong_cell_count=wrong_cell_count,
        notes=notes,
        ok=ok,
    )


# =========================
# Sanity checks: time & sample
# =========================
@dataclass
class TimeCheck:
    time_col: Optional[str]
    parse_ok_fraction: float
    n_records: int
    n_unique_time: int
    n_duplicates: int
    monotonic_increasing: Optional[bool]
    dt_seconds_median: Optional[float]
    dt_seconds_min: Optional[float]
    dt_seconds_max: Optional[float]
    n_gaps: int
    largest_gap_seconds: Optional[float]

@dataclass
class SampleCheck:
    sample_col: Optional[str]
    n_records: int
    parse_ok_fraction: float
    monotonic_increasing: Optional[bool]
    n_duplicates: int
    n_gaps: int
    first_sample: Optional[int]
    last_sample: Optional[int]

def time_sanity(df: pd.DataFrame, time_col: Optional[str], gap_factor: float = GAP_FACTOR) -> TimeCheck:
    n = len(df)
    if time_col is None:
        return TimeCheck(None, 0.0, n, 0, 0, None, None, None, None, 0, None)

    t = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    ok_frac = float(t.notna().mean()) if n else 0.0

    t2 = t.dropna()
    n_unique = int(t2.nunique())
    n_dups = int(len(t2) - n_unique)

    mono = None
    dt_med = dt_min = dt_max = None
    n_gaps = 0
    largest_gap = None

    if len(t2) >= 3:
        mono = bool(t2.is_monotonic_increasing)

        t_sorted = t2.sort_values()
        dt = t_sorted.diff().dt.total_seconds().dropna()
        if len(dt) > 0:
            dt_med = float(dt.median())
            dt_min = float(dt.min())
            dt_max = float(dt.max())
            if dt_med and dt_med > 0:
                gaps = dt[dt > gap_factor * dt_med]
                n_gaps = int(len(gaps))
                largest_gap = float(gaps.max()) if len(gaps) else None

    return TimeCheck(
        time_col=time_col,
        parse_ok_fraction=ok_frac,
        n_records=n,
        n_unique_time=n_unique,
        n_duplicates=n_dups,
        monotonic_increasing=mono,
        dt_seconds_median=dt_med,
        dt_seconds_min=dt_min,
        dt_seconds_max=dt_max,
        n_gaps=n_gaps,
        largest_gap_seconds=largest_gap,
    )

def sample_sanity(df: pd.DataFrame, sample_col: Optional[str]) -> SampleCheck:
    n = len(df)
    if sample_col is None:
        return SampleCheck(None, n, 0.0, None, 0, 0, None, None)

    s = pd.to_numeric(df[sample_col], errors="coerce")
    ok_frac = float(s.notna().mean()) if n else 0.0
    s2 = s.dropna()

    n_dups = int(len(s2) - s2.nunique())
    mono = None
    n_gaps = 0
    first = last = None

    if len(s2) >= 2:
        mono = bool(s2.is_monotonic_increasing)
        # integer cast safe-ish
        first = int(float(s2.iloc[0]))
        last = int(float(s2.iloc[-1]))
        su = pd.Series(sorted(pd.unique(s2.astype("int64", errors="ignore"))))
        dif = su.diff().dropna()
        n_gaps = int((dif > 1).sum())

    return SampleCheck(
        sample_col=sample_col,
        n_records=n,
        parse_ok_fraction=ok_frac,
        monotonic_increasing=mono,
        n_duplicates=n_dups,
        n_gaps=n_gaps,
        first_sample=first,
        last_sample=last,
    )


# =========================
# Reporting structures
# =========================
@dataclass
class FileReport:
    label: str
    path: str
    encoding: str
    delimiter: str
    n_rows: int
    n_cols: int
    columns: List[str]
    time_check: TimeCheck
    sample_check: SampleCheck
    header_validation: HeaderValidation
    notes: List[str]

def make_file_report(label: str, path: str) -> Tuple[pd.DataFrame, FileReport]:
    df, enc, sep = read_csv_robust(path)
    time_col = detect_time_column(df)
    sample_col = detect_sample_column(df)

    tc = time_sanity(df, time_col)
    sc = sample_sanity(df, sample_col)

    hv = validate_cell_beam_headers(list(df.columns), EXPECTED_CELLS, EXPECTED_VELOCITY_BEAMS)

    notes: List[str] = []
    if tc.time_col and tc.parse_ok_fraction < 0.95:
        notes.append(f"Tijdkolom '{tc.time_col}' parseert slechts {tc.parse_ok_fraction:.1%}.")
    if tc.time_col and tc.monotonic_increasing is False:
        notes.append("Tijdstempel is niet monotonic stijgend (volgorde bevat terug-sprongen).")
    if tc.n_duplicates > 0:
        notes.append(f"Duplicaten in tijdstempel: {tc.n_duplicates}.")
    if tc.n_gaps > 0:
        notes.append(f"Gaten: {tc.n_gaps} (dt > {GAP_FACTOR}× mediaan), grootste gat = {tc.largest_gap_seconds}s.")

    if sc.sample_col and sc.parse_ok_fraction < 0.98:
        notes.append(f"Samplekolom '{sc.sample_col}' parseert slechts {sc.parse_ok_fraction:.1%}.")
    if sc.sample_col and sc.monotonic_increasing is False:
        notes.append("Sample-nummer is niet monotonic stijgend.")
    if sc.n_duplicates > 0:
        notes.append(f"Duplicaten in sample: {sc.n_duplicates}.")
    if sc.n_gaps > 0:
        notes.append(f"Sample continuity gaps: {sc.n_gaps} (verschil > 1 in unieke samples).")

    # Label-specifieke interpretatie
    if label == "MAIN" and hv.has_cell_beam_velocity:
        notes.append("Opmerking: MAIN bevat toch CellX Velocity (beam).Y kolommen (ongewoon, check export).")
    if label in ("VELBEAM", "SNR") and (not hv.has_cell_beam_velocity):
        notes.append(f"Opmerking: {label} bevat geen CellX Velocity (beam).Y patroon — mogelijk andere export naming, of geen profieldata.")

    if hv.notes:
        notes.extend(hv.notes)

    rep = FileReport(
        label=label,
        path=os.path.abspath(path),
        encoding=enc,
        delimiter=sep,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=list(df.columns),
        time_check=tc,
        sample_check=sc,
        header_validation=hv,
        notes=notes,
    )
    return df, rep


# =========================
# Merge logic
# =========================
@dataclass
class MergeReport:
    method: str
    key: str
    tolerance_seconds: Optional[float]
    rows_main: int
    rows_vel: int
    rows_velbeam: int
    rows_snr: int
    rows_merged: int
    notes: List[str]

def _prepare_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", dayfirst=True)
    return out

def _suffix_nonkey(df: pd.DataFrame, key: str, suffix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c if c == key else f"{c}{suffix}" for c in out.columns]
    return out

def merge_all(
    df_main: pd.DataFrame, rep_main: FileReport,
    df_vel: Optional[pd.DataFrame], rep_vel: Optional[FileReport],
    df_velbeam: Optional[pd.DataFrame], rep_velbeam: Optional[FileReport],
    df_snr: Optional[pd.DataFrame], rep_snr: Optional[FileReport],
    time_tol_seconds: float = TIME_TOL_SECONDS,
) -> Tuple[pd.DataFrame, MergeReport]:

    notes: List[str] = []
    rows_main = len(df_main)
    rows_vel = len(df_vel) if df_vel is not None else 0
    rows_velbeam = len(df_velbeam) if df_velbeam is not None else 0
    rows_snr = len(df_snr) if df_snr is not None else 0

    # Join candidates (sample first)
    s_main = rep_main.sample_check.sample_col
    t_main = rep_main.time_check.time_col

    # Helpers to rename keys in other dfs
    def _align_key(df: pd.DataFrame, src_key: str, target_key: str) -> pd.DataFrame:
        if src_key == target_key:
            return df
        return df.rename(columns={src_key: target_key})

    # ---------- SAMPLE exact ----------
    # Vereist dat de andere files sample-kolom hebben
    if s_main:
        can_sample = True
        parts = []

        # MAIN is base, keep as is
        base_key = s_main

        # Prepare list of (df, rep, suffix)
        for df, rep, suffix in [
            (df_vel, rep_vel, "_VEL"),
            (df_velbeam, rep_velbeam, "_VELBEAM"),
            (df_snr, rep_snr, "_SNR"),
        ]:
            if df is None or rep is None:
                continue
            if not rep.sample_check.sample_col:
                can_sample = False
                notes.append(f"Sample-merge niet mogelijk: {rep.label} heeft geen sample-kolom.")
                break
            d = _align_key(df, rep.sample_check.sample_col, base_key)
            d = _suffix_nonkey(d, base_key, suffix)
            parts.append(d)

        if can_sample and parts:
            merged = df_main.copy()
            for p in parts:
                merged = merged.merge(p, on=base_key, how="inner")
            notes.append("Gekoppeld op sample-nummer (exact inner joins).")
            return merged, MergeReport(
                method="sample_exact_inner",
                key=base_key,
                tolerance_seconds=None,
                rows_main=rows_main,
                rows_vel=rows_vel,
                rows_velbeam=rows_velbeam,
                rows_snr=rows_snr,
                rows_merged=len(merged),
                notes=notes,
            )

    # ---------- TIME exact / asof ----------
    if t_main:
        base_key = t_main
        tol = pd.Timedelta(seconds=float(time_tol_seconds))

        # Parse & sort MAIN
        merged = _prepare_time(df_main, base_key).dropna(subset=[base_key]).sort_values(base_key)

        # Attempt exact first if all have same timestamps, else asof
        def _merge_time(merged_df: pd.DataFrame, df: pd.DataFrame, rep: FileReport, suffix: str) -> pd.DataFrame:
            nonlocal notes
            if not rep.time_check.time_col:
                notes.append(f"Tijd-merge overslaan: {rep.label} heeft geen tijdkolom.")
                return merged_df

            d = _align_key(df, rep.time_check.time_col, base_key)
            d = _prepare_time(d, base_key).dropna(subset=[base_key]).sort_values(base_key)
            d = _suffix_nonkey(d, base_key, suffix)

            # try exact inner to see if good overlap
            exact = merged_df.merge(d, on=base_key, how="inner")
            # overlap ratio
            ratio = len(exact) / max(1, len(merged_df))
            if ratio > 0.95:
                notes.append(f"{rep.label}: tijd-merge exact (inner) gebruikt (overlap ~{ratio:.1%}).")
                return exact.sort_values(base_key)

            # fallback: asof nearest
            notes.append(f"{rep.label}: tijd-merge ASOF nearest gebruikt (tolerantie {time_tol_seconds}s, overlap exact ~{ratio:.1%}).")
            return pd.merge_asof(
                merged_df.sort_values(base_key),
                d.sort_values(base_key),
                on=base_key,
                direction="nearest",
                tolerance=tol
            )

        if df_vel is not None and rep_vel is not None:
            merged = _merge_time(merged, df_vel, rep_vel, "_VEL")
        if df_velbeam is not None and rep_velbeam is not None:
            merged = _merge_time(merged, df_velbeam, rep_velbeam, "_VELBEAM")
        if df_snr is not None and rep_snr is not None:
            merged = _merge_time(merged, df_snr, rep_snr, "_SNR")

        method = "time_exact_or_asof"
        notes.append("Koppeling op tijd uitgevoerd (exact waar mogelijk, anders ASOF).")
        return merged, MergeReport(
            method=method,
            key=base_key,
            tolerance_seconds=float(time_tol_seconds),
            rows_main=rows_main,
            rows_vel=rows_vel,
            rows_velbeam=rows_velbeam,
            rows_snr=rows_snr,
            rows_merged=len(merged),
            notes=notes,
        )

    # fallback: enkel main
    notes.append("Kon niet koppelen: MAIN heeft geen sample- of tijdkolom (of niet detecteerbaar).")
    return df_main.copy(), MergeReport(
        method="none",
        key="",
        tolerance_seconds=None,
        rows_main=rows_main,
        rows_vel=rows_vel,
        rows_velbeam=rows_velbeam,
        rows_snr=rows_snr,
        rows_merged=len(df_main),
        notes=notes,
    )


# =========================
# Text report formatting
# =========================
def _fmt_time(tc: TimeCheck) -> str:
    if not tc.time_col:
        return "geen tijdkolom gedetecteerd"
    s = (f"kolom={tc.time_col} | parseOK={tc.parse_ok_fraction:.1%} | unique={tc.n_unique_time} | "
         f"dups={tc.n_duplicates} | monotonic={tc.monotonic_increasing}")
    if tc.dt_seconds_median is not None:
        s += (f" | dt(s): med={tc.dt_seconds_median:.3f}, min={tc.dt_seconds_min:.3f}, max={tc.dt_seconds_max:.3f} | "
              f"gaps={tc.n_gaps}, largestGap={tc.largest_gap_seconds}")
    return s

def _fmt_sample(sc: SampleCheck) -> str:
    if not sc.sample_col:
        return "geen samplekolom gedetecteerd"
    return (f"kolom={sc.sample_col} | parseOK={sc.parse_ok_fraction:.1%} | dups={sc.n_duplicates} | "
            f"gaps={sc.n_gaps} | monotonic={sc.monotonic_increasing} | first={sc.first_sample} | last={sc.last_sample}")

def _fmt_header(hv: HeaderValidation) -> str:
    if not hv.has_cell_beam_velocity:
        return "geen CellX Velocity (beam).Y patroon (OK voor MAIN/_VEL)."
    s = f"cellen={len(hv.detected_cells)} (verwacht {hv.expected_cells}), beams verwacht {hv.expected_beams}"
    if hv.wrong_cell_count:
        s += " | WRONG_CELL_COUNT"
    if hv.missing_beams_per_cell:
        s += f" | missingBeams(cellen)={len(hv.missing_beams_per_cell)}"
    if hv.unexpected_beams_per_cell:
        s += f" | unexpectedBeams(cellen)={len(hv.unexpected_beams_per_cell)}"
    s += f" | ok={hv.ok}"
    return s

def format_file_report(rep: FileReport) -> str:
    lines = []
    lines.append("-" * 92)
    lines.append(f"{rep.label}: {os.path.basename(rep.path)}")
    lines.append(f"  Pad: {rep.path}")
    lines.append(f"  Encoding={rep.encoding} | Delimiter={repr(rep.delimiter)} | rows={rep.n_rows} | cols={rep.n_cols}")
    lines.append(f"  Tijd:   {_fmt_time(rep.time_check)}")
    lines.append(f"  Sample: {_fmt_sample(rep.sample_check)}")
    lines.append(f"  Header: {_fmt_header(rep.header_validation)}")

    if rep.header_validation.has_cell_beam_velocity and rep.header_validation.detected_cells:
        # korte samenvatting beams van eerste/laatste cel
        cmin, cmax = rep.header_validation.detected_cells[0], rep.header_validation.detected_cells[-1]
        lines.append(f"  Voorbeeld beams: Cell{cmin} -> {rep.header_validation.beams_per_cell.get(cmin)} ; "
                     f"Cell{cmax} -> {rep.header_validation.beams_per_cell.get(cmax)}")

    if rep.notes:
        lines.append("  Notities:")
        for n in rep.notes:
            lines.append(f"    * {n}")

    lines.append("  Parameters (kolommen):")
    for c in rep.columns:
        lines.append(f"    - {c}")

    return "\n".join(lines)

def format_merge_report(mr: MergeReport) -> str:
    lines = []
    lines.append("=" * 92)
    lines.append("MERGE")
    lines.append(f"  Methode: {mr.method}")
    if mr.key:
        lines.append(f"  Key: {mr.key}")
    if mr.tolerance_seconds is not None:
        lines.append(f"  Tolerantie: {mr.tolerance_seconds}s")
    lines.append(f"  Rows: MAIN={mr.rows_main} | VEL={mr.rows_vel} | VELBEAM={mr.rows_velbeam} | SNR={mr.rows_snr} | MERGED={mr.rows_merged}")
    if mr.notes:
        lines.append("  Notities:")
        for n in mr.notes:
            lines.append(f"    * {n}")
    lines.append("=" * 92)
    return "\n".join(lines)


# =========================
# GUI / CLI
# =========================
def pick_one_file_gui() -> Optional[str]:
    if tk is None or filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Kies 1 SonTek-IQ CSV (bij voorkeur MAIN: <base>.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

def main(argv: List[str]) -> int:
    # 1) input
    selected = argv[1] if len(argv) > 1 else pick_one_file_gui()
    if not selected:
        print("Geen bestand gekozen. Stop.")
        return 2

    related = detect_related_files(selected)

    print("\nGevonden bestanden (zelfde map, zelfde base):")
    for k, v in related.items():
        print(f"  {k:7s}: {v if v else 'NIET GEVONDEN'}")

    if related["MAIN"] is None:
        print("\nFOUT: MAIN file (<base>.csv) niet gevonden. Kies een bestand uit de juiste map.")
        return 3

    # 2) inlezen + rapporten
    dfs: Dict[str, Optional[pd.DataFrame]] = {"MAIN": None, "VEL": None, "VELBEAM": None, "SNR": None}
    reps: Dict[str, Optional[FileReport]] = {"MAIN": None, "VEL": None, "VELBEAM": None, "SNR": None}

    for label in ["MAIN", "VEL", "VELBEAM", "SNR"]:
        path = related[label]
        if path is None:
            continue
        df, rep = make_file_report(label, path)
        dfs[label] = df
        reps[label] = rep

    # 3) merge (MAIN + optionele andere)
    merged, merge_rep = merge_all(
        dfs["MAIN"], reps["MAIN"],   # type: ignore
        dfs["VEL"], reps["VEL"],
        dfs["VELBEAM"], reps["VELBEAM"],
        dfs["SNR"], reps["SNR"],
        time_tol_seconds=TIME_TOL_SECONDS,
    )

    # 4) outputs
    out_dir = os.path.dirname(os.path.abspath(related["MAIN"]))  # type: ignore
    out_txt = os.path.join(out_dir, "iq_full_report.txt")
    out_json = os.path.join(out_dir, "iq_full_report.json")
    out_combined = os.path.join(out_dir, "iq_combined.csv" if WRITE_COMBINED_AS.lower() == "csv" else "iq_combined.tsv")

    # TXT report
    parts = []
    parts.append("SONTEK-IQ FULL REPORT")
    parts.append(f"EXPECTED_CELLS={EXPECTED_CELLS}, EXPECTED_VELOCITY_BEAMS={EXPECTED_VELOCITY_BEAMS}, TIME_TOL_SECONDS={TIME_TOL_SECONDS}")
    parts.append("")

    for label in ["MAIN", "VEL", "VELBEAM", "SNR"]:
        if reps[label] is not None:
            parts.append(format_file_report(reps[label]))  # type: ignore

    parts.append(format_merge_report(merge_rep))
    report_txt = "\n\n".join(parts)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_txt)

    # JSON report
    payload = {
        "config": {
            "EXPECTED_CELLS": EXPECTED_CELLS,
            "EXPECTED_VELOCITY_BEAMS": EXPECTED_VELOCITY_BEAMS,
            "TIME_TOL_SECONDS": TIME_TOL_SECONDS,
            "GAP_FACTOR": GAP_FACTOR,
            "WRITE_COMBINED_AS": WRITE_COMBINED_AS,
        },
        "files_found": related,
        "file_reports": {k: (asdict(v) if v is not None else None) for k, v in reps.items()},
        "merge_report": asdict(merge_rep),
        "outputs": {
            "report_txt": out_txt,
            "report_json": out_json,
            "combined": out_combined,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Combined export
    if WRITE_COMBINED_AS.lower() == "tsv":
        merged.to_csv(out_combined, index=False, sep="\t")
    else:
        merged.to_csv(out_combined, index=False)

    # console summary
    print("\n" + report_txt)
    print("\nBestanden geschreven:")
    print(f" - {out_txt}")
    print(f" - {out_json}")
    print(f" - {out_combined}")

    # exitcode
    # Let op: header-validatie kan OK zijn voor MAIN (geen cell-beam headers), maar niet OK voor VELBEAM/SNR.
    # We beschouwen 'succes' als: merge gelukt (niet-empty) + MAIN aanwezig.
    return 0 if len(merged) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
