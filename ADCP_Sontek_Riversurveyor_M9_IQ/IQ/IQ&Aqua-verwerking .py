#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SonTek IQ – CORRECTE verwerking uit VELBEAM:
- u/v/w berekend uit beam1..4 (Janus) per cel
- East/North/Up via heading (indien aanwezig)
- Pressure subplot onder elke tijdslijn
- Rose plot: frequentie (%) + gemiddelde snelheid (m/s) per sector
- Sanity checks: mean binnen [min,max] over cellen per tijdstap

INPUT:
- Kies 1 bestand; script zoekt automatisch <base>.csv, <base>_VEL.csv, <base>_VELBEAM.csv, <base>_SNR.csv
- Vereist: <base>_VELBEAM.csv

Opmerking:
- IQ heeft 5 beams, maar velocity Janus gebruikt 4 beams (1..4). Beam 5 is vertical beam voor depth, niet voor Janus u/v.
"""

from __future__ import annotations

import io
import os
import re
import csv
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from docx import Document
    from docx.shared import Inches
except ImportError:  # pragma: no cover - env may not have python-docx
    Document = None  # type: ignore[assignment]
    Inches = None  # type: ignore[assignment]


# -------------------------
# Config
# -------------------------
BEAMS = [1, 2, 3, 4]
DEFAULT_MA = 30
DEFAULT_ROSE_BINS = 36
ASOF_TOL_SECONDS = 1.0

# Beam angle t.o.v. verticaal (deg) – aanpasbaar in GUI
DEFAULT_BEAM_ANGLE_DEG = 25.0

# Conventie heading:
# heading = graden, 0=N, 90=E (standaard)
# instrument u/v (in instrument XY) -> rotate to EN:
# E = u*sin(h) + v*cos(h)
# N = u*cos(h) - v*sin(h)
# (dit is een veelgebruikte conventie; toggle voorzien)
DEFAULT_HEADING_CONVENTION = "ENU_from_heading"  # of "swap_sign" indien nodig


# -------------------------
# Robust CSV read
# -------------------------
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

def detect_delimiter(path: str, enc: str) -> str:
    raw = _read_sample_bytes(path)
    text = raw.decode(enc, errors="replace")
    lines = text.splitlines()
    sample = "\n".join(lines[:80])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        header = lines[0] if lines else ""
        best = ","
        best_cols = 1
        for d in [",", ";", "\t", "|"]:
            cols = header.count(d) + 1 if header else 1
            if cols > best_cols:
                best_cols = cols
                best = d
        return best

def read_csv_robust(path: str) -> pd.DataFrame:
    enc = detect_encoding(path)
    sep = detect_delimiter(path, enc)
    df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


# -------------------------
# File autodetect
# -------------------------
def detect_related_files(selected_file: str) -> Dict[str, Optional[str]]:
    folder = os.path.dirname(os.path.abspath(selected_file))
    base = os.path.splitext(os.path.basename(selected_file))[0]
    for suffix in ["_VELBEAM", "_VEL", "_SNR"]:
        if base.upper().endswith(suffix):
            base = base[: -len(suffix)]
            break
    cand = {
        "MAIN": os.path.join(folder, f"{base}.csv"),
        "VEL": os.path.join(folder, f"{base}_VEL.csv"),
        "VELBEAM": os.path.join(folder, f"{base}_VELBEAM.csv"),
        "SNR": os.path.join(folder, f"{base}_SNR.csv"),
    }
    return {k: (p if os.path.exists(p) else None) for k, p in cand.items()}


# -------------------------
# Column detection helpers
# -------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    m = {_norm(c): c for c in cols}
    for cand in candidates:
        k = _norm(cand)
        if k in m:
            return m[k]
    for c in cols:
        cn = _norm(c)
        for cand in candidates:
            if _norm(cand) in cn:
                return c
    return None

TIME_CANDS = ["Sample time", "Sample Time", "Timestamp", "DateTime", "Date Time", "Date/Time", "Time"]
HEADING_CANDS = ["Heading", "Heading (degrees)", "Heading (deg)"]
PRESS_CANDS = ["Depth (pressure) (dbar)", "Pressure (uncorrected) (dbar)", "Depth (m)", "Pressure (dbar)"]

# VELBEAM: CellX Velocity (beam).Y (m/s)
CELL_BEAM_RE = re.compile(
    r"^Cell\s*0*(?P<cell>\d+)\s+Velocity\s*\(beam\)\.(?P<beam>\d+)\s*(?:\([^)]*\))?\s*$",
    re.IGNORECASE
)

def parse_velbeam_map(df_velbeam: pd.DataFrame) -> Tuple[str, List[int], Dict[int, Dict[int, str]]]:
    time_col = find_col(df_velbeam, TIME_CANDS)
    if not time_col:
        raise ValueError("Geen tijdkolom gevonden in VELBEAM (verwacht bv. 'Sample Time').")

    cmap: Dict[int, Dict[int, str]] = {}
    for col in df_velbeam.columns:
        m = CELL_BEAM_RE.match(str(col).strip())
        if not m:
            continue
        cell = int(m.group("cell"))
        beam = int(m.group("beam"))
        if beam in BEAMS:
            cmap.setdefault(cell, {})[beam] = col

    cells = sorted(cmap.keys())
    if not cells:
        raise ValueError("Geen VELBEAM cell/beam kolommen gevonden (bv. 'Cell1 Velocity (beam).1 (m/s)').")
    return time_col, cells, cmap


# -------------------------
# Physics: Janus beams -> u,v,w
# -------------------------
def janus_to_uvwt(v1, v2, v3, v4, theta_deg: float):
    """
    Janus 4-beam (symmetrisch), theta = beam angle t.o.v. vertical.
    Gebruik standaard relaties:
      u_inst = (v1 - v2) / (2*sinθ)
      v_inst = (v4 - v3) / (2*sinθ)
      w_inst = (v1 + v2 + v3 + v4) / (4*cosθ)
    Tekens kunnen instrument-afhankelijk zijn; daarom houden we later sanity checks + heading toggle.
    """
    th = np.deg2rad(theta_deg)
    s = np.sin(th)
    c = np.cos(th)
    u = (v1 - v2) / (2.0 * s)
    v = (v4 - v3) / (2.0 * s)
    w = (v1 + v2 + v3 + v4) / (4.0 * c)
    return u, v, w

def rotate_uv_to_EN(u, v, heading_deg, convention: str = DEFAULT_HEADING_CONVENTION):
    """
    Rotate instrument (u,v) to East/North using heading.
    heading: 0=N, 90=E.
    """
    h = np.deg2rad(heading_deg.astype(float))
    if convention == "ENU_from_heading":
        E = u*np.sin(h) + v*np.cos(h)
        N = u*np.cos(h) - v*np.sin(h)
        return E, N
    elif convention == "swap_sign":
        # alternatieve fallback als instrument axes andersom blijken
        E = -(u*np.sin(h) + v*np.cos(h))
        N = -(u*np.cos(h) - v*np.sin(h))
        return E, N
    else:
        return u, v

def rolling_mean(s: pd.Series, n: int) -> pd.Series:
    n = max(1, int(n))
    if n == 1:
        return s
    return s.rolling(n, center=True, min_periods=max(1, n//3)).mean()

def speed_dir_from_EN(E, N):
    sp = np.hypot(E, N)
    ang = np.degrees(np.arctan2(E, N))  # atan2(E, N): 0=N, 90=E
    ang = (ang + 360.0) % 360.0
    return sp, ang


# -------------------------
# Rose stats
# -------------------------
def rose_bins(dir_deg: np.ndarray, sp: np.ndarray, n_bins: int):
    d = np.mod(dir_deg, 360.0)
    bins = np.linspace(0, 360, n_bins+1)
    idx = np.digitize(d, bins) - 1
    idx[idx == n_bins] = 0
    counts = np.zeros(n_bins)
    mean_sp = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = idx == k
        counts[k] = m.sum()
        if m.any():
            mean_sp[k] = np.nanmean(sp[m])
    centers = np.deg2rad((bins[:-1] + bins[1:]) / 2.0)
    freq_pct = 100.0 * counts / max(1.0, counts.sum())
    return centers, freq_pct, mean_sp


# -------------------------
# Aquadopp helpers
# -------------------------
def parse_aquadopp_hdr(hdr_path: str) -> Dict[str, Any]:
    with open(hdr_path, "r", errors="ignore") as f:
        text = f.read()

    def _find(pattern: str, cast=None, default=None):
        m = re.search(pattern, text)
        if not m:
            return default
        raw = m.group(1).strip()
        return cast(raw) if cast else raw

    n_meas = _find(r"Number of measurements\s+(\d+)", int)
    n_cells = _find(r"Number of cells\s+(\d+)", int)
    dt_sec = _find(r"Profile interval\s+([0-9.]+)\s*sec", float)
    t0_str = _find(
        r"(?im)^\s*(?:Time of first measurement|First measurement time|Time of first profile|Start time(?: of first measurement)?)\s*[:=]?\s*(.+?)\s*$",
        str,
    )

    if n_meas is None or n_cells is None or dt_sec is None or not t0_str:
        raise ValueError("Onvolledige .hdr metadata: n_meas/n_cells/dt/starttijd niet gevonden.")

    t0 = pd.NaT
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        t0 = pd.to_datetime(t0_str, format=fmt, dayfirst=True, errors="coerce")
        if not pd.isna(t0):
            break
    if pd.isna(t0):
        t0 = pd.to_datetime(t0_str, dayfirst=True, errors="coerce")
    if pd.isna(t0):
        raise ValueError(f"Kon starttijd niet parsen uit .hdr: '{t0_str}'.")

    mat_m = re.search(
        r"Transformation matrix\s+([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)\n\s*([0-9\.\-\s]+)",
        text,
    )
    if not mat_m:
        raise ValueError("Transformation matrix niet gevonden in .hdr.")
    rows = []
    for i in range(1, 4):
        row = [float(x) for x in mat_m.group(i).split()]
        if len(row) != 3:
            raise ValueError("Transformation matrix moet 3x3 zijn.")
        rows.append(row)
    T = np.array(rows, dtype=float)

    return {"n_meas": n_meas, "n_cells": n_cells, "dt_sec": dt_sec, "t0": t0, "T": T}


def load_aquadopp_v_file(path: str, n_meas: int, n_cells: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != n_cells:
        raise ValueError(f"{os.path.basename(path)}: verwacht {n_cells} kolommen, kreeg {arr.shape[1]}.")
    if arr.shape[0] != n_meas:
        arr = arr[: min(arr.shape[0], n_meas), :]
    return arr


def build_regular_time_index(t0: pd.Timestamp, n_meas: int, dt_sec: float) -> pd.DatetimeIndex:
    return pd.date_range(start=t0, periods=n_meas, freq=pd.to_timedelta(dt_sec, unit="s"))


def compute_xyz_from_beams(T: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = np.stack([b1, b2, b3], axis=-1)
    xyz = np.einsum("ij,tcj->tci", T, B)
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def rolling_mean_array(arr: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1:
        return arr
    return pd.DataFrame(arr).rolling(n, center=True, min_periods=max(1, n // 3)).mean().to_numpy()


def circular_diff_deg(a_deg: pd.Series, b_deg: pd.Series) -> pd.Series:
    d = (pd.Series(a_deg) - pd.Series(b_deg) + 180.0) % 360.0 - 180.0
    return d


# -------------------------
# GUI
# -------------------------
class IQApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SonTek IQ – Janus verwerking uit VELBEAM (correcte u/v/w, ENU, pressure subplots, rose)")
        self.geometry("1500x920")

        self.files = {"MAIN": None, "VEL": None, "VELBEAM": None, "SNR": None}
        self.df_main: Optional[pd.DataFrame] = None
        self.df_velbeam: Optional[pd.DataFrame] = None
        self.aqd_meta: Optional[Dict[str, Any]] = None
        self.aqd_time_index: Optional[pd.DatetimeIndex] = None
        self.aqd_v1: Optional[np.ndarray] = None
        self.aqd_v2: Optional[np.ndarray] = None
        self.aqd_v3: Optional[np.ndarray] = None
        self.aqd_cells: List[int] = []
        self.aqd_cell_vars: Dict[int, tk.BooleanVar] = {}
        self.aqd_base_path: Optional[str] = None

        self.time_col_vb: Optional[str] = None
        self.cells: List[int] = []
        self.cmap: Dict[int, Dict[int, str]] = {}

        self.heading_col: Optional[str] = None
        self.press_col: Optional[str] = None

        # UI vars
        self.ma_var = tk.IntVar(value=DEFAULT_MA)
        self.theta_var = tk.DoubleVar(value=DEFAULT_BEAM_ANGLE_DEG)
        self.heading_conv_var = tk.StringVar(value=DEFAULT_HEADING_CONVENTION)
        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")
        self.iq_shift_seconds_var = tk.DoubleVar(value=0.0)
        self.iq_stretch_var = tk.DoubleVar(value=1.0)
        self.iq_anchor_var = tk.StringVar(value="")

        self.beam_vars = {b: tk.BooleanVar(value=True) for b in BEAMS}
        self.cell_vars: Dict[int, tk.BooleanVar] = {}

        # debug / warnings
        self.msg_var = tk.StringVar(value="")

        # plots
        self.figures: Dict[str, Tuple[plt.Figure, FigureCanvasTkAgg]] = {}
        self.tab_titles: Dict[str, str] = {}
        self.last_processed: Optional[pd.DataFrame] = None
        self.last_export_meta: Optional[Dict[str, Any]] = None

        self._build_ui()

    def _build_ui(self):
        left_outer = ttk.Frame(self)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        self.left_scroll_canvas = tk.Canvas(left_outer, width=460, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=self.left_scroll_canvas.yview)
        self.left_scroll_inner = ttk.Frame(self.left_scroll_canvas)
        self.left_scroll_window = self.left_scroll_canvas.create_window((0, 0), window=self.left_scroll_inner, anchor="nw")
        self.left_scroll_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_scroll_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_scroll_inner.bind("<Configure>", self._update_left_scrollregion)
        self.left_scroll_canvas.bind("<Configure>", self._resize_left_scroll_inner)
        self.left_scroll_canvas.bind("<Enter>", self._bind_left_wheel)
        self.left_scroll_canvas.bind("<Leave>", self._unbind_left_wheel)

        left = self.left_scroll_inner
        right = ttk.Frame(self); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(left, text="Bestand kiezen (1 volstaat):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Button(left, text="Kies file…", command=self.choose_file).pack(fill=tk.X, pady=4)
        self.files_label = ttk.Label(left, text="Nog geen file gekozen.", wraplength=420, justify="left")
        self.files_label.pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Kies Aquadopp .hdr…", command=self.choose_aquadopp_hdr).pack(fill=tk.X, pady=(0, 4))
        self.aqd_label = ttk.Label(left, text="Aquadopp: niet geladen.", wraplength=420, justify="left")
        self.aqd_label.pack(fill=tk.X, pady=(0, 4))

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Beams (Janus):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for b in BEAMS:
            ttk.Checkbutton(left, text=f"Beam {b}", variable=self.beam_vars[b]).pack(anchor="w")

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Cellen (VELBEAM):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        frame = ttk.Frame(left); frame.pack(fill=tk.BOTH, expand=False)
        self.cells_canvas = tk.Canvas(frame, width=420, height=180)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.cells_canvas.yview)
        self.cells_inner = ttk.Frame(self.cells_canvas)
        self.cells_inner.bind("<Configure>", lambda e: self.cells_canvas.configure(scrollregion=self.cells_canvas.bbox("all")))
        self.cells_canvas.create_window((0, 0), window=self.cells_inner, anchor="nw")
        self.cells_canvas.configure(yscrollcommand=scroll.set)
        self.cells_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(left, text="Selecteer alle cellen", command=self.select_all).pack(fill=tk.X, pady=(6, 2))
        ttk.Button(left, text="Deselecteer alle cellen", command=self.deselect_all).pack(fill=tk.X, pady=(0, 6))

        ttk.Label(left, text="Cellen (Aquadopp):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        frame_aqd = ttk.Frame(left); frame_aqd.pack(fill=tk.BOTH, expand=False)
        self.aqd_canvas = tk.Canvas(frame_aqd, width=420, height=140)
        scroll_aqd = ttk.Scrollbar(frame_aqd, orient="vertical", command=self.aqd_canvas.yview)
        self.aqd_inner = ttk.Frame(self.aqd_canvas)
        self.aqd_inner.bind("<Configure>", lambda e: self.aqd_canvas.configure(scrollregion=self.aqd_canvas.bbox("all")))
        self.aqd_canvas.create_window((0, 0), window=self.aqd_inner, anchor="nw")
        self.aqd_canvas.configure(yscrollcommand=scroll_aqd.set)
        self.aqd_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_aqd.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(left, text="Selecteer alle Aquadopp-cellen", command=self.select_all_aqd).pack(fill=tk.X, pady=(6, 2))
        ttk.Button(left, text="Deselecteer alle Aquadopp-cellen", command=self.deselect_all_aqd).pack(fill=tk.X, pady=(0, 6))

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Parameters:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        r1 = ttk.Frame(left); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="MA (N):").pack(side=tk.LEFT)
        ttk.Spinbox(r1, from_=1, to=5000, textvariable=self.ma_var, width=8).pack(side=tk.LEFT, padx=6)

        r2 = ttk.Frame(left); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Beam angle θ (deg):").pack(side=tk.LEFT)
        ttk.Spinbox(r2, from_=10, to=40, increment=0.5, textvariable=self.theta_var, width=8).pack(side=tk.LEFT, padx=6)

        r3 = ttk.Frame(left); r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text="Heading conv:").pack(side=tk.LEFT)
        ttk.Combobox(r3, textvariable=self.heading_conv_var, values=["ENU_from_heading", "swap_sign"], width=16, state="readonly").pack(side=tk.LEFT, padx=6)

        r4 = ttk.Frame(left); r4.pack(fill=tk.X, pady=2)
        ttk.Label(r4, text="IQ shift (s):").pack(side=tk.LEFT)
        ttk.Entry(r4, textvariable=self.iq_shift_seconds_var, width=10).pack(side=tk.LEFT, padx=6)

        r5 = ttk.Frame(left); r5.pack(fill=tk.X, pady=2)
        ttk.Label(r5, text="IQ stretch factor:").pack(side=tk.LEFT)
        ttk.Entry(r5, textvariable=self.iq_stretch_var, width=10).pack(side=tk.LEFT, padx=6)

        r6 = ttk.Frame(left); r6.pack(fill=tk.X, pady=2)
        ttk.Label(r6, text="IQ anchor (optioneel):").pack(side=tk.LEFT)
        ttk.Entry(r6, textvariable=self.iq_anchor_var, width=24).pack(side=tk.LEFT, padx=6)

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Zoom (optioneel):", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Entry(left, textvariable=self.start_var).pack(fill=tk.X, pady=2)
        ttk.Entry(left, textvariable=self.end_var).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        ttk.Button(left, text="Herbereken en teken", command=self.recalc).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Exporteer data…", command=self.export_processed_data).pack(fill=tk.X, pady=(0, 6))
        ttk.Button(left, text="Exporteer Word…", command=self.export_to_word_report).pack(fill=tk.X, pady=(0, 6))

        ttk.Label(left, text="Sanity / meldingen:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10,2))
        ttk.Label(left, textvariable=self.msg_var, wraplength=420, justify="left").pack(fill=tk.X)

        # Notebook plots
        self.nb = ttk.Notebook(right); self.nb.pack(fill=tk.BOTH, expand=True)

        for b in BEAMS:
            self._add_timeseries_tab(f"Beam {b}", f"beam{b}")

        self._add_timeseries_tab("ENU East (E)", "E")
        self._add_timeseries_tab("ENU North (N)", "N")
        self._add_timeseries_tab("ENU Up (W)", "W")
        self._add_timeseries_tab("Speed (hor)", "speed")
        self._add_timeseries_tab("Direction (deg)", "dir")
        self._add_timeseries_tab("Vergelijking Speed", "cmp_speed")
        self._add_timeseries_tab("Vergelijking Direction", "cmp_dir")
        self._add_timeseries_tab("Vergelijking East", "cmp_E")
        self._add_timeseries_tab("Vergelijking North", "cmp_N")

        self._add_rose_tab("Rose (freq% + mean speed)")

    def _update_left_scrollregion(self, _event=None):
        self.left_scroll_canvas.configure(scrollregion=self.left_scroll_canvas.bbox("all"))

    def _resize_left_scroll_inner(self, event):
        self.left_scroll_canvas.itemconfigure(self.left_scroll_window, width=event.width)

    def _bind_left_wheel(self, _event=None):
        self.left_scroll_canvas.bind_all("<MouseWheel>", self._on_left_mousewheel)
        self.left_scroll_canvas.bind_all("<Button-4>", self._on_left_mousewheel)
        self.left_scroll_canvas.bind_all("<Button-5>", self._on_left_mousewheel)

    def _unbind_left_wheel(self, _event=None):
        self.left_scroll_canvas.unbind_all("<MouseWheel>")
        self.left_scroll_canvas.unbind_all("<Button-4>")
        self.left_scroll_canvas.unbind_all("<Button-5>")

    def _on_left_mousewheel(self, event):
        if hasattr(event, "delta") and event.delta:
            step = -1 if event.delta > 0 else 1
        else:
            num = getattr(event, "num", 0)
            step = -1 if num == 4 else 1
        self.left_scroll_canvas.yview_scroll(step, "units")

    def _add_timeseries_tab(self, title: str, key: str):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=title)
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        fig.add_subplot(211)  # main
        fig.add_subplot(212)  # pressure
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, frame).update()
        self.figures[key] = (fig, canvas)
        self.tab_titles[key] = title

    def _add_rose_tab(self, title: str):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=title)
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        fig.add_subplot(211, projection="polar")
        fig.add_subplot(212, projection="polar")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, frame).update()
        self.figures["rose"] = (fig, canvas)
        self.tab_titles["rose"] = title

    # ---------- actions ----------
    def choose_file(self):
        p = filedialog.askopenfilename(title="Kies 1 SonTek IQ CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not p:
            return
        self.files = detect_related_files(p)
        if not self.files["VELBEAM"]:
            messagebox.showerror("Fout", f"VELBEAM ontbreekt.\n{self.files}")
            return

        # load
        self.df_velbeam = read_csv_robust(self.files["VELBEAM"])
        self.time_col_vb, self.cells, self.cmap = parse_velbeam_map(self.df_velbeam)

        self.df_main = None
        if self.files["MAIN"]:
            self.df_main = read_csv_robust(self.files["MAIN"])
            self.heading_col = find_col(self.df_main, HEADING_CANDS)
            self.press_col = find_col(self.df_main, PRESS_CANDS)

        # build cell checkboxes
        for w in self.cells_inner.winfo_children():
            w.destroy()
        self.cell_vars.clear()
        for c in self.cells:
            v = tk.BooleanVar(value=(c <= 5))
            self.cell_vars[c] = v
            ttk.Checkbutton(self.cells_inner, text=f"Cell {c}", variable=v).pack(anchor="w")

        self._update_files_label()
        self.msg_var.set("Bestanden geladen. Kies cellen + druk ‘Herbereken en teken’.")

    def _update_files_label(self):
        self.files_label.configure(
            text="\n".join([f"{k}: {os.path.basename(v) if v else 'NIET'}" for k, v in self.files.items()])
        )
        if self.aqd_base_path:
            self.aqd_label.configure(text=f"Aquadopp: {os.path.basename(self.aqd_base_path)}")
        else:
            self.aqd_label.configure(text="Aquadopp: niet geladen.")

    def select_all(self):
        for v in self.cell_vars.values():
            v.set(True)

    def deselect_all(self):
        for v in self.cell_vars.values():
            v.set(False)

    def select_all_aqd(self):
        for v in self.aqd_cell_vars.values():
            v.set(True)

    def deselect_all_aqd(self):
        for v in self.aqd_cell_vars.values():
            v.set(False)

    def choose_aquadopp_hdr(self):
        p = filedialog.askopenfilename(title="Kies Aquadopp .hdr", filetypes=[("HDR", "*.hdr"), ("All", "*.*")])
        if not p:
            return
        try:
            meta = parse_aquadopp_hdr(p)
            base = os.path.splitext(p)[0]
            v1 = load_aquadopp_v_file(base + ".v1", meta["n_meas"], meta["n_cells"])
            v2 = load_aquadopp_v_file(base + ".v2", meta["n_meas"], meta["n_cells"])
            v3 = load_aquadopp_v_file(base + ".v3", meta["n_meas"], meta["n_cells"])
        except Exception as exc:
            messagebox.showerror("Aquadopp", f"Kon Aquadopp niet laden:\n{exc}")
            return

        n_eff = min(meta["n_meas"], v1.shape[0], v2.shape[0], v3.shape[0])
        self.aqd_meta = meta
        self.aqd_v1 = v1[:n_eff, :]
        self.aqd_v2 = v2[:n_eff, :]
        self.aqd_v3 = v3[:n_eff, :]
        self.aqd_time_index = build_regular_time_index(meta["t0"], n_eff, meta["dt_sec"])
        self.aqd_cells = list(range(meta["n_cells"]))
        self.aqd_base_path = base

        for w in self.aqd_inner.winfo_children():
            w.destroy()
        self.aqd_cell_vars.clear()
        for c in self.aqd_cells:
            v = tk.BooleanVar(value=True)
            self.aqd_cell_vars[c] = v
            ttk.Checkbutton(self.aqd_inner, text=f"Cell {c+1}", variable=v).pack(anchor="w")

        self._update_files_label()
        self.msg_var.set("IQ en Aquadopp geladen. Stel shift/stretch in en druk ‘Herbereken en teken’.")

    def _parse_time(self, s: str) -> Optional[pd.Timestamp]:
        s = (s or "").strip()
        if not s:
            return None
        t = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return None if pd.isna(t) else t

    def _transform_iq_time(self, t: pd.Series) -> Tuple[pd.Series, Optional[pd.Timestamp]]:
        t_series = pd.Series(pd.to_datetime(t, errors="coerce"))
        if t_series.empty:
            return t_series, None
        shift_sec = float(self.iq_shift_seconds_var.get())
        stretch = float(self.iq_stretch_var.get())
        if stretch <= 0.0:
            raise ValueError("IQ stretch factor moet > 0 zijn.")

        anchor_in = self._parse_time(self.iq_anchor_var.get())
        anchor = anchor_in if anchor_in is not None else pd.Timestamp(t_series.iloc[0])

        t_ns = t_series.astype("int64")
        anchor_ns = anchor.value
        shifted_ns = t_ns + int(round(shift_sec * 1e9))
        transformed_ns = anchor_ns + np.rint((shifted_ns - anchor_ns) * stretch).astype("int64")
        transformed = pd.to_datetime(transformed_ns)
        return pd.Series(transformed), anchor

    def recalc(self):
        if self.df_velbeam is None:
            return

        cells_sel = [c for c, v in self.cell_vars.items() if v.get()]
        beams_sel = [b for b, v in self.beam_vars.items() if v.get()]
        if not cells_sel:
            messagebox.showwarning("Selectie", "Vink minstens 1 cel aan.")
            return
        if set(beams_sel) != set(BEAMS):
            messagebox.showwarning("Beams", "Voor correcte Janus u/v/w moeten beams 1..4 actief zijn.")
            return

        ma_n = max(1, int(self.ma_var.get()))
        theta = float(self.theta_var.get())
        conv = self.heading_conv_var.get()

        t0 = self._parse_time(self.start_var.get())
        t1 = self._parse_time(self.end_var.get())

        # Prepare time
        dfb = self.df_velbeam.copy()
        dfb[self.time_col_vb] = pd.to_datetime(dfb[self.time_col_vb], errors="coerce", dayfirst=True)
        dfb = dfb.dropna(subset=[self.time_col_vb]).sort_values(self.time_col_vb)
        if t0 is not None:
            dfb = dfb[dfb[self.time_col_vb] >= t0]
        if t1 is not None:
            dfb = dfb[dfb[self.time_col_vb] <= t1]
        t_raw = dfb[self.time_col_vb]
        try:
            t, anchor_used = self._transform_iq_time(t_raw)
        except ValueError as exc:
            messagebox.showerror("IQ tijdtransformatie", str(exc))
            return

        # Pressure (from MAIN) -> merge asof on time
        pressure = None
        heading = None
        if self.df_main is not None:
            dm = self.df_main.copy()
            tcolm = find_col(dm, TIME_CANDS)
            if tcolm:
                dm[tcolm] = pd.to_datetime(dm[tcolm], errors="coerce", dayfirst=False)
                dm = dm.dropna(subset=[tcolm]).sort_values(tcolm)
                # asof merge (nearest)
                m = pd.merge_asof(
                    pd.DataFrame({self.time_col_vb: t_raw}).sort_values(self.time_col_vb),
                    dm[[tcolm] + ([self.press_col] if self.press_col else []) + ([self.heading_col] if self.heading_col else [])].rename(columns={tcolm: self.time_col_vb}),
                    on=self.time_col_vb,
                    direction="nearest",
                    tolerance=pd.Timedelta(seconds=ASOF_TOL_SECONDS),
                )
                if self.press_col and self.press_col in m.columns:
                    pressure = pd.to_numeric(m[self.press_col], errors="coerce")
                if self.heading_col and self.heading_col in m.columns:
                    heading = pd.to_numeric(m[self.heading_col], errors="coerce")

        # Build per-cell u/v/w then mean across selected cells at each time
        U_cells = []
        V_cells = []
        W_cells = []

        # also store beam means per cell if needed for plotting
        beam_series_by_cell = {b: [] for b in BEAMS}
        beam_mean_ma_results: Dict[int, pd.Series] = {}

        for c in cells_sel:
            cols = self.cmap.get(c, {})
            v1 = pd.to_numeric(dfb[cols[1]], errors="coerce")
            v2 = pd.to_numeric(dfb[cols[2]], errors="coerce")
            v3 = pd.to_numeric(dfb[cols[3]], errors="coerce")
            v4 = pd.to_numeric(dfb[cols[4]], errors="coerce")

            u, v, w = janus_to_uvwt(v1, v2, v3, v4, theta_deg=theta)
            U_cells.append(u)
            V_cells.append(v)
            W_cells.append(w)

            beam_series_by_cell[1].append(v1)
            beam_series_by_cell[2].append(v2)
            beam_series_by_cell[3].append(v3)
            beam_series_by_cell[4].append(v4)

        # Stack -> mean per tijdstap
        U_mat = np.vstack([s.to_numpy() for s in U_cells])
        V_mat = np.vstack([s.to_numpy() for s in V_cells])
        W_mat = np.vstack([s.to_numpy() for s in W_cells])

        valid_u = np.any(np.isfinite(U_mat), axis=0)
        valid_v = np.any(np.isfinite(V_mat), axis=0)
        valid_w = np.any(np.isfinite(W_mat), axis=0)

        U_mean = np.full(U_mat.shape[1], np.nan)
        V_mean = np.full(V_mat.shape[1], np.nan)
        W_mean = np.full(W_mat.shape[1], np.nan)
        U_min = np.full(U_mat.shape[1], np.nan)
        V_min = np.full(V_mat.shape[1], np.nan)
        W_min = np.full(W_mat.shape[1], np.nan)
        U_max = np.full(U_mat.shape[1], np.nan)
        V_max = np.full(V_mat.shape[1], np.nan)
        W_max = np.full(W_mat.shape[1], np.nan)

        U_mean[valid_u] = np.nanmean(U_mat[:, valid_u], axis=0)
        V_mean[valid_v] = np.nanmean(V_mat[:, valid_v], axis=0)
        W_mean[valid_w] = np.nanmean(W_mat[:, valid_w], axis=0)
        U_min[valid_u] = np.nanmin(U_mat[:, valid_u], axis=0)
        V_min[valid_v] = np.nanmin(V_mat[:, valid_v], axis=0)
        W_min[valid_w] = np.nanmin(W_mat[:, valid_w], axis=0)
        U_max[valid_u] = np.nanmax(U_mat[:, valid_u], axis=0)
        V_max[valid_v] = np.nanmax(V_mat[:, valid_v], axis=0)
        W_max[valid_w] = np.nanmax(W_mat[:, valid_w], axis=0)

        bad_u = np.where((U_mean < U_min) | (U_mean > U_max))[0]
        bad_v = np.where((V_mean < V_min) | (V_mean > V_max))[0]
        bad_w = np.where((W_mean < W_min) | (W_mean > W_max))[0]

        warn = []
        if bad_u.size > 0: warn.append(f"U-mean buiten min/max op {bad_u.size} punten (mag niet) → mapping/units fout.")
        if bad_v.size > 0: warn.append(f"V-mean buiten min/max op {bad_v.size} punten (mag niet) → mapping/units fout.")
        if bad_w.size > 0: warn.append(f"W-mean buiten min/max op {bad_w.size} punten (mag niet) → mapping/units fout.")

        # Rotate to ENU if heading exists
        if heading is not None and heading.notna().sum() > 0:
            E, N = rotate_uv_to_EN(pd.Series(U_mean), pd.Series(V_mean), heading, convention=conv)
        else:
            E, N = pd.Series(U_mean), pd.Series(V_mean)

        Up = pd.Series(W_mean)

        # Horizontal speed/dir
        sp, dr = speed_dir_from_EN(E, N)

        # Rolling means
        E_ma = rolling_mean(E, ma_n); N_ma = rolling_mean(N, ma_n); Up_ma = rolling_mean(Up, ma_n)
        sp_ma = rolling_mean(sp, ma_n); dr_ma = rolling_mean(dr, ma_n)
        press_ma = rolling_mean(pressure, ma_n) if pressure is not None else None

        # Extra sanity: magnitude order
        if np.nanmax(np.abs(sp_ma)) > 5.0:
            warn.append("Snelheden >5 m/s gedetecteerd → waarschijnlijk units probleem (mm/s?) of fout in beam angle/conventie.")

        # Tidal expectation check (ruw): direction spreiding
        if np.isfinite(dr_ma).sum() > 100:
            # direction clustering: als bijna alles in 1 sector valt, is dat verdacht
            centers, freq_pct, _ = rose_bins(dr_ma.to_numpy(), sp_ma.to_numpy(), DEFAULT_ROSE_BINS)
            if np.nanmax(freq_pct) > 70:
                warn.append("Rose toont >70% in één sector → waarschijnlijk richtingberekening/heading probleem (of stroming echt unidirectioneel).")

        self.msg_var.set(" | ".join(warn) if warn else "Sanity: OK (mean binnen min/max per tijdstap; waarden plausibel).")

        self.last_export_meta = {
            "ma": ma_n,
            "theta_deg": theta,
            "heading_conv": conv,
            "iq_shift_s": float(self.iq_shift_seconds_var.get()),
            "iq_stretch": float(self.iq_stretch_var.get()),
            "iq_anchor": anchor_used,
            "cells": cells_sel.copy(),
            "beams": beams_sel.copy(),
            "time_from": t0,
            "time_to": t1,
            "pressure_available": pressure is not None and pressure.notna().any(),
            "heading_available": heading is not None and heading.notna().any(),
            "aquadopp_loaded": self.aqd_meta is not None,
            "warnings": "",
        }

        # Plot beams (mean per cell over tijd, MA) + pressure subplot
        for b in BEAMS:
            fig, canvas = self.figures[f"beam{b}"]
            fig.clf()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)

            # plot each selected cell beam series (MA)
            for i, c in enumerate(cells_sel):
                s = beam_series_by_cell[b][i]
                ax1.plot(t, rolling_mean(s, ma_n), linewidth=0.8, label=f"Cell {c}")

            # average over cells for this beam (at each time)
            B_mat = np.vstack([s.to_numpy() for s in beam_series_by_cell[b]])
            valid_b = np.any(np.isfinite(B_mat), axis=0)
            B_mean = np.full(B_mat.shape[1], np.nan)
            if np.any(valid_b):
                B_mean[valid_b] = np.nanmean(B_mat[:, valid_b], axis=0)
            beam_mean_series = pd.Series(B_mean, index=t.index)
            beam_mean_ma = rolling_mean(beam_mean_series, ma_n)
            beam_mean_ma_results[b] = beam_mean_ma
            ax1.plot(t, beam_mean_ma, linewidth=2.2, label="Gemiddelde cellen")

            ax1.set_title(f"Beam {b} velocity (m/s) – MA={ma_n}")
            ax1.set_ylabel("m/s")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right", fontsize=8, ncols=2)

            # pressure
            if press_ma is not None:
                ax2.plot(t, press_ma, linewidth=1.2)
                ax2.set_ylabel("Pressure/Depth")
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.01, 0.8, "Pressure niet gevonden", transform=ax2.transAxes)
            ax2.set_xlabel("Tijd")

            canvas.draw()

        # ENU plots
        self._plot_timeseries("E", t, E_ma, "East (m/s)", press_ma)
        self._plot_timeseries("N", t, N_ma, "North (m/s)", press_ma)
        self._plot_timeseries("W", t, Up_ma, "Up (m/s)", press_ma)
        self._plot_timeseries("speed", t, sp_ma, "Speed_hor (m/s)", press_ma)
        self._plot_timeseries("dir", t, dr_ma, "Direction (deg)", press_ma, is_dir=True)

        # Rose plot (freq% + mean speed m/s)
        fig, canvas = self.figures["rose"]
        fig.clf()
        axf = fig.add_subplot(211, projection="polar")
        axs = fig.add_subplot(212, projection="polar")

        d_arr = dr_ma.to_numpy(dtype=float)
        s_arr = sp_ma.to_numpy(dtype=float)
        msk = np.isfinite(d_arr) & np.isfinite(s_arr)
        d_arr = d_arr[msk]; s_arr = s_arr[msk]

        if d_arr.size > 10:
            centers, freq_pct, mean_sp = rose_bins(d_arr, s_arr, DEFAULT_ROSE_BINS)
            width = 2*np.pi/DEFAULT_ROSE_BINS

            axf.bar(centers, freq_pct, width=width, align="center", alpha=0.45)
            axf.set_title(f"Frequentie per sector (%) – {DEFAULT_ROSE_BINS} sectoren")
            axf.set_theta_zero_location("N"); axf.set_theta_direction(-1)

            mean_sp2 = mean_sp.copy()
            mean_sp2[np.isnan(mean_sp2)] = 0.0
            axs.bar(centers, mean_sp2, width=width, align="center", alpha=0.45)
            axs.set_title("Gemiddelde snelheid per sector (m/s)")
            axs.set_theta_zero_location("N"); axs.set_theta_direction(-1)

        else:
            axf.set_title("Te weinig geldige punten voor rose.")

        canvas.draw()
        aqd_summary = self._compute_aquadopp_summary(ma_n, t0, t1)
        self._plot_compare_if_available(
            iq_time=t,
            iq_E=E_ma,
            iq_N=N_ma,
            iq_speed=sp_ma,
            iq_dir=dr_ma,
            aqd_summary=aqd_summary,
        )
        compare_msg = self._east_north_compare_text(t, E_ma, N_ma, aqd_summary)
        if compare_msg:
            self.msg_var.set(f"{self.msg_var.get()} | {compare_msg}")
        self.last_export_meta["warnings"] = self.msg_var.get()

        self.last_processed = self._build_export_df(
            t,
            t_raw,
            beam_mean_ma_results,
            E_ma,
            N_ma,
            Up_ma,
            sp_ma,
            dr_ma,
            pressure,
            press_ma,
            heading,
            aqd_summary,
        )

    def _east_north_compare_text(
        self,
        iq_time: pd.Series,
        iq_E: pd.Series,
        iq_N: pd.Series,
        aqd_summary: Optional[pd.DataFrame],
    ) -> str:
        if aqd_summary is None or aqd_summary.empty:
            return "E/N vergelijking: Aquadopp niet beschikbaar."

        iq_df = pd.DataFrame({
            "DateTime": pd.to_datetime(iq_time, errors="coerce"),
            "IQ_E": pd.Series(iq_E),
            "IQ_N": pd.Series(iq_N),
        }).dropna(subset=["DateTime"]).sort_values("DateTime")
        aqd_df = aqd_summary[["DateTime", "AQD_E_MA", "AQD_N_MA"]].dropna(subset=["DateTime"]).sort_values("DateTime")

        merge = pd.merge_asof(
            iq_df,
            aqd_df,
            on="DateTime",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=max(1.0, ASOF_TOL_SECONDS * 5.0)),
        )
        merge = merge.dropna(subset=["IQ_E", "IQ_N", "AQD_E_MA", "AQD_N_MA"])
        if merge.empty:
            return "E/N vergelijking: geen overlappende punten binnen tolerantieraam."

        dE = merge["IQ_E"] - merge["AQD_E_MA"]
        dN = merge["IQ_N"] - merge["AQD_N_MA"]
        bias_E = float(np.nanmean(dE))
        bias_N = float(np.nanmean(dN))
        rmse_E = float(np.sqrt(np.nanmean(np.square(dE))))
        rmse_N = float(np.sqrt(np.nanmean(np.square(dN))))
        n = int(len(merge))
        return (
            f"E/N vergelijking (n={n}): "
            f"bias_E={bias_E:.3f} m/s, RMSE_E={rmse_E:.3f} m/s; "
            f"bias_N={bias_N:.3f} m/s, RMSE_N={rmse_N:.3f} m/s"
        )

    def _compute_aquadopp_summary(self, ma_n: int, t0: Optional[pd.Timestamp], t1: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
        if self.aqd_meta is None or self.aqd_time_index is None:
            return None
        if self.aqd_v1 is None or self.aqd_v2 is None or self.aqd_v3 is None:
            return None
        cells = [c for c, v in self.aqd_cell_vars.items() if v.get()]
        if not cells:
            return None

        mask = np.ones(len(self.aqd_time_index), dtype=bool)
        if t0 is not None:
            mask &= self.aqd_time_index >= t0
        if t1 is not None:
            mask &= self.aqd_time_index <= t1
        if not mask.any():
            return None

        v1 = self.aqd_v1[mask, :]
        v2 = self.aqd_v2[mask, :]
        v3 = self.aqd_v3[mask, :]
        t = pd.Series(self.aqd_time_index[mask]).reset_index(drop=True)

        x, y, _z = compute_xyz_from_beams(self.aqd_meta["T"], v1, v2, v3)
        x_ma = rolling_mean_array(x, ma_n)
        y_ma = rolling_mean_array(y, ma_n)

        E = pd.Series(np.nanmean(x_ma[:, cells], axis=1))
        N = pd.Series(np.nanmean(y_ma[:, cells], axis=1))
        speed, direction = speed_dir_from_EN(E, N)

        return pd.DataFrame({
            "DateTime": t,
            "AQD_E_MA": E.reset_index(drop=True),
            "AQD_N_MA": N.reset_index(drop=True),
            "AQD_Speed_MA": pd.Series(speed).reset_index(drop=True),
            "AQD_Direction_MA": pd.Series(direction).reset_index(drop=True),
        })

    def _plot_compare_if_available(
        self,
        iq_time: pd.Series,
        iq_E: pd.Series,
        iq_N: pd.Series,
        iq_speed: pd.Series,
        iq_dir: pd.Series,
        aqd_summary: Optional[pd.DataFrame],
    ) -> None:
        if aqd_summary is None or aqd_summary.empty:
            for key in ["cmp_speed", "cmp_dir", "cmp_E", "cmp_N"]:
                fig, canvas = self.figures[key]
                fig.clf()
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Aquadopp niet beschikbaar of geen geldige selectie.", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                canvas.draw()
            return

        self._plot_compare_timeseries("cmp_speed", iq_time, iq_speed, aqd_summary["DateTime"], aqd_summary["AQD_Speed_MA"], "Speed (m/s)")
        self._plot_compare_timeseries("cmp_E", iq_time, iq_E, aqd_summary["DateTime"], aqd_summary["AQD_E_MA"], "East (m/s)")
        self._plot_compare_timeseries("cmp_N", iq_time, iq_N, aqd_summary["DateTime"], aqd_summary["AQD_N_MA"], "North (m/s)")
        self._plot_compare_timeseries(
            "cmp_dir",
            iq_time,
            iq_dir,
            aqd_summary["DateTime"],
            aqd_summary["AQD_Direction_MA"],
            "Direction (deg)",
            is_direction=True,
        )

    def _plot_compare_timeseries(
        self,
        key: str,
        tiq: pd.Series,
        yiq: pd.Series,
        taqd: pd.Series,
        yaqd: pd.Series,
        ylabel: str,
        is_direction: bool = False,
    ) -> None:
        fig, canvas = self.figures[key]
        fig.clf()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        tiq = pd.Series(pd.to_datetime(tiq, errors="coerce")).reset_index(drop=True)
        taqd = pd.Series(pd.to_datetime(taqd, errors="coerce")).reset_index(drop=True)
        yiq = pd.Series(yiq).reset_index(drop=True)
        yaqd = pd.Series(yaqd).reset_index(drop=True)

        ax1.plot(tiq, yiq, linewidth=1.7, label="IQ")
        ax1.plot(taqd, yaqd, linewidth=1.2, label="Aquadopp")
        ax1.set_title(f"Vergelijking {ylabel}")
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        if is_direction:
            ax1.set_ylim(0, 360)

        merge = pd.merge_asof(
            pd.DataFrame({"DateTime": tiq, "IQ": yiq}).dropna().sort_values("DateTime"),
            pd.DataFrame({"DateTime": taqd, "AQD": yaqd}).dropna().sort_values("DateTime"),
            on="DateTime",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=max(1.0, ASOF_TOL_SECONDS * 5.0)),
        )
        merge = merge.dropna(subset=["IQ", "AQD"])
        if not merge.empty:
            if is_direction:
                diff = circular_diff_deg(merge["IQ"], merge["AQD"])
                ax2.set_ylabel("Delta dir (deg)")
                ax2.set_ylim(-180, 180)
            else:
                diff = merge["IQ"] - merge["AQD"]
                ax2.set_ylabel("Delta (IQ-AQD)")
            ax2.plot(merge["DateTime"], diff, linewidth=1.2)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.02, 0.8, "Geen overlap binnen tolerantie.", transform=ax2.transAxes)
        ax2.set_xlabel("Tijd")
        canvas.draw()

    def _plot_timeseries(self, key, t, y, ylabel, press_ma, is_dir: bool=False):
        fig, canvas = self.figures[key]
        fig.clf()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        ax1.plot(t, y, linewidth=1.8)
        ax1.set_title(f"{ylabel} – MA={int(self.ma_var.get())}")
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)
        if is_dir:
            ax1.set_ylim(0, 360)

        if press_ma is not None:
            ax2.plot(t, press_ma, linewidth=1.2)
            ax2.set_ylabel("Pressure/Depth")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.01, 0.8, "Pressure niet gevonden", transform=ax2.transAxes)
        ax2.set_xlabel("Tijd")
        canvas.draw()

    def _build_export_df(
        self,
        t: pd.Series,
        t_raw: pd.Series,
        beam_mean_ma: Dict[int, pd.Series],
        E_ma: pd.Series,
        N_ma: pd.Series,
        Up_ma: pd.Series,
        sp_ma: pd.Series,
        dr_ma: pd.Series,
        pressure: Optional[pd.Series],
        press_ma: Optional[pd.Series],
        heading: Optional[pd.Series],
        aqd_summary: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        def normalize(series: Optional[pd.Series]) -> Optional[pd.Series]:
            if series is None:
                return None
            return pd.Series(series).reset_index(drop=True)

        df = pd.DataFrame({
            self.time_col_vb: t_raw.reset_index(drop=True),
            f"{self.time_col_vb}_IQ_adjusted": t.reset_index(drop=True),
        })
        helpers = {
            "E_MA": E_ma,
            "N_MA": N_ma,
            "Up_MA": Up_ma,
            "Speed_MA": sp_ma,
            "Direction_MA": dr_ma,
        }
        for name, series in helpers.items():
            ser = normalize(series)
            if ser is not None:
                df[name] = ser

        for b, series in beam_mean_ma.items():
            ser = normalize(series)
            if ser is not None:
                df[f"Beam{b}_MA"] = ser

        if heading is not None:
            ser = normalize(heading)
            if ser is not None:
                df["Heading"] = ser

        if pressure is not None:
            ser = normalize(pressure)
            if ser is not None:
                name = self.press_col or "Pressure"
                df[name] = ser
        if press_ma is not None:
            ser = normalize(press_ma)
            if ser is not None:
                name = (self.press_col or "Pressure") + "_MA"
                df[name] = ser

        if aqd_summary is not None and not aqd_summary.empty:
            merged = pd.merge_asof(
                df[[f"{self.time_col_vb}_IQ_adjusted"]].rename(columns={f"{self.time_col_vb}_IQ_adjusted": "DateTime"}).sort_values("DateTime"),
                aqd_summary.sort_values("DateTime"),
                on="DateTime",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=max(1.0, ASOF_TOL_SECONDS * 5.0)),
            )
            merged = merged.rename(columns={"DateTime": "DateTime_AQD_match"})
            for col in ["AQD_E_MA", "AQD_N_MA", "AQD_Speed_MA", "AQD_Direction_MA"]:
                if col in merged.columns:
                    df[col] = merged[col].reset_index(drop=True)
            if "AQD_E_MA" in df.columns:
                df["Delta_E_IQ_minus_AQD"] = df["E_MA"] - df["AQD_E_MA"]
            if "AQD_N_MA" in df.columns:
                df["Delta_N_IQ_minus_AQD"] = df["N_MA"] - df["AQD_N_MA"]
            if "AQD_Speed_MA" in df.columns:
                df["Delta_Speed_IQ_minus_AQD"] = df["Speed_MA"] - df["AQD_Speed_MA"]
            if "AQD_Direction_MA" in df.columns:
                df["Delta_Direction_IQ_minus_AQD"] = circular_diff_deg(df["Direction_MA"], df["AQD_Direction_MA"])

        return df

    def export_processed_data(self):
        if self.last_processed is None:
            messagebox.showwarning("Export", "Herbereken de data eerst voordat je exporteert.")
            return
        default_name = f"iq_processed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            title="Bewaar export",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not path:
            return
        sep = "\t" if path.lower().endswith(".tsv") else ","
        self.last_processed.to_csv(path, index=False, sep=sep)
        messagebox.showinfo("Export", f"Export geschreven naar:\n{path}")

    def _format_time_for_report(self, ts: Optional[pd.Timestamp]) -> str:
        if ts is None or not pd.notna(ts):
            return "—"
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def export_to_word_report(self):
        if Document is None:
            messagebox.showerror(
                "Word export",
                "python-docx is niet geïnstalleerd. Installeer het (pip install python-docx) en probeer opnieuw."
            )
            return
        if self.last_export_meta is None or self.last_processed is None:
            messagebox.showwarning("Word export", "Herbereken eerst zodat de figuren en instellingen beschikbaar zijn.")
            return

        path = filedialog.asksaveasfilename(
            title="Bewaar Wordrapport (.docx)",
            defaultextension=".docx",
            filetypes=[("Word document", "*.docx"), ("All files", "*.*")],
            initialfile=f"iq_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.docx",
        )
        if not path:
            return

        meta = self.last_export_meta
        doc = Document()
        doc.add_heading("SonTek IQ – Verwerkingsrapport", level=1)

        doc.add_heading("Instellingen", level=2)
        table = doc.add_table(rows=1, cols=2)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Instelling"
        hdr_cells[1].text = "Waarde"

        def add_row(label: str, value: str) -> None:
            row_cells = table.add_row().cells
            row_cells[0].text = label
            row_cells[1].text = value

        add_row("MA (N samples)", str(meta["ma"]))
        add_row("Beam angle θ (deg)", str(meta["theta_deg"]))
        add_row("Heading-conventie", str(meta["heading_conv"]))
        add_row("IQ tijdshift (s)", str(meta.get("iq_shift_s", 0.0)))
        add_row("IQ stretch factor", str(meta.get("iq_stretch", 1.0)))
        add_row("IQ ankerpunt", self._format_time_for_report(meta.get("iq_anchor")))
        add_row("Geselecteerde cellen", ", ".join(f"Cell {c}" for c in meta["cells"]))
        add_row("Actieve beams", ", ".join(str(b) for b in meta["beams"]))
        add_row("Tijdsvak (start → einde)", f"{self._format_time_for_report(meta['time_from'])} → {self._format_time_for_report(meta['time_to'])}")
        add_row("Pressure opgenomen in MAIN", "Ja" if meta["pressure_available"] else "Nee")
        add_row("Heading beschikbaar", "Ja" if meta["heading_available"] else "Nee")
        add_row("Aquadopp geladen", "Ja" if meta.get("aquadopp_loaded") else "Nee")

        doc.add_paragraph(f"Meldingen/warnings: {meta.get('warnings', 'Geen')}")

        doc.add_heading("Figuurtabs", level=2)
        img_width = Inches(6) if Inches is not None else None
        for key, title in self.tab_titles.items():
            fig, _ = self.figures[key]
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            doc.add_heading(title, level=3)
            if img_width is not None:
                doc.add_picture(buf, width=img_width)
            else:
                doc.add_picture(buf)
            doc.add_paragraph(f"Fonte: figuur '{title}' uit de GUI.")

        doc.save(path)
        messagebox.showinfo("Word export", f"Rapport weggeschreven naar:\n{path}")


if __name__ == "__main__":
    app = IQApp()
    app.mainloop()
