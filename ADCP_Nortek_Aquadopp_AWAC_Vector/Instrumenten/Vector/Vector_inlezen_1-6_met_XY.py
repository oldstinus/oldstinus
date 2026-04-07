#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nortek Vector (.dat + .sen + .hdr) • Visualizer met correcte f Hz tijdas + X–Y scatter
- .sen = 1 Hz tijdstempels
- .dat = f Hz samples (f uit .hdr; anders geschat uit rijenratio)
- .dat-tijdas: exact t = t0 + i*(1/f) s (geen drift)
- 2-cijferige jaren fix in .sen
- (Optioneel) .hdr: druk unit/scale/offset; snelheidsunit heuristisch (mm/s→m/s)
- GUI: Tijd/Index, Y & kleur, smoothing, slice-export, figuur-save, X–Y scatter
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, cast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # <-- voor type hints

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.dates import DateFormatter, AutoDateLocator

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("VectorOnly_fHz")

# ---------------- Kolomnamen ----------------
DAT_COLS = [
    'Burst_counter', 'Ensemble_counter',
    'Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3',
    'Amplitude_Beam1', 'Amplitude_Beam2', 'Amplitude_Beam3',
    'SNR_Beam1', 'SNR_Beam2', 'SNR_Beam3',
    'Correlation_Beam1', 'Correlation_Beam2', 'Correlation_Beam3',
    'Pressure', 'Analog_input1', 'Analog_input2', 'Checksum'
]

SEN_COLS = [
    'Month','Day','Year','Hour','Minute','Second',
    'Error_code','Status_code','Battery_voltage',
    'Soundspeed','Heading','Pitch','Roll','Temperature',
    'Analog_input','Checksum'
]

# ---------------- .hdr helpers ----------------
SAMPLING_PAT = re.compile(r"sampling\s*[-_:]?\s*rate.*?(\d+(?:\.\d+)?)\s*hz", re.I)

def read_hdr_lines(hdr_path: str | Path) -> List[str]:
    p = Path(hdr_path)
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()

def parse_sampling_rate(hdr_lines: List[str]) -> Optional[float]:
    for ln in hdr_lines:
        m = SAMPLING_PAT.search(ln)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None

def parse_hdr_units_meta(hdr_lines: List[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    text = "\n".join(hdr_lines)

    # Pressure units
    if re.search(r"\bpressure\b.*\bdbar\b", text, re.I):
        meta["pressure_unit"] = "dbar"
    elif re.search(r"\bpressure\b.*\bkpa\b", text, re.I):
        meta["pressure_unit"] = "kPa"
    elif re.search(r"\bpressure\b.*(?<!k)\bpa\b", text, re.I):
        meta["pressure_unit"] = "Pa"
    elif re.search(r"\bpressure\b.*\bm\b", text, re.I):
        meta["pressure_unit"] = "m"

    # scale/offset regels
    for ln in hdr_lines:
        km = re.match(r"\s*pressure\s+(scale|offset)\s+([-\d\.eE]+)", ln, re.I)
        if km:
            key = km.group(1).lower()
            try:
                val = float(km.group(2))
                meta[f"pressure_{key}"] = val
            except Exception:
                pass

    # Eventuele klok-offset
    for ln in hdr_lines:
        if re.search(r"(clock|time).*(offset|drift)", ln, re.I):
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", ln)
            if nums:
                try:
                    meta["clock_offset_sec"] = float(nums[0])
                except Exception:
                    pass

    return meta

# ---------------- Data helpers ----------------
def _read_table(path: str | Path, names: List[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=names,
        comment="#", engine="python", dtype="float64",
        na_values=["NaN", "nan", "INF", "-INF"]
    )
    for col in ("Burst_counter","Ensemble_counter","Checksum"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def _build_datetime_from_sen(df_sen: pd.DataFrame) -> pd.Series:
    """
    Type-safe: bouw DataFrame → pd.to_datetime (vermijdt Pylance overload meldingen).
    """
    year = df_sen["Year"].astype("int64")
    year = np.where(year < 100, np.where(year < 80, 2000 + year, 1900 + year), year)

    dt_input = pd.DataFrame({
        "year":   pd.Series(year, dtype="int64"),
        "month":  df_sen["Month"].astype("int64"),
        "day":    df_sen["Day"].astype("int64"),
        "hour":   df_sen["Hour"].astype("int64"),
        "minute": df_sen["Minute"].astype("int64"),
        "second": df_sen["Second"].astype("int64"),
    })
    dt = pd.to_datetime(dt_input, errors="coerce")
    return dt

def _velocity_unit_heuristic(df: pd.DataFrame) -> Tuple[str, Dict[str,float]]:
    vcols = [c for c in df.columns if c.startswith("Velocity_Beam")]
    applied = {'scale': 1.0, 'offset': 0.0}
    if not vcols:
        return "unknown", applied
    vmax = pd.concat([df[c].abs() for c in vcols], axis=1).max(axis=1).quantile(0.95)
    if pd.isna(vmax):
        return "unknown", applied
    if vmax > 5.0:
        df[vcols] = df[vcols] / 1000.0
        applied['scale'] = 1/1000.0
        return "m/s (converted from mm/s)", applied
    return "m/s", applied

def _apply_pressure_meta(series: pd.Series, meta: Dict[str, Any]) -> Tuple[pd.Series, str, Dict[str,float]]:
    unit = meta.get("pressure_unit", "raw")
    applied = {'scale': 1.0, 'offset': 0.0}
    s = series.copy()
    if "pressure_scale" in meta:
        s = s * float(meta["pressure_scale"])
        applied['scale'] *= float(meta["pressure_scale"])
    if "pressure_offset" in meta:
        s = s + float(meta["pressure_offset"])
        applied['offset'] += float(meta["pressure_offset"])
    return s, unit, applied

def _build_dat_time_index_from_sen(dt_sen: pd.Series, n_dat: int, fs_hz: float) -> pd.DatetimeIndex:
    if len(dt_sen) == 0 or pd.isna(dt_sen.iloc[0]):
        raise ValueError("Geen geldige .sen timestamps om tijdas te bouwen.")
    t0 = pd.to_datetime(dt_sen.iloc[0])
    offs = pd.to_timedelta(np.arange(n_dat, dtype=np.float64) / float(fs_hz), unit="s")
    return pd.DatetimeIndex(t0 + offs)

def load_vector_data(dat_file: str | Path, sen_file: str | Path,
                     hdr_file: Optional[str | Path] = None,
                     use_checksum_filter: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"fs_hz": None, "velocity_unit": None, "velocity_applied": {},
                            "pressure_unit": None, "pressure_applied": {}, "hdr_path": None}

    dat_file = Path(dat_file); sen_file = Path(sen_file)
    if not dat_file.exists() or not sen_file.exists():
        raise FileNotFoundError("Bestand niet gevonden (controleer paden).")

    # HDR
    hdr_lines: List[str] = []
    if hdr_file and Path(hdr_file).exists():
        hdr_lines = read_hdr_lines(hdr_file)
        info["hdr_path"] = str(hdr_file)
    elif dat_file.with_suffix(".hdr").exists():
        hdr_file = dat_file.with_suffix(".hdr")
        hdr_lines = read_hdr_lines(hdr_file)
        info["hdr_path"] = str(hdr_file)

    fs_hz = parse_sampling_rate(hdr_lines) if hdr_lines else None
    info["fs_hz"] = fs_hz

    # DAT/SEN
    logger.info(f"Lezen DAT: {dat_file}")
    df_dat = _read_table(dat_file, DAT_COLS)
    logger.info(f"  DAT rijen: {len(df_dat)}")

    logger.info(f"Lezen SEN: {sen_file}")
    df_sen = _read_table(sen_file, SEN_COLS)
    logger.info(f"  SEN rijen: {len(df_sen)} (~1 Hz verwacht)")

    if len(df_dat) == 0 or len(df_sen) == 0:
        raise ValueError("Lege .dat of .sen; controleer bestanden.")

    dt_sen = _build_datetime_from_sen(df_sen)
    if dt_sen.isna().all():
        raise ValueError("Kon geen geldige datums opbouwen uit .sen.")

    # Sampling rate schatten indien afwezig
    if fs_hz is None:
        fs_est = max(1, int(round(len(df_dat) / max(1, len(df_sen)))))
        fs_hz = float(fs_est)
        info["fs_hz"] = fs_hz
        logger.info(f"Sampling rate geschat op {fs_hz} Hz uit rijenratio.")

    # Checksum-filter
    if use_checksum_filter and "Checksum" in df_dat.columns:
        before = len(df_dat)
        df_dat = df_dat[df_dat["Checksum"].fillna(0) == 0]
        after = len(df_dat)
        logger.info(f"  Checksum-filter: {after}/{before} rijen behouden")
        if after == 0:
            raise ValueError("Na checksum-filter blijven geen rijen over.")

    # Tijdindex: exact f Hz vanaf eerste .sen tijd
    idx_dat = _build_dat_time_index_from_sen(dt_sen, len(df_dat), fs_hz)
    df_dat.insert(0, "Datetime", idx_dat)  # type-safe
    df_dat = df_dat.set_index("Datetime").sort_index()

    # Snelheidsunit (heuristiek mm/s → m/s)
    vunit, vapplied = _velocity_unit_heuristic(df_dat)
    info["velocity_unit"] = vunit
    info["velocity_applied"] = vapplied

    # Drukconversie uit .hdr (indien aanwezig)
    if "Pressure" in df_dat.columns:
        if hdr_lines:
            meta = parse_hdr_units_meta(hdr_lines)
            s, punit, papplied = _apply_pressure_meta(df_dat["Pressure"], meta)
            df_dat["Pressure"] = s
            info["pressure_unit"] = punit
            info["pressure_applied"] = papplied
        else:
            info["pressure_unit"] = "raw"

    # Resultant Speed (beam-resultant; geen ENU-rotatie)
    req = ["Velocity_Beam1","Velocity_Beam2","Velocity_Beam3"]
    if all(c in df_dat.columns for c in req):
        df_dat["Resultant_Speed"] = np.sqrt(
            df_dat["Velocity_Beam1"]**2 +
            df_dat["Velocity_Beam2"]**2 +
            df_dat["Velocity_Beam3"]**2
        )

    return df_dat, info

# ---------------- GUI ----------------
class VectorOnlyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nortek Vector • Reader & Plotter (f Hz tijdas + X–Y scatter)")
        self.geometry("1020x640")
        self.minsize(980, 620)

        self.df: Optional[pd.DataFrame] = None
        self._last_dir: Path = Path.home()
        self._smoothing_win = tk.IntVar(value=1)
        self._use_checksum = tk.BooleanVar(value=True)
        self._info: Dict[str, Any] = {}

        outer = ttk.Frame(self, padding=8); outer.pack(fill="both", expand=True)

        # Bovenste rij
        top = ttk.Frame(outer); top.pack(fill="x", pady=(0,8))
        ttk.Button(top, text="Laad .dat + .sen (+ .hdr)", command=self.cmd_load).pack(side="left")
        ttk.Checkbutton(top, text="Checksum-filter", variable=self._use_checksum).pack(side="left", padx=10)
        ttk.Label(top, text="Smoothing (N):").pack(side="left", padx=(20,4))
        ttk.Entry(top, width=6, textvariable=self._smoothing_win).pack(side="left")
        ttk.Button(top, text="Export slice → CSV", command=self.export_slice_csv).pack(side="right")

        # Selecties: Tijd/Index plot + Colormap
        row1 = ttk.Frame(outer); row1.pack(fill="x", pady=(0,8))
        ttk.Label(row1, text="Tijd/Index plot — X-as:").grid(row=0, column=0, sticky="w")
        self.x_cb = ttk.Combobox(row1, state="readonly", width=22, values=["Time", "Sample_Index"])
        self.x_cb.current(0); self.x_cb.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(row1, text="Y-as:").grid(row=0, column=2, sticky="w")
        self.y_cb = ttk.Combobox(row1, state="readonly", width=32); self.y_cb.grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Label(row1, text="Kleur (colormap):").grid(row=0, column=4, sticky="w")
        self.c_cb = ttk.Combobox(row1, state="readonly", width=28); self.c_cb.grid(row=0, column=5, sticky="ew", padx=6)
        for c in (1,3,5): row1.columnconfigure(c, weight=1)

        # Tijdsvenster
        row2 = ttk.Frame(outer); row2.pack(fill="x", pady=(0,6))
        ttk.Label(row2, text="Start (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0, sticky="w")
        self.start_entry = ttk.Entry(row2); self.start_entry.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(row2, text="Einde (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=2, sticky="w")
        self.end_entry = ttk.Entry(row2); self.end_entry.grid(row=0, column=3, sticky="ew", padx=6)
        row2.columnconfigure(1, weight=1); row2.columnconfigure(3, weight=1)

        # Sliders
        ttk.Label(outer, text="Selecteer tijdsbereik (%)").pack(anchor="w")
        slf = ttk.Frame(outer); slf.pack(fill="x", pady=(0,8))
        self.sldr0 = tk.Scale(slf, from_=0, to=100, orient="horizontal",
                              label="Start %", command=self.on_slider_start)
        self.sldr1 = tk.Scale(slf, from_=0, to=100, orient="horizontal",
                              label="Einde %", command=self.on_slider_end)
        self.sldr1.set(100)
        self.sldr0.pack(side="left", fill="x", expand=True, padx=6)
        self.sldr1.pack(side="left", fill="x", expand=True, padx=6)

        # Acties tijd/colormap
        btnf = ttk.Frame(outer); btnf.pack(pady=4)
        ttk.Button(btnf, text="Plot Tijdreeks", command=self.plot_timeseries).pack(side="left", padx=6)
        ttk.Button(btnf, text="Plot Colormap (Y vs Tijd, kleur = var)", command=self.plot_colormap).pack(side="left", padx=6)
        ttk.Button(btnf, text="Sla figuur op (PNG)", command=self.save_current_figure).pack(side="left", padx=6)

        # --- X–Y scatter ---
        xy_frame = ttk.LabelFrame(outer, text="X–Y scatter (parameter vs parameter)")
        xy_frame.pack(fill="x", pady=(8,8))
        ttk.Label(xy_frame, text="X:").grid(row=0, column=0, sticky="w", padx=(6,2), pady=4)
        self.xy_x_cb = ttk.Combobox(xy_frame, state="readonly", width=30)
        self.xy_x_cb.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(xy_frame, text="Y:").grid(row=0, column=2, sticky="w", padx=(6,2))
        self.xy_y_cb = ttk.Combobox(xy_frame, state="readonly", width=30)
        self.xy_y_cb.grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Label(xy_frame, text="Kleur (optioneel):").grid(row=0, column=4, sticky="w", padx=(6,2))
        self.xy_c_cb = ttk.Combobox(xy_frame, state="readonly", width=28)
        self.xy_c_cb.grid(row=0, column=5, sticky="ew", padx=6)
        ttk.Button(xy_frame, text="Plot X–Y scatter", command=self.plot_xy_scatter).grid(row=0, column=6, padx=8)
        for c in (1,3,5):
            xy_frame.columnconfigure(c, weight=1)

        # Metadata
        meta_frame = ttk.LabelFrame(outer, text="Metadata (.hdr) & toegepaste instellingen")
        meta_frame.pack(fill="both", expand=True, pady=(0,0))
        self.meta_text = tk.Text(meta_frame, height=6, wrap="word")
        self.meta_text.pack(fill="both", expand=True)

        self._current_fig: Optional[Figure] = None  # <-- type-safe

    # ---- helpers ----
    def _initialdir(self) -> str:
        return str(self._last_dir if self._last_dir.exists() else Path.home())

    def _populate_param_boxes(self):
        df = self.df
        if df is None or df.empty:
            return
        cols = list(df.columns)
        # Voor tijd/colormap sectie
        self.y_cb["values"] = cols
        self.c_cb["values"] = cols
        default_y = "Resultant_Speed" if "Resultant_Speed" in cols else cols[0]
        default_c = "Pressure" if "Pressure" in cols else default_y
        self.y_cb.set(default_y); self.c_cb.set(default_c)
        # Voor X–Y scatter
        self.xy_x_cb["values"] = cols
        self.xy_y_cb["values"] = cols
        self.xy_c_cb["values"] = ["(geen kleur)"] + cols
        self.xy_x_cb.set("Resultant_Speed" if "Resultant_Speed" in cols else cols[0])
        self.xy_y_cb.set("Pressure" if "Pressure" in cols else cols[min(1, len(cols)-1)])
        self.xy_c_cb.set("(geen kleur)")

        # Tijdsveld init
        t0 = pd.to_datetime(df.index.min()).strftime("%Y-%m-%d %H:%M:%S")
        t1 = pd.to_datetime(df.index.max()).strftime("%Y-%m-%d %H:%M:%S")
        self.start_entry.delete(0,"end"); self.start_entry.insert(0, t0)
        self.end_entry.delete(0,"end");   self.end_entry.insert(0, t1)
        self.sldr0.set(0); self.sldr1.set(100)

    def _update_meta_panel(self):
        self.meta_text.delete("1.0","end")
        lines: List[str] = []
        if self._info.get("hdr_path"): lines.append(f"HDR: {self._info['hdr_path']}")
        if self._info.get("fs_hz") is not None: lines.append(f"Sampling rate: {self._info['fs_hz']} Hz")
        vu = self._info.get("velocity_unit")
        if vu: lines.append(f"Snelheidsunit: {vu} (heur.)")
        va = self._info.get("velocity_applied", {})
        if va: lines.append(f"Snelheid scale={va.get('scale',1.0)}, offset={va.get('offset',0.0)}")
        pu = self._info.get("pressure_unit")
        if pu: lines.append(f"Drukunits: {pu}")
        pa = self._info.get("pressure_applied", {})
        if pa: lines.append(f"Druk scale={pa.get('scale',1.0)}, offset={pa.get('offset',0.0)}")
        self.meta_text.insert("1.0", "\n".join(lines))

    def get_slice(self) -> pd.DataFrame:
        df = self.df
        if df is None or df.empty:
            messagebox.showerror("Geen data", "Laad eerst je data.")
            return pd.DataFrame()
        try:
            s = pd.to_datetime(self.start_entry.get())
            e = pd.to_datetime(self.end_entry.get())
            if e < s: raise ValueError("Einde ligt voor start.")
            df2 = df.loc[s:e]
            win = int(self._smoothing_win.get())
            if win > 1:
                df2 = df2.rolling(win, min_periods=1, center=True).mean()
            return df2
        except Exception as ex:
            messagebox.showerror("Slice fout", str(ex))
            return pd.DataFrame()

    # ---- events ----
    def cmd_load(self):
        datf = filedialog.askopenfilename(
            title="Kies .dat bestand", filetypes=[("DAT","*.dat"),("Alle bestanden","*.*")],
            initialdir=self._initialdir()
        )
        if not datf: return
        dat_path = Path(datf); self._last_dir = dat_path.parent

        senf = filedialog.askopenfilename(
            title="Kies .sen bestand", filetypes=[("SEN","*.sen"),("Alle bestanden","*.*")],
            initialdir=str(self._last_dir)
        )
        if not senf: return

        hdr_path = dat_path.with_suffix(".hdr")
        if not hdr_path.exists():
            hdr_opt = filedialog.askopenfilename(
                title="(Optioneel) Kies .hdr bestand", filetypes=[("HDR","*.hdr"),("Alle bestanden","*.*")],
                initialdir=str(self._last_dir)
            )
            hdr_path = Path(hdr_opt) if hdr_opt else None

        try:
            df, info = load_vector_data(
                dat_path, Path(senf), hdr_file=hdr_path, use_checksum_filter=self._use_checksum.get()
            )
        except Exception as e:
            messagebox.showerror("Fout bij laden", str(e))
            logger.exception("Laden mislukt"); return

        # Bewaar
        self.df = df.copy()
        # Sample index (alternatieve X-as)
        self.df["Sample_Index"] = np.arange(len(self.df), dtype=float)
        self._info = info

        self._populate_param_boxes()
        self._update_meta_panel()
        messagebox.showinfo("OK", f"Dataset geladen: {len(self.df)} rijen")

    def on_slider_start(self, pct: str):
        df = self.df
        if df is None or df.empty: return
        idx = int(float(pct)/100.0 * (len(df.index)-1))
        idx = max(0, min(idx, len(df.index)-1))
        t = pd.to_datetime(df.index[idx]).strftime("%Y-%m-%d %H:%M:%S")
        self.start_entry.delete(0,"end"); self.start_entry.insert(0, t)

    def on_slider_end(self, pct: str):
        df = self.df
        if df is None or df.empty: return
        idx = int(float(pct)/100.0 * (len(df.index)-1))
        idx = max(0, min(idx, len(df.index)-1))
        t = pd.to_datetime(df.index[idx]).strftime("%Y-%m-%d %H:%M:%S")
        self.end_entry.delete(0,"end"); self.end_entry.insert(0, t)

    # ---- plotting: Tijd/Colormap ----
    def _x_values(self, df2: pd.DataFrame) -> Tuple[np.ndarray, str]:
        if self.x_cb.get() == "Sample_Index":
            return df2["Sample_Index"].to_numpy(dtype=float), "Sample index"
        # Type-safe: datetime64 → numpy array van datetime64[ns]
        return df2.index.to_numpy(dtype="datetime64[ns]"), "Tijd"


    def plot_timeseries(self):
        df2 = self.get_slice()
        if df2.empty: return
        ycol = self.y_cb.get()
        if ycol not in df2.columns: ...
        x, xlabel = self._x_values(df2)
        y = pd.to_numeric(df2[ycol], errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots()
        ax.plot(x, y)  # x = ndarray[datetime64], y = ndarray[float]
        ax.set_title(f"Tijdreeks: {ycol}")
        ax.set_xlabel(xlabel); ax.set_ylabel(ycol)
        if xlabel == "Tijd":
            ax.xaxis.set_major_locator(AutoDateLocator())
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d-%m"))
            fig.autofmt_xdate()
        self._current_fig = fig
        plt.show()

    def plot_colormap(self):
        df2 = self.get_slice()
        if df2.empty: return
        ycol = self.y_cb.get(); ccol = self.c_cb.get()
        for c in (ycol, ccol):
            if c not in df2.columns:
                messagebox.showerror("Kolom ontbreekt", f"Kolom '{c}' bestaat niet."); return
        x, xlabel = self._x_values(df2)
        y = pd.to_numeric(df2[ycol], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(df2[ccol], errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots()
        sc = ax.scatter(x, y, c=c, marker="s", s=10, cmap="viridis")
        ax.set_title(f"Colormap: {ccol} (kleur) vs {ycol}")
        ax.set_xlabel(xlabel); ax.set_ylabel(ycol)
        if xlabel == "Tijd":
            ax.xaxis.set_major_locator(AutoDateLocator())
            ax.xaxis.set_major_formatter(DateFormatter("%H:%M\n%d-%m"))
            fig.autofmt_xdate()
        fig.colorbar(sc, ax=ax, label=ccol)
        self._current_fig = fig
        plt.show()

    # ---- X–Y scatter ----
    def plot_xy_scatter(self):
        df2 = self.get_slice()
        if df2.empty: return
        xcol = self.xy_x_cb.get()
        ycol = self.xy_y_cb.get()
        csel = self.xy_c_cb.get()
        if xcol not in df2.columns or ycol not in df2.columns:
            messagebox.showerror("Kolom ontbreekt", "Kies geldige X en Y kolommen."); return

        x = pd.to_numeric(df2[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df2[ycol], errors="coerce").to_numpy(dtype=float)

        fig, ax = plt.subplots()
        if csel and csel != "(geen kleur)" and csel in df2.columns:
            c = pd.to_numeric(df2[csel], errors="coerce").to_numpy(dtype=float)
            sc = ax.scatter(x, y, c=c, s=10, marker="o", cmap="viridis")
            fig.colorbar(sc, ax=ax, label=csel)
            ax.set_title(f"X–Y scatter: {ycol} vs {xcol}  (kleur = {csel})")
        else:
            ax.scatter(x, y, s=10, marker="o")
            ax.set_title(f"X–Y scatter: {ycol} vs {xcol}")

        ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        self._current_fig = fig
        plt.show()

    # ---- export/save ----
    def export_slice_csv(self):
        df2 = self.get_slice()
        if df2.empty: return
        out = filedialog.asksaveasfilename(
            title="Bewaar slice als CSV", defaultextension=".csv",
            filetypes=[("CSV","*.csv")], initialdir=self._initialdir(),
            initialfile="vector_slice.csv"
        )
        if not out: return
        try:
            df2.to_csv(out, index=True, date_format="%Y-%m-%d %H:%M:%S")
            messagebox.showinfo("OK", f"CSV opgeslagen:\n{out}")
        except Exception as e:
            messagebox.showerror("Export fout", str(e))

    def save_current_figure(self):
        if self._current_fig is None:
            messagebox.showwarning("Geen figuur", "Maak eerst een plot."); return
        out = filedialog.asksaveasfilename(
            title="Bewaar figuur (PNG)", defaultextension=".png",
            filetypes=[("PNG","*.png")], initialdir=self._initialdir(),
            initialfile="vector_plot.png"
        )
        if not out: return
        try:
            self._current_fig.savefig(out, dpi=150, bbox_inches="tight")
            messagebox.showinfo("OK", f"Figuur opgeslagen:\n{out}")
        except Exception as e:
            messagebox.showerror("Opslaan fout", str(e))

# ---------------- Main ----------------
if __name__ == "__main__":
    app = VectorOnlyApp()
    app.mainloop()
