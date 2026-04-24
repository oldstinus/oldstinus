#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vergelijk RTK Sokkia (.mxl MAXML) met RTK Carlson (CSV/TXT):
- Match op tijd (met instelbare tijdshift per dataset + tolerantie)
- Vergelijk horizontale positie (in Lambert72) + hoogte
- Kaart (Folium) met beide punten + lijnen tussen matched paren
- Outputs: *_compare.csv, *_summary.txt, *_map.html

GUI (Tkinter):
- Laad Sokkia .mxl
- Laad Carlson CSV/TXT + kies kolommen (E,N,H en tijdkolom of date+time)
- Kies CRS per dataset (default EPSG:31370 Lambert72)
- Tijdshift: seconds voor Sokkia en Carlson (positief = tijd vooruit)
- Match tolerantie in seconden
- Opties: swap E/N voor Carlson, swap E/N voor Sokkia (voor uitzonderingen)
"""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer

import folium

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -----------------------------
# CRS helpers
# -----------------------------

CRS_OPTIONS = [
    ("Lambert72 (EPSG:31370)", "EPSG:31370"),
    ("ETRS89 / UTM 31N (EPSG:25831)", "EPSG:25831"),
    ("WGS84 / UTM 31N (EPSG:32631)", "EPSG:32631"),
    ("WGS84 lat/lon (EPSG:4326)", "EPSG:4326"),
]


def to_lambert72(E: np.ndarray, N: np.ndarray, crs_in: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (E_l72, N_l72) in EPSG:31370."""
    if crs_in == "EPSG:31370":
        return E, N
    if crs_in == "EPSG:4326":
        # E=lon, N=lat
        tr = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
        e, n = tr.transform(E, N)
        return np.asarray(e, float), np.asarray(n, float)
    tr = Transformer.from_crs(crs_in, "EPSG:31370", always_xy=True)
    e, n = tr.transform(E, N)
    return np.asarray(e, float), np.asarray(n, float)


def lambert72_to_wgs84(E: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tr = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(E, N)
    return np.asarray(lat, float), np.asarray(lon, float)


# -----------------------------
# Sokkia .mxl reader (streaming)
# -----------------------------

def read_sokkia_mxl(mxl_path: str | Path) -> pd.DataFrame:
    """
    Extract per GPSPosition:
      - time_utc (timezone aware UTC)
      - E, N, H (PlaneStation/Position/NEH: East/North/Height)
      - TypeSolution
      - StationName
    """
    mxl_path = Path(mxl_path)
    rows = []

    # Streaming parse to avoid huge memory
    for event, elem in ET.iterparse(str(mxl_path), events=("end",)):
        if not elem.tag.endswith("GPSPosition"):
            continue

        t = elem.findtext("./{*}Time")
        sol = elem.findtext("./{*}TypeSolution") or ""
        name = elem.findtext("./{*}Station/{*}Name") or ""

        north = elem.findtext("./{*}PlaneStation/{*}Position/{*}NEH/{*}North")
        east  = elem.findtext("./{*}PlaneStation/{*}Position/{*}NEH/{*}East")
        h     = elem.findtext("./{*}PlaneStation/{*}Position/{*}NEH/{*}Height")

        if t and north and east:
            rows.append((t.strip(), float(east), float(north), float(h) if h else np.nan, sol.strip(), name.strip()))

        # free memory
        elem.clear()

    df = pd.DataFrame(rows, columns=["time_raw", "E", "N", "H", "TypeSolution", "Station"])
    if df.empty:
        return df

    # Parse ISO8601; most Sokkia times are like 2025-12-10T07:59:04Z
    df["time_utc"] = pd.to_datetime(df["time_raw"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    return df


# -----------------------------
# Carlson CSV/TXT reader
# -----------------------------

def sniff_delimiter(path: str | Path, default=";") -> str:
    sample = Path(path).read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return default


def read_carlson_table(path: str | Path, delimiter: str, has_header: bool) -> pd.DataFrame:
    path = Path(path)
    if delimiter == "auto":
        delimiter = sniff_delimiter(path)
    if delimiter == "\\t":
        delimiter = "\t"
    if has_header:
        df = pd.read_csv(path, sep=delimiter, engine="python")
    else:
        df = pd.read_csv(path, sep=delimiter, header=None, engine="python")
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_time_from_columns(df: pd.DataFrame, col_dt: str, col_date: str | None, col_time: str | None) -> pd.Series:
    if col_dt and col_dt != "(none)":
        return pd.to_datetime(df[col_dt], utc=True, errors="coerce")
    if col_date and col_time and col_date != "(none)" and col_time != "(none)":
        combined = df[col_date].astype(str).str.strip() + " " + df[col_time].astype(str).str.strip()
        # We treat as UTC by default (user can shift afterwards)
        return pd.to_datetime(combined, utc=True, errors="coerce")
    return pd.Series([pd.NaT] * len(df))


# -----------------------------
# Matching & metrics
# -----------------------------

@dataclass
class MatchConfig:
    sokkia_shift_s: float = 0.0
    carlson_shift_s: float = 0.0
    tol_s: float = 2.0
    sokkia_crs: str = "EPSG:31370"
    carlson_crs: str = "EPSG:31370"
    swap_sokkia_en: bool = False
    swap_carlson_en: bool = False


def match_by_time(sokkia: pd.DataFrame, carlson: pd.DataFrame, cfg: MatchConfig) -> pd.DataFrame:
    """
    Nearest match per Sokkia record (merge_asof), within tolerance.
    Returns a table with both coords and deltas.
    """
    if sokkia.empty or carlson.empty:
        return pd.DataFrame()

    s = sokkia.copy()
    c = carlson.copy()

    # Shifts
    s["t"] = s["time_utc"] + pd.to_timedelta(cfg.sokkia_shift_s, unit="s")
    c["t"] = c["time_utc"] + pd.to_timedelta(cfg.carlson_shift_s, unit="s")

    # Swap if required
    if cfg.swap_sokkia_en:
        s[["E","N"]] = s[["N","E"]].values
    if cfg.swap_carlson_en:
        c[["E","N"]] = c[["N","E"]].values

    # Ensure numeric
    for col in ["E","N","H"]:
        if col in s.columns:
            s[col] = pd.to_numeric(s[col], errors="coerce")
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")

    s = s.dropna(subset=["t","E","N"]).sort_values("t").reset_index(drop=True)
    c = c.dropna(subset=["t","E","N"]).sort_values("t").reset_index(drop=True)

    # Convert both to Lambert72 for comparison
    sE, sN = to_lambert72(s["E"].to_numpy(float), s["N"].to_numpy(float), cfg.sokkia_crs)
    cE, cN = to_lambert72(c["E"].to_numpy(float), c["N"].to_numpy(float), cfg.carlson_crs)
    s["E_L72"], s["N_L72"] = sE, sN
    c["E_L72"], c["N_L72"] = cE, cN

    # Match: for each Sokkia, take nearest Carlson
    tol = pd.Timedelta(seconds=float(cfg.tol_s))
    merged = pd.merge_asof(
        s.sort_values("t"),
        c.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=tol,
        suffixes=("_sokkia", "_carlson"),
    )

    # Drop unmatched
    merged = merged.dropna(subset=["E_L72_carlson", "N_L72_carlson"]).copy()

    # dt
    merged["dt_s"] = (merged["time_utc_carlson"] - merged["time_utc_sokkia"]).dt.total_seconds()

    # deltas in Lambert72
    merged["dE_m"] = merged["E_L72_carlson"] - merged["E_L72_sokkia"]
    merged["dN_m"] = merged["N_L72_carlson"] - merged["N_L72_sokkia"]
    merged["dXY_m"] = np.sqrt(merged["dE_m"]**2 + merged["dN_m"]**2)

    # height
    if "H_sokkia" in merged.columns and "H_carlson" in merged.columns:
        merged["dH_m"] = merged["H_carlson"] - merged["H_sokkia"]
    else:
        merged["dH_m"] = np.nan

    # WGS84 for mapping
    lat_s, lon_s = lambert72_to_wgs84(merged["E_L72_sokkia"].to_numpy(float), merged["N_L72_sokkia"].to_numpy(float))
    lat_c, lon_c = lambert72_to_wgs84(merged["E_L72_carlson"].to_numpy(float), merged["N_L72_carlson"].to_numpy(float))
    merged["lat_sokkia"], merged["lon_sokkia"] = lat_s, lon_s
    merged["lat_carlson"], merged["lon_carlson"] = lat_c, lon_c

    return merged.reset_index(drop=True)


def summary_stats(m: pd.DataFrame) -> str:
    if m.empty:
        return "Geen matches."
    def fmt(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return "n=0"
        return f"n={len(s)} | mean={s.mean():.3f} | std={s.std(ddof=1):.3f} | RMS={math.sqrt((s**2).mean()):.3f} | max={s.abs().max():.3f}"
    lines = []
    lines.append(f"Matches: {len(m)}")
    lines.append(f"dE_m:  {fmt(m['dE_m'])}")
    lines.append(f"dN_m:  {fmt(m['dN_m'])}")
    lines.append(f"dXY_m: {fmt(m['dXY_m'])}")
    if "dH_m" in m.columns:
        lines.append(f"dH_m:  {fmt(m['dH_m'])}")
    lines.append(f"dt_s (Carlson - Sokkia): {fmt(m['dt_s'])}")
    return "\n".join(lines)


# -----------------------------
# Map
# -----------------------------

def make_match_map(m: pd.DataFrame, out_html: str | Path, every_n: int = 1) -> None:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    if m.empty:
        # create empty map around Belgium
        folium.Map(location=[50.85, 4.35], zoom_start=8, tiles="OpenStreetMap").save(str(out_html))
        return

    ds = m.iloc[::max(1, int(every_n))].copy()

    center = [float(ds["lat_sokkia"].median()), float(ds["lon_sokkia"].median())]
    mp = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap")

    # Draw matched pairs
    for i, r in ds.iterrows():
        # Sokkia marker
        tip_s = f"SOKKIA | L72 E={r['E_L72_sokkia']:.3f} N={r['N_L72_sokkia']:.3f} | WGS84 {r['lat_sokkia']:.8f},{r['lon_sokkia']:.8f}"
        pop_s = folium.Popup(
            f"<b>SOKKIA</b><br>"
            f"<b>Time</b>: {r['time_utc_sokkia']}<br>"
            f"<b>Lambert72</b> E={r['E_L72_sokkia']:.3f} N={r['N_L72_sokkia']:.3f}<br>"
            f"<b>WGS84</b> lat={r['lat_sokkia']:.8f} lon={r['lon_sokkia']:.8f}<br>"
            f"<b>H</b>: {r.get('H_sokkia', np.nan)}",
            max_width=420
        )
        folium.CircleMarker([r["lat_sokkia"], r["lon_sokkia"]], radius=4, color="blue",
                            fill=True, fill_opacity=0.9, tooltip=tip_s, popup=pop_s).add_to(mp)

        # Carlson marker
        tip_c = f"CARLSON | L72 E={r['E_L72_carlson']:.3f} N={r['N_L72_carlson']:.3f} | WGS84 {r['lat_carlson']:.8f},{r['lon_carlson']:.8f}"
        pop_c = folium.Popup(
            f"<b>CARLSON</b><br>"
            f"<b>Time</b>: {r['time_utc_carlson']}<br>"
            f"<b>Lambert72</b> E={r['E_L72_carlson']:.3f} N={r['N_L72_carlson']:.3f}<br>"
            f"<b>WGS84</b> lat={r['lat_carlson']:.8f} lon={r['lon_carlson']:.8f}<br>"
            f"<b>H</b>: {r.get('H_carlson', np.nan)}",
            max_width=420
        )
        folium.CircleMarker([r["lat_carlson"], r["lon_carlson"]], radius=4, color="red",
                            fill=True, fill_opacity=0.9, tooltip=tip_c, popup=pop_c).add_to(mp)

        # Link line
        tip_line = f"dXY={r['dXY_m']:.3f} m | dE={r['dE_m']:.3f} m | dN={r['dN_m']:.3f} m | dH={r.get('dH_m', np.nan)} m"
        folium.PolyLine(
            locations=[[r["lat_sokkia"], r["lon_sokkia"]], [r["lat_carlson"], r["lon_carlson"]]],
            weight=2, opacity=0.9, tooltip=tip_line, color="green"
        ).add_to(mp)

    mp.save(str(out_html))


# -----------------------------
# GUI
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vergelijk RTK: Sokkia (.mxl) vs Carlson (CSV/TXT)")
        self.geometry("1180x760")

        self.var_mxl = tk.StringVar()
        self.var_carlson = tk.StringVar()
        self.var_outbase = tk.StringVar()

        self.var_c_delim = tk.StringVar(value="auto")
        self.var_c_header = tk.BooleanVar(value=True)

        self.var_s_crs = tk.StringVar(value="EPSG:31370")
        self.var_c_crs = tk.StringVar(value="EPSG:31370")

        self.var_shift_s = tk.DoubleVar(value=0.0)
        self.var_shift_c = tk.DoubleVar(value=0.0)
        self.var_tol = tk.DoubleVar(value=2.0)

        self.var_swap_s = tk.BooleanVar(value=False)
        self.var_swap_c = tk.BooleanVar(value=False)

        self.var_map_every = tk.IntVar(value=1)

        # Carlson columns
        self.var_colE = tk.StringVar()
        self.var_colN = tk.StringVar()
        self.var_colH = tk.StringVar(value="(none)")
        self.var_colDT = tk.StringVar(value="(none)")
        self.var_colDate = tk.StringVar(value="(none)")
        self.var_colTime = tk.StringVar(value="(none)")

        self.df_s: Optional[pd.DataFrame] = None
        self.df_c_raw: Optional[pd.DataFrame] = None
        self.df_c: Optional[pd.DataFrame] = None
        self.df_match: Optional[pd.DataFrame] = None

        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        # Files
        ttk.Label(frm, text="Sokkia RTK (.mxl)").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_mxl, width=100).grid(row=1, column=0, sticky="we", padx=(0,8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_mxl).grid(row=1, column=1, sticky="e")

        ttk.Label(frm, text="Carlson RTK (CSV/TXT)").grid(row=2, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.var_carlson, width=100).grid(row=3, column=0, sticky="we", padx=(0,8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_carlson).grid(row=3, column=1, sticky="e")

        ttk.Label(frm, text="Output basisnaam (zonder extensie)").grid(row=4, column=0, sticky="w", pady=(10,0))
        ttk.Entry(frm, textvariable=self.var_outbase, width=100).grid(row=5, column=0, sticky="we", padx=(0,8))
        ttk.Button(frm, text="Kies…", command=self._pick_outbase).grid(row=5, column=1, sticky="e")

        # Options
        opt = ttk.LabelFrame(frm, text="Opties", padding=10)
        opt.grid(row=6, column=0, columnspan=2, sticky="we", pady=(12,0))

        ttk.Label(opt, text="Sokkia CRS:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opt, textvariable=self.var_s_crs, values=[c[1] for c in CRS_OPTIONS], width=18, state="readonly").grid(row=0, column=1, sticky="w", padx=(6,0))
        ttk.Checkbutton(opt, text="swap E/N (Sokkia)", variable=self.var_swap_s).grid(row=0, column=2, sticky="w", padx=(16,0))

        ttk.Label(opt, text="Carlson CRS:").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Combobox(opt, textvariable=self.var_c_crs, values=[c[1] for c in CRS_OPTIONS], width=18, state="readonly").grid(row=1, column=1, sticky="w", padx=(6,0), pady=(8,0))
        ttk.Checkbutton(opt, text="swap E/N (Carlson)", variable=self.var_swap_c).grid(row=1, column=2, sticky="w", padx=(16,0), pady=(8,0))

        ttk.Label(opt, text="Tijdshift Sokkia (s):").grid(row=2, column=0, sticky="w", pady=(8,0))
        ttk.Entry(opt, textvariable=self.var_shift_s, width=10).grid(row=2, column=1, sticky="w", padx=(6,0), pady=(8,0))
        ttk.Label(opt, text="Tijdshift Carlson (s):").grid(row=2, column=2, sticky="w", padx=(16,0), pady=(8,0))
        ttk.Entry(opt, textvariable=self.var_shift_c, width=10).grid(row=2, column=3, sticky="w", padx=(6,0), pady=(8,0))

        ttk.Label(opt, text="Match tolerantie (s):").grid(row=3, column=0, sticky="w", pady=(8,0))
        ttk.Entry(opt, textvariable=self.var_tol, width=10).grid(row=3, column=1, sticky="w", padx=(6,0), pady=(8,0))

        ttk.Label(opt, text="Kaart: toon elke n matches:").grid(row=3, column=2, sticky="w", padx=(16,0), pady=(8,0))
        ttk.Entry(opt, textvariable=self.var_map_every, width=10).grid(row=3, column=3, sticky="w", padx=(6,0), pady=(8,0))

        # Carlson reading
        copt = ttk.LabelFrame(frm, text="Carlson inlezen", padding=10)
        copt.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10,0))

        ttk.Label(copt, text="Delimiter:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(copt, textvariable=self.var_c_delim, values=["auto", ";", ",", "\\t", "|"], width=8, state="readonly").grid(row=0, column=1, sticky="w", padx=(6,0))
        ttk.Checkbutton(copt, text="Bestand heeft header", variable=self.var_c_header).grid(row=0, column=2, sticky="w", padx=(16,0))
        ttk.Button(copt, text="Lees Carlson + kies kolommen", command=self._load_carlson).grid(row=0, column=3, sticky="w", padx=(16,0))

        cols = ttk.LabelFrame(frm, text="Carlson kolommen", padding=10)
        cols.grid(row=8, column=0, columnspan=2, sticky="we", pady=(10,0))

        ttk.Label(cols, text="Easting (E):").grid(row=0, column=0, sticky="w")
        self.cbE = ttk.Combobox(cols, textvariable=self.var_colE, values=[], width=25, state="readonly")
        self.cbE.grid(row=0, column=1, sticky="w", padx=(6,0))

        ttk.Label(cols, text="Northing (N):").grid(row=0, column=2, sticky="w", padx=(16,0))
        self.cbN = ttk.Combobox(cols, textvariable=self.var_colN, values=[], width=25, state="readonly")
        self.cbN.grid(row=0, column=3, sticky="w", padx=(6,0))

        ttk.Label(cols, text="Hoogte H (optioneel):").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.cbH = ttk.Combobox(cols, textvariable=self.var_colH, values=["(none)"], width=25, state="readonly")
        self.cbH.grid(row=1, column=1, sticky="w", padx=(6,0), pady=(8,0))

        ttk.Label(cols, text="Datetime kolom (optioneel):").grid(row=1, column=2, sticky="w", padx=(16,0), pady=(8,0))
        self.cbDT = ttk.Combobox(cols, textvariable=self.var_colDT, values=["(none)"], width=25, state="readonly")
        self.cbDT.grid(row=1, column=3, sticky="w", padx=(6,0), pady=(8,0))

        ttk.Label(cols, text="Of: Date kolom:").grid(row=2, column=0, sticky="w", pady=(8,0))
        self.cbDate = ttk.Combobox(cols, textvariable=self.var_colDate, values=["(none)"], width=25, state="readonly")
        self.cbDate.grid(row=2, column=1, sticky="w", padx=(6,0), pady=(8,0))

        ttk.Label(cols, text="Time kolom:").grid(row=2, column=2, sticky="w", padx=(16,0), pady=(8,0))
        self.cbTime = ttk.Combobox(cols, textvariable=self.var_colTime, values=["(none)"], width=25, state="readonly")
        self.cbTime.grid(row=2, column=3, sticky="w", padx=(6,0), pady=(8,0))

        # Actions
        act = ttk.Frame(frm)
        act.grid(row=9, column=0, columnspan=2, sticky="we", pady=(12,0))
        ttk.Button(act, text="Run vergelijking + outputs", command=self._run).pack(side="left")
        ttk.Button(act, text="Maak kaart (HTML)", command=self._make_map_only).pack(side="left", padx=(8,0))
        ttk.Button(act, text="Sluiten", command=self.destroy).pack(side="right")

        # Log
        self.txt = tk.Text(frm, height=14, wrap="word")
        self.txt.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=(12,0))
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(10, weight=1)

    def _log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _pick_mxl(self):
        p = filedialog.askopenfilename(title="Selecteer Sokkia .mxl", filetypes=[("MAXML", "*.mxl"), ("All", "*.*")])
        if p:
            self.var_mxl.set(p)
            if not self.var_outbase.get():
                self.var_outbase.set(str(Path(p).with_suffix("")) + "_vs_carlson")

    def _pick_carlson(self):
        p = filedialog.askopenfilename(title="Selecteer Carlson CSV/TXT", filetypes=[("Text/CSV", "*.txt *.csv *.dat *.log"), ("All", "*.*")])
        if p:
            self.var_carlson.set(p)
            if not self.var_outbase.get():
                self.var_outbase.set(str(Path(p).with_suffix("")) + "_vs_sokkia")

    def _pick_outbase(self):
        p = filedialog.asksaveasfilename(title="Kies output basisnaam", defaultextension="")
        if p:
            self.var_outbase.set(str(Path(p).with_suffix("")))

    def _load_carlson(self):
        try:
            fp = self.var_carlson.get().strip()
            if not fp:
                raise ValueError("Kies eerst een Carlson bestand.")
            delim = self.var_c_delim.get()
            has_header = bool(self.var_c_header.get())
            self._log("Lezen Carlson…")
            df = read_carlson_table(fp, delimiter=delim, has_header=has_header)
            self.df_c_raw = df
            cols = list(df.columns)
            self._log(f"  Rijen: {len(df)} | kolommen: {len(cols)}")

            self.cbE["values"] = cols
            self.cbN["values"] = cols
            self.cbH["values"] = ["(none)"] + cols
            self.cbDT["values"] = ["(none)"] + cols
            self.cbDate["values"] = ["(none)"] + cols
            self.cbTime["values"] = ["(none)"] + cols

            # naive guesses
            low = {c.lower(): c for c in cols}
            def pick(cands):
                for c in cands:
                    if c.lower() in low:
                        return low[c.lower()]
                return ""

            e_guess = pick(["E","East","Easting","Oost","X"])
            n_guess = pick(["N","North","Northing","Noord","Y"])
            h_guess = pick(["H","Height","Z","EllH","OrthoH","hoogte"])
            dt_guess = pick(["time_utc","time","Time","Datetime","DateTime","timestamp","Timestamp"])

            if e_guess: self.var_colE.set(e_guess)
            if n_guess: self.var_colN.set(n_guess)
            self.var_colH.set(h_guess if h_guess else "(none)")
            self.var_colDT.set(dt_guess if dt_guess else "(none)")
            # date/time guesses
            date_guess = pick(["date","Date"])
            time_guess = pick(["time","Time"])
            if date_guess and self.var_colDT.get() == "(none)":
                self.var_colDate.set(date_guess)
            if time_guess and self.var_colDT.get() == "(none)":
                self.var_colTime.set(time_guess)

            messagebox.showinfo("OK", "Carlson ingelezen. Controleer kolomkeuzes.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _prepare(self):
        mxl = self.var_mxl.get().strip()
        carl = self.var_carlson.get().strip()
        outb = self.var_outbase.get().strip()
        if not mxl or not carl or not outb:
            raise ValueError("Kies Sokkia .mxl, Carlson bestand en output basisnaam.")
        outb = str(Path(outb))

        # Read sokkia
        self._log("Lezen Sokkia .mxl…")
        s = read_sokkia_mxl(mxl)
        if s.empty:
            raise ValueError("Geen Sokkia RTK punten gevonden in .mxl.")
        self._log(f"  Sokkia punten: {len(s)} | {s.time_utc.min()} → {s.time_utc.max()}")
        self.df_s = s

        # Read carlson (if not yet)
        if self.df_c_raw is None:
            self._load_carlson()
        if self.df_c_raw is None:
            raise ValueError("Carlson data niet geladen.")
        df = self.df_c_raw.copy()

        colE = self.var_colE.get().strip()
        colN = self.var_colN.get().strip()
        if not colE or not colN:
            raise ValueError("Selecteer Carlson kolommen voor E en N.")

        df["E"] = pd.to_numeric(df[colE], errors="coerce")
        df["N"] = pd.to_numeric(df[colN], errors="coerce")

        colH = self.var_colH.get().strip()
        df["H"] = pd.to_numeric(df[colH], errors="coerce") if (colH and colH != "(none)") else np.nan

        colDT = self.var_colDT.get().strip()
        colDate = self.var_colDate.get().strip()
        colTime = self.var_colTime.get().strip()
        df["time_utc"] = parse_time_from_columns(df, col_dt=colDT, col_date=(colDate if colDate != "(none)" else None), col_time=(colTime if colTime != "(none)" else None))
        df = df.dropna(subset=["time_utc"]).copy()
        if df.empty:
            raise ValueError("Geen geldige timestamps gevonden in Carlson data. Kies een datetime-kolom of date+time.")

        df = df.sort_values("time_utc").reset_index(drop=True)
        self._log(f"  Carlson punten: {len(df)} | {df.time_utc.min()} → {df.time_utc.max()}")
        self.df_c = df

        return outb

    def _run(self):
        try:
            outb = self._prepare()

            cfg = MatchConfig(
                sokkia_shift_s=float(self.var_shift_s.get()),
                carlson_shift_s=float(self.var_shift_c.get()),
                tol_s=float(self.var_tol.get()),
                sokkia_crs=self.var_s_crs.get(),
                carlson_crs=self.var_c_crs.get(),
                swap_sokkia_en=bool(self.var_swap_s.get()),
                swap_carlson_en=bool(self.var_swap_c.get()),
            )

            self._log("Matching op tijd…")
            m = match_by_time(self.df_s, self.df_c, cfg)
            self.df_match = m
            self._log(summary_stats(m))

            out_csv = str(Path(outb).with_suffix("")) + "_compare.csv"
            out_sum = str(Path(outb).with_suffix("")) + "_summary.txt"
            out_map = str(Path(outb).with_suffix("")) + "_map.html"

            m.to_csv(out_csv, index=False, encoding="utf-8")
            Path(out_sum).write_text(summary_stats(m), encoding="utf-8")
            make_match_map(m, out_html=out_map, every_n=int(self.var_map_every.get()))

            self._log(f"Saved: {out_csv}")
            self._log(f"Saved: {out_sum}")
            self._log(f"Saved: {out_map}")

            messagebox.showinfo("Klaar", f"Vergelijking klaar.\n\n{out_csv}\n{out_sum}\n{out_map}")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")

    def _make_map_only(self):
        try:
            if self.df_match is None:
                raise ValueError("Run eerst de vergelijking zodat er matches zijn.")
            outb = self.var_outbase.get().strip()
            if not outb:
                raise ValueError("Kies een output basisnaam.")
            out_map = str(Path(outb).with_suffix("")) + "_map.html"
            make_match_map(self.df_match, out_html=out_map, every_n=int(self.var_map_every.get()))
            self._log(f"Saved: {out_map}")
            messagebox.showinfo("OK", f"Kaart opgeslagen:\n{out_map}")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
