#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M9 (.mat) + RTK (.mxl / MAXML) koppelen op tijd, M9 bottom-track (relatieve XY)
georefereren naar absolute RTK Lambert72, boothoogte corrigeren,
bodemhoogte afleiden, spikes filteren (ondieptes), kaart (HTML) maken,
én optioneel een nieuwe .mat wegschrijven met de aangepaste (absolute) coördinaten
waarbij de bestaande MAT-structuur behouden blijft.

Kern:
- Tijdmatching: nearest binnen tolerantie (s)
- Optie: 1 uur aftrekken van M9 tijd
- Offset: RTK_xy - M9_rel_xy -> forward fill naar M9
- boot_H = RTK_H - h_sub + h_add
- bed_H_raw = boat_H - depth_raw
- Filtering: Hampel (rolling median + MAD) op bed_H_raw (standaard)
  Outliers -> vervangen door rolling median; daarna depth_clean = boat_H - bed_H_clean
- Rapport: robuuste bodemvariatie = P95-P05(bed_H_clean)

Outputs:
- <out>.csv                 : merged + clean + flags + lat/lon (+ m9_row)
- <out>_offsets.csv         : offsets per RTK-anchor
- <out>_map.html            : folium kaart (Lambert72→WGS84)
- <out>_georef.mat (optie)  : originele MAT met Summary.Track geüpdatet naar Lambert72 (E,N)

Opmerking bij .mat wegschrijven:
- Scipy kan geen MATLAB v7.3 (HDF5) schrijven. Als je input .mat v7.3 is, zal
  savemat falen. In dat geval moet je exporteren via MATLAB of h5py-werkflow.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import scipy.io as sio
import xml.etree.ElementTree as ET

from pyproj import Transformer
import folium


# -----------------------------
# IO helpers
# -----------------------------
def _m9_time_to_utc(seconds_since_2000: np.ndarray) -> pd.DatetimeIndex:
    base = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    td = pd.to_timedelta(np.asarray(seconds_since_2000, dtype=float).reshape(-1), unit="s")
    return (pd.Timestamp(base) + td).tz_convert("UTC")


def read_m9_mat(mat_path: str | Path, subtract_one_hour: bool = True) -> pd.DataFrame:
    """
    Lees M9 .mat en haal System.Time + Summary.Track (+ optioneel Summary.Depth).
    Bewaart m9_row zodat we later exact dezelfde rijen terug kunnen wegschrijven.
    """
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "System" not in mat or "Summary" not in mat:
        raise ValueError("MAT mist System/Summary.")
    sys_ = mat["System"]
    summ = mat["Summary"]

    t = np.asarray(sys_.Time).astype(float).reshape(-1)
    time_utc = _m9_time_to_utc(t)
    if subtract_one_hour:
        time_utc = time_utc - pd.Timedelta(hours=1)

    track = np.asarray(summ.Track).astype(float)
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError(f"Summary.Track vorm onverwacht: {track.shape}")

    n = len(time_utc)
    if track.shape[0] != n:
        raise ValueError(f"Lengte mismatch: System.Time={n}, Summary.Track={track.shape}")

    df = pd.DataFrame({
        "m9_row": np.arange(n, dtype=int),
        "time_utc": time_utc,
        "m9_E_rel_m": track[:, 0],
        "m9_N_rel_m": track[:, 1],
    })

    if hasattr(summ, "Depth"):
        depth = np.asarray(summ.Depth).astype(float).reshape(-1)
        if len(depth) == len(df):
            df["m9_depth_raw_m"] = depth

    # merge_asof heeft sortering nodig
    df["t"] = df["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("t").reset_index(drop=True)


def extract_type_solutions(mxl_path: str | Path) -> list[str]:
    """Lees unieke <TypeSolution> waarden uit een .mxl (MAXML), gesorteerd."""
    try:
        tree = ET.parse(str(mxl_path))
        root = tree.getroot()
        ns_uri = root.tag[root.tag.find('{') + 1:root.tag.find('}')] if root.tag.startswith('{') else ""
        ns = "{" + ns_uri + "}" if ns_uri else ""

        obs = root.find(f"{ns}ObservationSession")
        if obs is None:
            return []

        vals: list[str] = []
        for gp in obs.findall(f"{ns}GPSPosition"):
            v = gp.findtext(f"{ns}TypeSolution")
            if v:
                v = str(v).strip()
                if v:
                    vals.append(v)
        return sorted(set(vals))
    except Exception:
        return []


def read_rtk_maxml(mxl_path: str | Path) -> pd.DataFrame:
    tree = ET.parse(str(mxl_path))
    root = tree.getroot()

    ns_uri = root.tag[root.tag.find('{') + 1:root.tag.find('}')] if root.tag.startswith('{') else ""
    ns = "{" + ns_uri + "}" if ns_uri else ""

    obs = root.find(f"{ns}ObservationSession")
    if obs is None:
        raise ValueError("MAXML: ObservationSession ontbreekt.")

    rows = []
    for gp in obs.findall(f"{ns}GPSPosition"):
        t_str = gp.findtext(f"{ns}Time")
        if not t_str:
            continue
        t = dt.datetime.fromisoformat(t_str.replace("Z", "+00:00")).astimezone(dt.timezone.utc)

        neh = gp.find(f"{ns}PlaneStation/{ns}Position/{ns}NEH")
        if neh is None:
            continue

        N = float(neh.findtext(f"{ns}North"))
        E = float(neh.findtext(f"{ns}East"))
        H = float(neh.findtext(f"{ns}Height"))
        sol = (gp.findtext(f"{ns}TypeSolution") or "").strip()
        rows.append((pd.Timestamp(t), E, N, H, sol))

    df = pd.DataFrame(rows, columns=["time_utc", "rtk_E_m", "rtk_N_m", "rtk_H_m", "rtk_solution"])
    if df.empty:
        raise ValueError("Geen RTK posities (NEH) gevonden.")
    df["t"] = df["time_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("t").reset_index(drop=True)


# -----------------------------
# Filtering
# -----------------------------
def hampel(series: pd.Series, window: int = 31, n_sigmas: float = 4.0) -> tuple[pd.Series, pd.Series]:
    x = series.astype(float)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1

    med = x.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    sigma = 1.4826 * mad
    thr = n_sigmas * sigma

    is_out = (abs_dev > thr) & (sigma > 0)
    cleaned = x.copy()
    cleaned[is_out] = med[is_out]
    return cleaned, is_out.fillna(False)


# -----------------------------
# Coupling
# -----------------------------
def couple(
    m9: pd.DataFrame,
    rtk: pd.DataFrame,
    tol_s: float = 2.0,
    h_sub: float = 2.0,
    h_add: float = 0.47,
    do_filter: bool = True,
    window: int = 31,
    sigmas: float = 4.0,
    filter_on: str = "bed",  # "bed" | "depth" | "both"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # offsets op RTK epochs (nearest M9)
    rtk_m9 = pd.merge_asof(
        rtk[["t", "rtk_E_m", "rtk_N_m", "rtk_H_m", "rtk_solution"]],
        m9[["t", "m9_E_rel_m", "m9_N_rel_m"]],
        on="t",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tol_s)),
    ).dropna(subset=["m9_E_rel_m", "m9_N_rel_m"])

    if rtk_m9.empty:
        raise ValueError("Geen RTK–M9 overlap binnen tolerantietijd (check uurcorrectie/tol).")

    rtk_m9["offset_E_m"] = rtk_m9["rtk_E_m"] - rtk_m9["m9_E_rel_m"]
    rtk_m9["offset_N_m"] = rtk_m9["rtk_N_m"] - rtk_m9["m9_N_rel_m"]

    offsets = rtk_m9[["t", "rtk_E_m", "rtk_N_m", "rtk_H_m", "rtk_solution", "offset_E_m", "offset_N_m"]].copy()
    offsets.rename(columns={"t": "rtk_anchor_time"}, inplace=True)
    offsets.sort_values("rtk_anchor_time", inplace=True)
    offsets.reset_index(drop=True, inplace=True)

    merged = pd.merge_asof(
        m9,
        offsets,
        left_on="t",
        right_on="rtk_anchor_time",
        direction="backward",
    )

    # voor M9 vóór eerste RTK: eerste offset
    first = offsets.iloc[0]
    for col in ["rtk_E_m", "rtk_N_m", "rtk_H_m", "rtk_solution", "offset_E_m", "offset_N_m", "rtk_anchor_time"]:
        merged[col] = merged[col].fillna(first[col])

    merged["m9_E_abs_m"] = merged["m9_E_rel_m"] + merged["offset_E_m"]
    merged["m9_N_abs_m"] = merged["m9_N_rel_m"] + merged["offset_N_m"]

    merged["boat_H_m"] = merged["rtk_H_m"] - float(h_sub) + float(h_add)

    stats: dict[str, float | int] = {}

    if "m9_depth_raw_m" in merged.columns:
        merged["bed_H_raw_m"] = merged["boat_H_m"] - merged["m9_depth_raw_m"]

        if do_filter:
            d_clean, d_sp = hampel(merged["m9_depth_raw_m"], window=window, n_sigmas=sigmas)
            b_clean, b_sp = hampel(merged["bed_H_raw_m"], window=window, n_sigmas=sigmas)

            if filter_on == "depth":
                merged["m9_depth_clean_m"] = d_clean
                merged["bed_H_clean_m"] = merged["boat_H_m"] - merged["m9_depth_clean_m"]
                merged["any_spike"] = d_sp
            elif filter_on == "both":
                combo = d_sp | b_sp
                b2 = merged["bed_H_raw_m"].copy()
                b2[combo] = b_clean[combo]
                merged["bed_H_clean_m"] = b2
                merged["m9_depth_clean_m"] = merged["boat_H_m"] - merged["bed_H_clean_m"]
                merged["any_spike"] = combo
            else:  # bed
                merged["bed_H_clean_m"] = b_clean
                merged["m9_depth_clean_m"] = merged["boat_H_m"] - merged["bed_H_clean_m"]
                merged["any_spike"] = b_sp
        else:
            merged["m9_depth_clean_m"] = merged["m9_depth_raw_m"]
            merged["bed_H_clean_m"] = merged["bed_H_raw_m"]
            merged["any_spike"] = False

        bedc = merged["bed_H_clean_m"].dropna()
        if len(bedc) > 20:
            stats["bed_p95_minus_p05_m"] = float(bedc.quantile(0.95) - bedc.quantile(0.05))

        dr = merged["m9_depth_raw_m"].dropna().to_numpy()
        med = float(np.median(dr))
        mad = float(np.median(np.abs(dr - med)))
        stats["depth_auto_max_m"] = med + 6.0 * 1.4826 * mad
        stats["spike_count"] = int(pd.Series(merged["any_spike"]).sum())

    merged["rtk_dt_s"] = (merged["t"] - merged["rtk_anchor_time"]).dt.total_seconds()
    merged["time_utc"] = merged["t"].dt.tz_localize("UTC")

    # WGS84 voor kaart
    tr = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(merged["m9_E_abs_m"].to_numpy(), merged["m9_N_abs_m"].to_numpy())
    merged["lon"] = lon
    merged["lat"] = lat

    # Outputkolommen (incl. m9_row voor MAT-wegschrijven)
    cols = [
        "m9_row",
        "time_utc",
        "m9_E_abs_m", "m9_N_abs_m",
        "boat_H_m",
        "m9_depth_raw_m", "m9_depth_clean_m",
        "bed_H_raw_m", "bed_H_clean_m",
        "any_spike",
        "offset_E_m", "offset_N_m",
        "rtk_solution", "rtk_anchor_time", "rtk_dt_s",
        "lat", "lon",
    ]
    cols = [c for c in cols if c in merged.columns]
    merged_out = merged[cols].copy()
    merged_out.sort_values("time_utc", inplace=True)
    merged_out.reset_index(drop=True, inplace=True)

    offsets_out = offsets.copy()
    offsets_out["rtk_anchor_time"] = offsets_out["rtk_anchor_time"].dt.tz_localize("UTC")
    return merged_out, offsets_out, stats


# -----------------------------
# Map + CSV
# -----------------------------
def make_map(df: pd.DataFrame, out_html: str | Path, every_n: int = 5) -> None:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    center = [float(df["lat"].median()), float(df["lon"].median())]
    m = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap")
    coords = df[["lat", "lon"]].astype(float).values.tolist()
    folium.PolyLine(coords, weight=4, opacity=0.9).add_to(m)

    s = df.iloc[0]; e = df.iloc[-1]
    folium.Marker([float(s.lat), float(s.lon)], popup="Start").add_to(m)
    folium.Marker([float(e.lat), float(e.lon)], popup="Einde").add_to(m)

    ds = df.iloc[::max(1, int(every_n))].copy()
    for _, r in ds.iterrows():
        popup = (
            f"{r['time_utc']}<br>"
            f"E31370={r.get('m9_E_abs_m', float('nan')):.3f} m<br>"
            f"N31370={r.get('m9_N_abs_m', float('nan')):.3f} m<br>"
            f"lat={r.get('lat', float('nan')):.7f}<br>"
            f"lon={r.get('lon', float('nan')):.7f}<br>"
            f"Depth_raw={r.get('m9_depth_raw_m', float('nan')):.2f}<br>"
            f"Depth_clean={r.get('m9_depth_clean_m', float('nan')):.2f}<br>"
            f"Bed_clean={r.get('bed_H_clean_m', float('nan')):.2f}"
        )
        color = "red" if bool(r.get("any_spike", False)) else "blue"
        folium.CircleMarker(
            [float(r.lat), float(r.lon)],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=360),
        ).add_to(m)

    m.save(str(out_html))


def save_csv(df: pd.DataFrame, out_path: str | Path, sep: str = ";") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=sep)


# -----------------------------
# MAT update (preserve structure)
# -----------------------------
def _unwrap_singleton_object_array(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return x.item()
    return x


def _ensure_2d_track(track: np.ndarray) -> np.ndarray:
    track = np.asarray(track)
    # Soms komt Track als (1, n, 2) of (n, 2, 1) binnen
    if track.ndim == 3 and 1 in track.shape:
        track = np.squeeze(track)
    if track.ndim != 2:
        raise ValueError(f"Summary.Track ndim verwacht 2, kreeg {track.ndim} met shape {track.shape}")
    return track


def save_georeferenced_mat(
    mat_in: str | Path,
    mat_out: str | Path,
    merged: pd.DataFrame,
    e_col: str = "m9_E_abs_m",
    n_col: str = "m9_N_abs_m",
) -> None:
    """
    Schrijf een nieuwe .mat file weg, met aangepaste coördinaten in Summary.Track.
    De rest van de MAT (keys/structs/arrays) blijft behouden.

    We zetten (E,N) in Lambert72 (EPSG:31370) in de eerste twee kolommen van Summary.Track.
    """
    mat_in = Path(mat_in)
    mat_out = Path(mat_out)
    mat_out.parent.mkdir(parents=True, exist_ok=True)

    if "m9_row" not in merged.columns:
        raise ValueError("merged mist 'm9_row' (nodig om Track rijen correct terug te schrijven).")

    # arrays in originele M9-volgorde
    m = merged.sort_values("m9_row")
    absE = m[e_col].to_numpy(dtype=float)
    absN = m[n_col].to_numpy(dtype=float)

    # laad met squeeze_me=False om originele dimensies zo goed mogelijk te behouden
    mat = sio.loadmat(str(mat_in), squeeze_me=False, struct_as_record=False)
    if "Summary" not in mat:
        raise ValueError("MAT mist 'Summary'.")

    summ = _unwrap_singleton_object_array(mat["Summary"])
    if summ is None or not hasattr(summ, "Track"):
        raise ValueError("Kon Summary.Track niet vinden in MAT.")

    track = _ensure_2d_track(getattr(summ, "Track"))

    # track kan (n, m) of (m, n) zijn; detecteer n
    n = len(absE)

    # probeer dtype/structuur maximaal te behouden
    track_out = np.array(track, copy=True)
    if track_out.dtype == object:
        # object -> numeriek, anders is wegschrijven toch niet stabiel
        track_out = track_out.astype(float)

    # zorg dat input in dezelfde dtype past
    absE_cast = np.asarray(absE, dtype=track_out.dtype)
    absN_cast = np.asarray(absN, dtype=track_out.dtype)

    if track_out.shape[0] == n:
        track_out[:, 0] = absE_cast
        track_out[:, 1] = absN_cast
    elif track_out.shape[1] == n and track_out.shape[0] >= 2:
        track_out[0, :] = absE_cast
        track_out[1, :] = absN_cast
    else:
        raise ValueError(
            f"Kan Summary.Track niet mappen op {n} rijen. Track shape={track.shape}. "
            "Controleer of System.Time en Summary.Track dezelfde lengte hebben."
        )

    setattr(summ, "Track", track_out)

    # verwijder meta keys (scipy voegt eigen header toe)
    mat_to_save = {k: v for k, v in mat.items() if not k.startswith("__")}
    sio.savemat(str(mat_out), mat_to_save, do_compression=True)


# -----------------------------
# GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("M9↔RTK Lambert72 + filtering + kaart + .mat")
        self.geometry("1000x690")

        self.var_m9 = tk.StringVar()
        self.var_rtk = tk.StringVar()
        self.var_out = tk.StringVar()

        self.var_subtract = tk.BooleanVar(value=True)
        self.var_tol = tk.DoubleVar(value=2.0)

        # RTK TypeSolution selectie (wordt dynamisch gevuld bij het kiezen van .mxl)
        self.sol_vars: dict[str, tk.BooleanVar] = {}

        self.var_hsub = tk.DoubleVar(value=2.0)
        self.var_hadd = tk.DoubleVar(value=0.47)

        self.var_filter = tk.BooleanVar(value=True)
        self.var_win = tk.IntVar(value=31)
        self.var_sig = tk.DoubleVar(value=4.0)
        self.var_filter_on = tk.StringVar(value="bed")

        self.var_map = tk.BooleanVar(value=True)
        self.var_every = tk.IntVar(value=5)

        # nieuw: MAT output
        self.var_save_mat = tk.BooleanVar(value=True)

        self._build()

    def _build(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="M9 .mat").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_m9, width=92).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_m9).grid(row=1, column=1, sticky="e")

        ttk.Label(frm, text="RTK .mxl (MAXML)").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=self.var_rtk, width=92).grid(row=3, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Bladeren…", command=self._pick_rtk).grid(row=3, column=1, sticky="e")

        ttk.Label(frm, text="Output basisnaam (.csv)").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frm, textvariable=self.var_out, width=92).grid(row=5, column=0, sticky="we", padx=(0, 8))
        ttk.Button(frm, text="Opslaan als…", command=self._pick_out).grid(row=5, column=1, sticky="e")

        opt = ttk.LabelFrame(frm, text="Matching", padding=10)
        opt.grid(row=6, column=0, columnspan=2, sticky="we", pady=(10, 0))

        ttk.Checkbutton(opt, text="Trek 1 uur af van M9 tijd", variable=self.var_subtract).grid(row=0, column=0, sticky="w")
        ttk.Label(opt, text="Tolerantie (s):").grid(row=0, column=1, sticky="e", padx=(20, 0))
        ttk.Entry(opt, textvariable=self.var_tol, width=10).grid(row=0, column=2, sticky="w", padx=(6, 0))

        ttk.Label(opt, text="RTK TypeSolution filter:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.sol_container = ttk.Frame(opt)
        self.sol_container.grid(row=1, column=1, columnspan=2, sticky="we", pady=(8, 0))

        self.sol_canvas = tk.Canvas(self.sol_container, height=90)
        self.sol_scroll = ttk.Scrollbar(self.sol_container, orient="vertical", command=self.sol_canvas.yview)
        self.sol_inner = ttk.Frame(self.sol_canvas)

        self.sol_inner.bind("<Configure>", lambda e: self.sol_canvas.configure(scrollregion=self.sol_canvas.bbox("all")))
        self.sol_canvas.create_window((0, 0), window=self.sol_inner, anchor="nw")
        self.sol_canvas.configure(yscrollcommand=self.sol_scroll.set)

        self.sol_canvas.pack(side="left", fill="both", expand=True)
        self.sol_scroll.pack(side="right", fill="y")

        self.sol_hint = ttk.Label(self.sol_inner, text="Kies eerst een RTK .mxl om TypeSolution opties te laden…")
        self.sol_hint.grid(row=0, column=0, sticky="w")

        self.sol_btns = ttk.Frame(opt)
        self.sol_btns.grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Button(self.sol_btns, text="Selecteer alles", command=self._solutions_select_all).pack(side="left")
        ttk.Button(self.sol_btns, text="Selecteer geen", command=self._solutions_select_none).pack(side="left", padx=(8, 0))

        hfrm = ttk.LabelFrame(frm, text="Hoogte boot", padding=10)
        hfrm.grid(row=7, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Label(hfrm, text="boat_H = RTK_H - subtract + add").grid(row=0, column=0, sticky="w")
        ttk.Label(hfrm, text="subtract (m):").grid(row=1, column=0, sticky="w")
        ttk.Entry(hfrm, textvariable=self.var_hsub, width=10).grid(row=1, column=1, sticky="w", padx=(6, 20))
        ttk.Label(hfrm, text="add (m):").grid(row=1, column=2, sticky="w")
        ttk.Entry(hfrm, textvariable=self.var_hadd, width=10).grid(row=1, column=3, sticky="w", padx=(6, 0))

        ffrm = ttk.LabelFrame(frm, text="Filtering (spikes door ondieptes)", padding=10)
        ffrm.grid(row=8, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Checkbutton(ffrm, text="Filtering aan", variable=self.var_filter).grid(row=0, column=0, sticky="w")
        ttk.Label(ffrm, text="Hampel window:").grid(row=0, column=1, sticky="e", padx=(20, 0))
        ttk.Entry(ffrm, textvariable=self.var_win, width=10).grid(row=0, column=2, sticky="w", padx=(6, 0))
        ttk.Label(ffrm, text="Sigmas:").grid(row=0, column=3, sticky="e", padx=(20, 0))
        ttk.Entry(ffrm, textvariable=self.var_sig, width=10).grid(row=0, column=4, sticky="w", padx=(6, 0))
        ttk.Label(ffrm, text="Filter op:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(ffrm, textvariable=self.var_filter_on, values=["bed", "depth", "both"], width=10, state="readonly").grid(row=1, column=1, sticky="w", pady=(8, 0))

        mfrm = ttk.LabelFrame(frm, text="Kaart + MAT", padding=10)
        mfrm.grid(row=9, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Checkbutton(mfrm, text="Maak kaart HTML", variable=self.var_map).grid(row=0, column=0, sticky="w")
        ttk.Label(mfrm, text="Toon elke n punten:").grid(row=0, column=1, sticky="e", padx=(20, 0))
        ttk.Entry(mfrm, textvariable=self.var_every, width=10).grid(row=0, column=2, sticky="w", padx=(6, 0))
        ttk.Checkbutton(mfrm, text="Schrijf nieuwe .mat met georef Summary.Track", variable=self.var_save_mat).grid(row=1, column=0, sticky="w", pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=10, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Button(btns, text="Run", command=self._run).pack(side="left")
        ttk.Button(btns, text="Sluiten", command=self.destroy).pack(side="right")

        self.txt = tk.Text(frm, height=10, wrap="word")
        self.txt.grid(row=11, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(11, weight=1)

    def _log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _pick_m9(self):
        p = filedialog.askopenfilename(title="Selecteer M9 .mat", filetypes=[("MAT", "*.mat"), ("All", "*.*")])
        if p:
            self.var_m9.set(p)
            if not self.var_out.get():
                self.var_out.set(str(Path(p).with_suffix("").as_posix()) + "_filtered.csv")

    def _solutions_select_all(self):
        for v in self.sol_vars.values():
            v.set(True)

    def _solutions_select_none(self):
        for v in self.sol_vars.values():
            v.set(False)

    def _populate_solutions(self, mxl_path: str):
        # Clear existing checkboxes
        for w in list(self.sol_inner.winfo_children()):
            w.destroy()
        self.sol_vars = {}

        sols = extract_type_solutions(mxl_path)
        if not sols:
            ttk.Label(self.sol_inner, text="Geen TypeSolution gevonden in .mxl.").grid(row=0, column=0, sticky="w")
            return

        # Maak checkbuttons (2 kolommen)
        for i, s in enumerate(sols):
            var = tk.BooleanVar(value=True)  # default: alles aan
            self.sol_vars[s] = var
            cb = ttk.Checkbutton(self.sol_inner, text=s, variable=var)
            cb.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 18))

    def _pick_rtk(self):
        p = filedialog.askopenfilename(title="Selecteer RTK .mxl", filetypes=[("MXL", "*.mxl"), ("XML", "*.xml"), ("All", "*.*")])
        if p:
            self.var_rtk.set(p)
            self._populate_solutions(p)

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="Kies output CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.var_out.set(p)

    def _run(self):
        try:
            m9p = self.var_m9.get().strip()
            rtkp = self.var_rtk.get().strip()
            outp = self.var_out.get().strip()
            if not m9p or not rtkp or not outp:
                messagebox.showerror("Input", "Kies M9, RTK en output.")
                return

            base = Path(outp).with_suffix("")
            subtract = bool(self.var_subtract.get())
            tol = float(self.var_tol.get())
            hsub = float(self.var_hsub.get())
            hadd = float(self.var_hadd.get())
            dof = bool(self.var_filter.get())
            win = int(self.var_win.get())
            sig = float(self.var_sig.get())
            fon = self.var_filter_on.get()
            dom = bool(self.var_map.get())
            every = int(self.var_every.get())
            do_mat = bool(self.var_save_mat.get())

            self._log("Lezen M9…")
            m9 = read_m9_mat(m9p, subtract_one_hour=subtract)
            self._log(f"  M9: {len(m9)} ensembles")

            self._log("Lezen RTK…")
            rtk = read_rtk_maxml(rtkp)
            self._log(f"  RTK: {len(rtk)} punten")

            # RTK TypeSolution filter (enkel geselecteerde oplossingen)
            if self.sol_vars:
                allowed = [k for k, v in self.sol_vars.items() if bool(v.get())]
                if not allowed:
                    raise ValueError("Geen TypeSolution aangevinkt. Vink minstens één kwaliteit aan.")
                before = len(rtk)
                rtk["rtk_solution"] = rtk["rtk_solution"].astype(str).str.strip()
                rtk = rtk[rtk["rtk_solution"].isin([a.strip() for a in allowed])].copy()
                self._log(f"  RTK TypeSolution filter toegepast: {len(rtk)} (van {before}) | allowed={allowed}")
                if rtk.empty:
                    raise ValueError("Na TypeSolution filter zijn er geen RTK punten meer. Vink een kwaliteit aan die voorkomt in de .mxl.")
            else:
                self._log("  RTK TypeSolution filter niet toegepast (geen opties geladen).")

            self._log("Koppelen + filtering…")
            merged, offsets, stats = couple(
                m9, rtk,
                tol_s=tol,
                h_sub=hsub,
                h_add=hadd,
                do_filter=dof,
                window=win,
                sigmas=sig,
                filter_on=fon
            )

            merged_csv = str(base) + ".csv"
            offsets_csv = str(base) + "_offsets.csv"
            save_csv(merged, merged_csv, sep=";")
            save_csv(offsets, offsets_csv, sep=";")
            self._log(f"Saved: {merged_csv}")
            self._log(f"Saved: {offsets_csv}")

            if stats:
                if "spike_count" in stats:
                    self._log(f"Spikes gefilterd: {stats['spike_count']}")
                if "bed_p95_minus_p05_m" in stats:
                    self._log(f"Bodemvariatie (P95-P05): {stats['bed_p95_minus_p05_m']:.3f} m")
                if "depth_auto_max_m" in stats:
                    self._log(f"Depth auto-max (med+6*MAD): {stats['depth_auto_max_m']:.3f} m")

            if dom:
                map_html = str(base) + "_map.html"
                make_map(merged, map_html, every_n=every)
                self._log(f"Saved: {map_html}")

            if do_mat:
                out_mat = str(base) + "_georef.mat"
                self._log("Schrijven georeferentie .mat…")
                save_georeferenced_mat(m9p, out_mat, merged)
                self._log(f"Saved: {out_mat}")

            messagebox.showinfo("OK", "Klaar.")
        except Exception as e:
            messagebox.showerror("Fout", str(e))
            self._log(f"ERROR: {e}")


# -----------------------------
# CLI
# -----------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m9", type=str)
    ap.add_argument("--rtk", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--subtract-hour", action="store_true")
    ap.add_argument("--no-subtract-hour", action="store_true")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--h-sub", type=float, default=2.0)
    ap.add_argument("--h-add", type=float, default=0.47)
    ap.add_argument("--filter", action="store_true")
    ap.add_argument("--win", type=int, default=31)
    ap.add_argument("--sig", type=float, default=4.0)
    ap.add_argument("--filter-on", type=str, default="bed", choices=["bed", "depth", "both"])
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--every-n", type=int, default=5)
    ap.add_argument("--save-mat", action="store_true", help="Schrijf <out>_georef.mat met aangepaste Summary.Track")
    ap.add_argument("--nogui", action="store_true")
    args = ap.parse_args(argv)

    if not args.nogui and (args.m9 is None or args.rtk is None or args.out is None):
        App().mainloop()
        return 0

    if args.m9 is None or args.rtk is None or args.out is None:
        ap.error("--m9 --rtk --out verplicht in CLI")

    subtract = True
    if args.no_subtract_hour:
        subtract = False
    if args.subtract_hour:
        subtract = True

    m9 = read_m9_mat(args.m9, subtract_one_hour=subtract)
    rtk = read_rtk_maxml(args.rtk)

    merged, offsets, stats = couple(
        m9, rtk,
        tol_s=args.tol,
        h_sub=args.h_sub,
        h_add=args.h_add,
        do_filter=args.filter,
        window=args.win,
        sigmas=args.sig,
        filter_on=args.filter_on
    )

    base = Path(args.out).with_suffix("")
    save_csv(merged, str(base) + ".csv", sep=";")
    save_csv(offsets, str(base) + "_offsets.csv", sep=";")
    if args.map:
        make_map(merged, str(base) + "_map.html", every_n=args.every_n)
    if args.save_mat:
        save_georeferenced_mat(args.m9, str(base) + "_georef.mat", merged)

    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
